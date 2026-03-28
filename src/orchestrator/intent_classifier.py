"""
Task 2.2 - 结构化输出与意图分类模块
功能：接收用户自然语言输入，通过 LLM 进行意图分类，返回结构化 JSON 结果
"""

from __future__ import annotations

import json
import os
import platform
import re
import subprocess
from typing import Any

from jsonschema import ValidationError, validate
from openai import OpenAIError

try:
    from .llm_client import (
        DEFAULT_MODEL,
        generate_text_response,
        get_llm_client,
        stream_text_response,
    )
except ImportError:
    from llm_client import (  # type: ignore
        DEFAULT_MODEL,
        generate_text_response,
        get_llm_client,
        stream_text_response,
    )


client = get_llm_client()
IntentResult = dict[str, Any]

# === 1. JSON Schema 定义（用于 jsonschema 校验）===
INTENT_JSON_SCHEMA = {
    "type": "object",
    "required": ["reasoning", "confidence", "intent", "risk_level"],
    "additionalProperties": False,
    "properties": {
        "reasoning": {"type": "string", "minLength": 1},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "intent": {
            "type": "string",
            "enum": ["shell_agent", "tool_agent", "direct_answer", "clarification"],
        },
        "risk_level": {
            "type": "string",
            "enum": ["low", "medium", "high"],
        },
        "task_description": {"type": "string", "minLength": 1},
        "context_passed": {"type": "array", "items": {"type": "string"}},
        "reply": {"type": "string", "minLength": 1},
        "question": {"type": "string", "minLength": 1},
        "options": {"type": "array", "items": {"type": "string"}},
    },
    "oneOf": [
        {
            "properties": {"intent": {"enum": ["shell_agent", "tool_agent"]}},
            "required": ["task_description", "context_passed"],
        },
        {
            "properties": {"intent": {"const": "direct_answer"}},
            "required": ["reply"],
        },
        {
            "properties": {"intent": {"const": "clarification"}},
            "required": ["question", "options"],
        },
    ],
}

# Confidence 阈值常量
CONFIDENCE_HIGH = 0.8
CONFIDENCE_LOW = 0.5


# === 2. 环境上下文收集 ===
def get_advanced_context() -> dict[str, str]:
    """获取当前操作系统、工作目录、文件列表和 Git 状态。"""
    context = {
        "os": platform.system(),
        "pwd": os.getcwd(),
        "shell": os.environ.get("SHELL", "unknown"),
        "files": "",
        "git_status": "",
    }

    try:
        ls_cmd = ["ls", "-la"] if platform.system() != "Windows" else ["dir"]
        ls_output = subprocess.check_output(ls_cmd, text=True, stderr=subprocess.DEVNULL)
        context["files"] = "\n".join(ls_output.splitlines()[:20])
    except Exception:
        context["files"] = "无法获取文件列表"

    try:
        git_output = subprocess.check_output(
            ["git", "status", "-s"], text=True, stderr=subprocess.DEVNULL
        )
        context["git_status"] = git_output.strip() or "干净的工作区 (没有未提交的更改)"
    except Exception:
        context["git_status"] = "当前不是 Git 仓库"

    return context


# === 3. System Prompt 生成 ===
def get_system_prompt(context: dict[str, str]) -> str:
    """生成带环境上下文的系统提示词，要求 LLM 输出结构化 JSON。"""
    return f"""你是一个多 Agent 命令行系统的"总控大脑"。你的唯一任务是分析用户的自然语言输入，判断其真实意图，并输出严格的结构化 JSON 进行路由。

【当前环境上下文】
- 操作系统: {context['os']}
- Shell 类型: {context['shell']}
- 工作目录: {context['pwd']}
- 目录概览 (前20项):
{context['files']}
- Git 状态:
{context['git_status']}

【意图分类标准与边界】
1. "shell_agent": 确定性的本地系统操作。
   - 边界：只需执行标准命令或管道（如 grep, awk, ls, zip）即可完成，不需要大模型的语义理解或复杂推理。
   - 示例："帮我看看当前目录有哪些文件"、"找出这段日志里的 error 行"。
2. "tool_agent": 需要认知能力、语义分析或调用外部资源的复杂任务。
   - 边界：涉及对文件内容的阅读理解、代码审查、翻译，或需要调用第三方 API。
   - 示例："读取 config.json 并总结里面的数据库配置"、"分析这段代码的复杂度"。
3. "direct_answer": 纯知识问答或闲聊，无需系统执行任何操作。
   - 示例："什么是 Python 的 GIL？"、"多 Agent 系统怎么设计？"
4. "clarification": (最高优先级) 指令模糊或存在安全隐患，必须追问。
   - 边界：存在你认为无从确定的模糊指代；或者用户要求操作的实体在【当前环境上下文】中不存在并且你无法推断其含义；或者操作极度危险但意图不明。

【评估与校准规则】
你必须进行以下两项严格评估：
- confidence (置信度 0.0~1.0): 基础分根据你的判断打在 0.8~1.0之间。
    - 特殊处理规则：若用户指令有歧义扣 0.3；若要求的操作目标在环境中找不到，立刻扣 0.4。如果最终低于 0.5，必须将意图改为 "clarification"。
- risk_level (风险等级): 评估任务对系统的破坏性。枚举值："low"(安全查询), "medium"(修改普通文件), "high"(删除、格式化、修改系统配置)。

【输出要求】
你必须且只能输出一个合法的 JSON 对象，不要有任何 Markdown 标记（如 ```json）或其他文字。
为了彻底消除“事后合理化”的幻觉，你必须严格按照以下自回归逻辑链顺序输出字段：
先输出 reasoning (思考推演) -> 再输出 confidence (严格打分) -> 接着输出 intent (根据reasoning和confidence定论) -> 然后是 risk_level -> 最后动态输出专属字段。

请根据最终确定的 intent，选择以下三种 JSON 结构之一进行输出：

>>> 如果 intent 是 "shell_agent" 或 "tool_agent"：
{{
    "reasoning": "分析用户指令 -> 检查上下文文件是否存在 -> 评估风险 -> 决定委派给下级 Agent",
    "confidence": 0.95,
    "intent": "shell_agent" | "tool_agent",
    "risk_level": "low" | "medium" | "high",
    "task_description": "将用户的话转化为专业、清晰的任务指令，提供给下级 Agent",
    "context_passed": ["提取出的文件名、路径或关键参数"]
}}

>>> 如果 intent 是 "direct_answer"：
{{
    "reasoning": "识别为纯自然语言互动，无需操作系统",
    "confidence": 1.0,
    "intent": "direct_answer",
    "risk_level": "low",
    "reply": "直接在这里写出完整的回答内容，一步到位"
}}

>>> 如果 intent 是 "clarification"：
{{
    "reasoning": "发现指令缺失关键目标（如上下文无此文件），或指代不明，触发降级策略",
    "confidence": 0.4,
    "intent": "clarification",
    "risk_level": "low",
    "question": "向用户提出礼貌的追问，例如：请问您要删除的是当前目录的 test.py 还是 src 目录下的？",
    "options": ["候选选项1", "候选选项2", "其他"]
}}
"""


# === 4. 三层容错 JSON 解析 ===
def parse_llm_json(raw_text: str) -> IntentResult | None:
    """三层容错机制解析 LLM 返回的 JSON。"""
    raw_text = raw_text.strip()

    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        pass

    try:
        if raw_text.startswith("```"):
            lines = raw_text.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            cleaned = "\n".join(lines).strip()
            return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    try:
        match = re.search(r"\{.*\}", raw_text, re.DOTALL)
        if match:
            return json.loads(match.group())
    except json.JSONDecodeError:
        pass

    return None


# === 5. Schema 校验 ===
def validate_intent_result(data: IntentResult) -> tuple[bool, str]:
    """用 jsonschema 校验解析结果是否符合预期格式。"""
    try:
        validate(instance=data, schema=INTENT_JSON_SCHEMA)
        if data["confidence"] < CONFIDENCE_LOW and data["intent"] != "clarification":
            return False, "confidence < 0.5 时 intent 必须为 clarification"
        if data["intent"] == "direct_answer" and data["risk_level"] != "low":
            return False, "direct_answer 的 risk_level 必须为 low"
        return True, ""
    except ValidationError as exc:
        return False, f"Schema 校验失败: {exc.message}"


# === 6. 降级与后处理 ===
def _make_fallback(user_input: str, raw_text: str) -> IntentResult:
    """生成降级 fallback 结果。"""
    raw_text = raw_text.strip()
    extracted_text = raw_text
    extracted_key = ""
    try:
        parsed_raw = json.loads(raw_text)
        if isinstance(parsed_raw, dict):
            for key in ("reply", "answer", "response", "message", "content", "text"):
                value = parsed_raw.get(key)
                if isinstance(value, str) and value.strip():
                    extracted_text = value.strip()
                    extracted_key = key
                    break
    except Exception:
        pass

    lowered_input = user_input.lower()
    clarification_markers = [
        "请问",
        "能否",
        "哪个",
        "哪一个",
        "更多信息",
        "请提供",
        "具体",
        "是指",
        "还是",
    ]
    direct_answer_markers = [
        "你好",
        "你是谁",
        "什么",
        "是什么",
        "为啥",
        "为什么",
        "怎么",
        "如何",
        "介绍",
        "解释",
        "是谁",
    ]

    looks_like_clarification = any(marker in raw_text for marker in clarification_markers)
    looks_like_clarification = looks_like_clarification or any(
        marker in extracted_text for marker in clarification_markers
    )
    looks_like_direct_answer = (
        bool(extracted_text)
        and (
            extracted_key in {"reply", "answer", "response", "content", "text"}
            or
            any(marker in lowered_input for marker in direct_answer_markers)
            or user_input.endswith(("?", "？"))
            or len(extracted_text) >= 24
        )
    )

    if looks_like_direct_answer and not looks_like_clarification:
        return {
            "intent": "direct_answer",
            "reasoning": f"结构化解析失败，但模型已直接给出自然语言回答，降级为 direct_answer。原始内容: {raw_text[:200]}",
            "confidence": 0.6,
            "risk_level": "low",
            "reply": extracted_text,
        }

    return {
        "intent": "clarification",
        "reasoning": f"JSON 解析/校验失败，自动降级为 clarification。原始内容: {raw_text[:200]}",
        "confidence": 0.0,
        "risk_level": "low",
        "question": (
            extracted_text
            if looks_like_clarification and extracted_text
            else f"我暂时无法可靠判断你的意图。你是希望我处理这个任务吗：{user_input}"
        ),
        "options": ["shell_agent", "tool_agent", "direct_answer", "其他"],
    }


def apply_confidence_policy(result: IntentResult) -> IntentResult:
    """根据 confidence 阈值做最终降级与补默认值。"""
    result = dict(result)
    confidence = float(result.get("confidence", 0.0))

    if confidence < CONFIDENCE_LOW:
        result["intent"] = "clarification"
        result["risk_level"] = "low"
        result.setdefault("question", "你的指令不太明确，能否提供更多细节？")
        if not isinstance(result.get("options"), list):
            result["options"] = []

    return result


# === 7. 核心意图分类 ===
def classify_intent(
    user_input: str,
    llm_client=None,
    model: str | None = None,
    max_retries: int = 1,
    verbose: bool = False,
) -> IntentResult:
    """调用 LLM 进行意图分类，返回结构化结果 dict。"""
    llm_client = llm_client or client
    model = model or DEFAULT_MODEL
    context = get_advanced_context()
    system_prompt = get_system_prompt(context)

    for attempt in range(1 + max_retries):
        try:
            raw_text = generate_text_response(
                system_prompt,
                user_input,
                llm_client=llm_client,
                model=model,
                temperature=0.1,
                json_mode=True,
            )
            if verbose:
                print(f"\n[LLM 原始返回]\n{raw_text}\n")

            parsed = parse_llm_json(raw_text)
            if parsed is None:
                if attempt < max_retries:
                    if verbose:
                        print(f"[重试 {attempt + 1}/{max_retries}] JSON 解析失败，正在重试...")
                    continue
                return _make_fallback(user_input, raw_text)

            valid, err_msg = validate_intent_result(parsed)
            if not valid:
                if attempt < max_retries:
                    if verbose:
                        print(f"[重试 {attempt + 1}/{max_retries}] {err_msg}，正在重试...")
                    continue
                return _make_fallback(user_input, raw_text)

            return parsed

        except OpenAIError as exc:
            if verbose:
                print(f"[API 错误] {exc}")
            return _make_fallback(user_input, str(exc))
        except Exception as exc:
            if verbose:
                print(f"[未知错误] {exc}")
            return _make_fallback(user_input, str(exc))

    return _make_fallback(user_input, "重试耗尽")


# === 8. 对外统一入口 ===
def handle_intent(
    user_input: str,
    llm_client=None,
    model: str | None = None,
    max_retries: int = 1,
    verbose: bool = False,
) -> IntentResult:
    """分类并应用置信度策略，返回最终可路由结果。"""
    result = classify_intent(
        user_input,
        llm_client=llm_client,
        model=model,
        max_retries=max_retries,
        verbose=verbose,
    )
    return apply_confidence_policy(result)


# === 9. direct_answer 的流式回复 ===
def stream_direct_answer(
    user_input: str,
    llm_client=None,
    model: str | None = None,
    on_delta=None,
) -> str:
    """当 direct_answer 需要完整回答时，发起流式请求。"""
    llm_client = llm_client or client
    model = model or DEFAULT_MODEL

    return stream_text_response(
        "你是一个有用的 AI 助手，请直接回答用户的问题。",
        user_input,
        llm_client=llm_client,
        model=model,
        temperature=0.7,
        on_delta=on_delta,
    )
