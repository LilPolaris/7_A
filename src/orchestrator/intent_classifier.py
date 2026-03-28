"""
Task 2.2 - 结构化输出与意图分类模块
功能：接收用户自然语言输入，通过 LLM 进行意图分类，返回结构化 JSON 结果
"""

import os
import sys
import json
import re
import platform
import subprocess
from dotenv import load_dotenv
from openai import OpenAI, APIError, AuthenticationError, RateLimitError
from jsonschema import validate, ValidationError

# === 初始化 ===
load_dotenv()
try:
    client = OpenAI()
except Exception as e:
    print(f"初始化失败，请检查 .env 文件是否配置正确: {e}")
    sys.exit(1)

# 默认模型，可在调用时覆盖
DEFAULT_MODEL = "claude-haiku-4-5-20251001"

# === 1. JSON Schema 定义（用于 jsonschema 校验）===
INTENT_JSON_SCHEMA = {
    "type": "object",
    "required": ["intent", "reasoning", "confidence", "params"],
    "properties": {
        "intent": {
            "type": "string",
            "enum": ["shell_agent", "tool_agent", "direct_answer", "clarification"]
        },
        "reasoning": {
            "type": "string",
            "minLength": 1
        },
        "confidence": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0
        },
        "params": {
            "type": "object",
            "properties": {
                "task_description": {"type": "string"},
                "suggested_tools": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "question": {"type": "string"},
                "options": {
                    "type": "array",
                    "items": {"type": "string"}
                }
            }
        },
        "fallback_response": {
            "type": "string"
        }
    }
}

# Confidence 阈值常量
CONFIDENCE_HIGH = 0.8
CONFIDENCE_LOW = 0.5


# === 2. 环境上下文收集（复用 111.py 的 get_advanced_context）===
def get_advanced_context():
    """获取当前操作系统、工作目录、文件列表和 Git 状态"""
    context = {
        "os": platform.system(),
        "pwd": os.getcwd(),
        "shell": os.environ.get("SHELL", "unknown"),
        "files": "",
        "git_status": ""
    }

    try:
        ls_cmd = ['ls', '-la'] if platform.system() != 'Windows' else ['dir']
        ls_output = subprocess.check_output(ls_cmd, text=True, stderr=subprocess.DEVNULL)
        context["files"] = '\n'.join(ls_output.splitlines()[:20])
    except Exception:
        context["files"] = "无法获取文件列表"

    try:
        git_output = subprocess.check_output(
            ['git', 'status', '-s'], text=True, stderr=subprocess.DEVNULL
        )
        context["git_status"] = git_output.strip() or "干净的工作区 (没有未提交的更改)"
    except Exception:
        context["git_status"] = "当前不是 Git 仓库"

    # 读取 AGENTS.md（Bonus 4）
    try:
        agents_md_path = os.path.join(os.getcwd(), "AGENTS.md")
        with open(agents_md_path, "r", encoding="utf-8") as f:
            context["agents_md"] = f.read().strip()
    except FileNotFoundError:
        context["agents_md"] = ""

    return context


# === 3. System Prompt 生成 ===
def get_system_prompt(context: dict) -> str:
    """生成带环境上下文的系统提示词，要求 LLM 输出结构化 JSON"""
    return f"""你是一个多 Agent 命令行系统的"总控 Agent"。你的唯一任务是分析用户的自然语言输入，判断其意图，并输出结构化的 JSON 结果。

【当前环境上下文】
- 操作系统: {context['os']}
- Shell 类型: {context['shell']}
- 工作目录: {context['pwd']}
- 目录概览 (前20项):
{context['files']}
- Git 状态:
{context['git_status']}
""" + (f"""
【项目级 Agent 规则（来自 AGENTS.md）】
{context['agents_md']}
""" if context.get('agents_md') else "") + """
【意图分类标准】
1. "shell_agent" — 用户想执行系统命令。
   判断依据：涉及文件操作（查看/创建/删除/移动）、目录操作、进程管理、网络请求、
   压缩打包、系统信息查询等需要在终端执行命令才能完成的任务。
   示例："帮我看看当前目录有哪些文件"、"把项目打包成 zip"、"查看系统内存使用情况"

2. "tool_agent" — 用户想调用外部工具或需要读取文件内容后进行处理。
   判断依据：涉及文件内容的读取与分析、代码审查、数据处理、调用特定 API 或工具。
   示例："读取 config.json 的内容并总结"、"分析这段代码的复杂度"、"调用翻译 API"

3. "direct_answer" — 用户在闲聊或询问一般性知识，不需要执行任何操作。
   判断依据：问题是关于概念解释、知识问答、观点讨论等，不涉及对当前系统的操作。
   示例："什么是 Python 的 GIL？"、"解释一下 REST API"、"今天天气怎么样"

4. "clarification" — 用户的指令模糊不清，缺少关键信息，需要追问。
   判断依据：指令中包含"那个"、"这个"等模糊指代，或缺少必要的操作对象/参数。
   示例："帮我删除那个文件"（哪个文件？）、"把它发给他"（发什么？给谁？）

【输出要求】
你必须且只能输出一个合法的 JSON 对象，不要有任何其他文字。
请先在 reasoning 字段中写出你的推理过程（CoT），然后再给出结论。

JSON 格式：
{{
    "intent": "上述四种之一",
    "reasoning": "先分析用户输入的关键词和语义，结合环境上下文，再得出结论",
    "confidence": 0.95,
    "params": {{
        "task_description": "给下游 Agent 的简洁任务描述",
        "suggested_tools": ["仅 tool_agent 时填写建议工具，否则为空数组"],
        "question": "仅 clarification 时填写追问问题，否则为空字符串",
        "options": ["仅 clarification 时填写候选选项，否则为空数组"]
    }},
    "fallback_response": "仅 direct_answer 时填写完整回答内容，其他情况为空字符串"
}}"""


# === 4. 三层容错 JSON 解析 ===
def parse_llm_json(raw_text: str) -> dict | None:
    """
    三层容错机制解析 LLM 返回的 JSON：
    第一层：直接 json.loads()
    第二层：剥离 markdown 包裹（```json ... ```）后再解析
    第三层：用正则提取第一个 {...} 再解析
    全部失败返回 None
    """
    raw_text = raw_text.strip()

    # 第一层：直接解析
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        pass

    # 第二层：剥离 markdown 代码块包裹
    try:
        if raw_text.startswith("```"):
            lines = raw_text.split('\n')
            # 去掉第一行 ```json 和最后一行 ```
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            cleaned = '\n'.join(lines).strip()
            return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # 第三层：正则提取第一个完整的 JSON 对象 {...}
    try:
        # 匹配最外层的 { ... }，用非贪婪 + DOTALL 模式
        match = re.search(r'\{.*\}', raw_text, re.DOTALL)
        if match:
            return json.loads(match.group())
    except json.JSONDecodeError:
        pass

    return None


# === 5. Schema 校验 ===
def validate_intent_result(data: dict) -> tuple[bool, str]:
    """
    用 jsonschema 校验解析结果是否符合预期格式。
    返回 (是否通过, 错误信息)
    """
    try:
        validate(instance=data, schema=INTENT_JSON_SCHEMA)
        return True, ""
    except ValidationError as e:
        return False, f"Schema 校验失败: {e.message}"


# === 6. 核心意图分类（非流式，等完整 JSON）===
def classify_intent(user_input: str, llm_client=None, model: str = None, max_retries: int = 1):
    """
    调用 LLM 进行意图分类，返回结构化结果 dict。
    包含重试机制：解析失败时自动重试 max_retries 次，仍失败则降级为 direct_answer。
    """
    llm_client = llm_client or client
    model = model or DEFAULT_MODEL
    context = get_advanced_context()
    system_prompt = get_system_prompt(context)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input}
    ]

    # 检测是否支持 JSON mode（OpenAI / DeepSeek 系列）
    extra_params = {}
    if "deepseek" in model.lower() or model.startswith("gpt-"):
        extra_params["response_format"] = {"type": "json_object"}

    for attempt in range(1 + max_retries):
        try:
            response = llm_client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.1,
                **extra_params
            )
            raw_text = response.choices[0].message.content
            print(f"\n[LLM 原始返回]\n{raw_text}\n")

            # 三层容错解析
            parsed = parse_llm_json(raw_text)
            if parsed is None:
                if attempt < max_retries:
                    print(f"[重试 {attempt + 1}/{max_retries}] JSON 解析失败，正在重试...")
                    continue
                else:
                    print("[降级] JSON 解析多次失败，降级为 direct_answer")
                    return _make_fallback(user_input, raw_text)

            # Schema 校验
            valid, err_msg = validate_intent_result(parsed)
            if not valid:
                if attempt < max_retries:
                    print(f"[重试 {attempt + 1}/{max_retries}] {err_msg}，正在重试...")
                    continue
                else:
                    print(f"[降级] {err_msg}，降级为 direct_answer")
                    return _make_fallback(user_input, raw_text)

            return parsed

        except (AuthenticationError, RateLimitError, APIError) as e:
            print(f"[API 错误] {e}")
            return _make_fallback(user_input, str(e))
        except Exception as e:
            print(f"[未知错误] {e}")
            return _make_fallback(user_input, str(e))

    return _make_fallback(user_input, "重试耗尽")


def _make_fallback(user_input: str, raw_text: str) -> dict:
    """生成降级 fallback 结果"""
    return {
        "intent": "direct_answer",
        "reasoning": f"JSON 解析/校验失败，自动降级。原始内容: {raw_text[:200]}",
        "confidence": 0.0,
        "params": {
            "task_description": user_input,
            "suggested_tools": [],
            "question": "",
            "options": []
        },
        "fallback_response": ""
    }


# === 7. direct_answer 的流式回复 ===
def stream_direct_answer(user_input: str, llm_client=None, model: str = None):
    """当 direct_answer 需要完整回答时，发起流式请求"""
    llm_client = llm_client or client
    model = model or DEFAULT_MODEL

    messages = [
        {"role": "system", "content": "你是一个有用的 AI 助手，请直接回答用户的问题。"},
        {"role": "user", "content": user_input}
    ]

    print("Agent: ", end="")
    full_response = ""

    try:
        response = llm_client.chat.completions.create(
            model=model,
            messages=messages,
            stream=True,
            temperature=0.7
        )
        for chunk in response:
            if len(chunk.choices) > 0 and chunk.choices[0].delta.content is not None:
                content = chunk.choices[0].delta.content
                print(content, end="", flush=True)
                full_response += content
        print("\n")
    except Exception as e:
        print(f"\n[流式回复错误] {e}")

    return full_response


# === 8. 整合入口：分类 → 阈值判断 → 分发/回复 ===
def handle_intent(user_input: str, llm_client=None, model: str = None) -> dict:
    """
    完整处理流程：
    1. 调用 classify_intent 获取分类结果
    2. 根据 confidence 阈值进行判断
       >= 0.8  → 直接分发
       0.5~0.8 → 分发但标注 "[低置信度]"
       < 0.5   → 自动降级为 clarification
    3. 若为 direct_answer，判断是否需要流式补充回答
    """
    llm_client = llm_client or client
    model = model or DEFAULT_MODEL

    print(f"\nUser: {user_input}")
    print("-" * 50)

    # Step 1: 意图分类
    result = classify_intent(user_input, llm_client, model)
    confidence = result.get("confidence", 0.0)
    intent = result.get("intent", "direct_answer")

    # Step 2: Confidence 阈值判断
    if confidence < CONFIDENCE_LOW:
        print(f"[置信度过低 ({confidence:.2f} < {CONFIDENCE_LOW})] 自动降级为 clarification")
        result["intent"] = "clarification"
        intent = "clarification"
        if not result["params"].get("question"):
            result["params"]["question"] = "你的指令不太明确，能否提供更多细节？"
            result["params"]["options"] = []

    confidence_tag = ""
    if CONFIDENCE_LOW <= confidence < CONFIDENCE_HIGH:
        confidence_tag = " [低置信度]"
        print(f"[注意] 置信度偏低 ({confidence:.2f})，结果可能不够准确")

    # Step 3: 输出分类结果
    print(f"\n{'=' * 50}")
    print(f"意图分类{confidence_tag}: {intent}")
    print(f"置信度: {confidence:.2f}")
    print(f"推理过程: {result.get('reasoning', 'N/A')}")
    print(f"任务描述: {result['params'].get('task_description', 'N/A')}")

    if intent == "shell_agent":
        print(f"→ 分发给 Shell Agent 执行")

    elif intent == "tool_agent":
        tools = result['params'].get('suggested_tools', [])
        print(f"→ 分发给 Tool Agent，建议工具: {tools}")

    elif intent == "direct_answer":
        fallback = result.get("fallback_response", "")
        # 判断 fallback_response 是否足够完整
        if len(fallback) >= 20 and confidence >= CONFIDENCE_HIGH:
            print(f"→ 直接回答（来自分类结果）:")
            print(f"  {fallback}")
        else:
            print(f"→ fallback_response 不够完整，发起流式补充回答:")
            stream_direct_answer(user_input, llm_client, model)

    elif intent == "clarification":
        question = result['params'].get('question', '请补充更多信息')
        options = result['params'].get('options', [])
        print(f"→ 需要澄清: {question}")
        if options:
            print(f"  候选选项: {options}")

    print(f"{'=' * 50}\n")
    return result


# === 主程序入口 ===
if __name__ == "__main__":
    print("=" * 60)
    print("  Task 2.2 - 意图分类模块 (结构化输出)")
    print("=" * 60)

    test_input = input("\n请输入指令（或直接回车使用默认测试）: ").strip()
    if not test_input:
        test_input = "帮我看看当前目录下有哪些 Python 文件"

    handle_intent(test_input)
