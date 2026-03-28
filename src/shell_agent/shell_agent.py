"""Shell Agent：负责命令生成、风险审查、确认执行与交互式澄清。"""

from __future__ import annotations

import asyncio
import json
import os
import platform
import re
import subprocess
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Literal

from jsonschema import ValidationError, validate
from openai import OpenAI
from rich.text import Text

try:
    from ..orchestrator.llm_client import DEFAULT_MODEL, generate_text_response, get_llm_client
except ImportError:
    try:
        from src.orchestrator.llm_client import (  # type: ignore
            DEFAULT_MODEL,
            generate_text_response,
            get_llm_client,
        )
    except ImportError:
        from orchestrator.llm_client import (  # type: ignore
            DEFAULT_MODEL,
            generate_text_response,
            get_llm_client,
        )

if TYPE_CHECKING:
    from ..orchestrator.orchestrator import OrchestratorOutput


RiskLevel = Literal["low", "medium", "high"]
ShellExecutor = Callable[[str, "OrchestratorOutput"], Awaitable[None]]
ShellPlan = dict[str, Any]


@dataclass(frozen=True)
class RiskRule:
    """本地风险规则。"""

    pattern: str
    risk_level: RiskLevel
    description: str


@dataclass
class CommandRiskAssessment:
    """命令风险评估结果。"""

    command: str
    final_risk: RiskLevel
    llm_risk: RiskLevel
    orchestrator_risk: RiskLevel
    local_risk: RiskLevel
    matched_rules: list[RiskRule]

    def detail_lines(self) -> list[Text]:
        risk_text = {
            "low": ("低", "bold green"),
            "medium": ("中", "bold yellow"),
            "high": ("高", "bold red"),
        }
        label, style = risk_text.get(self.final_risk, (self.final_risk, ""))
        detail = Text("风险等级：")
        detail.append(label, style=style)
        return [detail]


SHELL_AGENT_SCHEMA_NAME = "shell_agent_plan"
SHELL_AGENT_OUTPUT_SCHEMA = {
    "type": "object",
    "required": ["intent", "reason", "risk_level", "command", "question", "options"],
    "additionalProperties": False,
    "properties": {
        "intent": {
            "type": "string",
            "enum": ["run_command", "ask_clarification", "refuse"],
        },
        "command": {
            "anyOf": [
                {"type": "string"},
                {"type": "null"},
            ]
        },
        "reason": {"type": "string"},
        "risk_level": {
            "type": "string",
            "enum": ["low", "medium", "high"],
        },
        "question": {
            "anyOf": [
                {"type": "string"},
                {"type": "null"},
            ]
        },
        "options": {
            "anyOf": [
                {"type": "array", "items": {"type": "string"}},
                {"type": "null"},
            ]
        },
    },
}


class ShellAgent:
    """将 orchestrator 的 shell 任务落到具体命令执行。"""

    MANUAL_INPUT_OPTION = "手动输入..."
    MAX_CLARIFICATION_ROUNDS = 3

    SYSTEM_PROMPT = """
你是多 Agent CLI 系统中的 Shell Agent。
你的职责是：把总控路由给你的 shell 任务，转换为“安全、明确、可直接执行”的单条 shell 命令，或者在必要时追问/拒绝。

你必须且只能输出一个 JSON 对象，不要输出 Markdown、解释性前缀或代码块。
输出必须符合以下 schema：
{json_schema}

规则：
1. intent="run_command"
   - 仅当任务足够明确时使用。
   - command 必须是可直接执行的 shell 命令，不要带前导 /。
   - 如果命令具有副作用，也要给出 reason 说明为什么需要这样执行。
2. intent="ask_clarification"
   - 当目标不明确、路径不明确、或仍需要补充参数时使用。
   - question 要明确告诉用户还缺什么信息。
   - options 若能给出候选项就给，否则为 null。
3. intent="refuse"
   - 当请求危险、破坏性强、越权、或你无法安全完成时使用。

风险定义：
- low: 查询/读取类操作，如 ls、pwd、find、cat、grep。
- medium: 会创建/修改普通文件或目录，如 mkdir、touch、cp、mv。
- high: 删除、覆盖、格式化、提权、关机重启、危险 SQL 等高风险操作。

输出格式要求：
- run_command: question=null, options=null
- ask_clarification: command=null
- refuse: command=null, question=null, options=null

【总控路由结果】
- 原始用户输入: {user_input}
- task_description: {task_description}
- context_passed: {context_passed}
- orchestrator_risk_level: {orchestrator_risk_level}

【运行时环境】
- os: {os_info}
- shell: {shell_info}
- cwd: {cwd}
- files_preview:
{files_preview}
""".strip()

    LOCAL_RISK_RULES = [
        RiskRule(r"\brm\s+-rf\b", "high", "匹配高风险删除规则：rm -rf"),
        RiskRule(r"\bsudo\b", "high", "匹配提权规则：sudo"),
        RiskRule(r"\bmkfs(?:\.[a-z0-9]+)?\b", "high", "匹配磁盘格式化规则：mkfs"),
        RiskRule(r"\bdd\s+if=", "high", "匹配底层磁盘写入规则：dd if="),
        RiskRule(r"\bdrop\s+table\b", "high", "匹配危险 SQL 规则：DROP TABLE"),
        RiskRule(r"\btruncate\s+-s\s+0\b", "high", "匹配清空文件规则：truncate -s 0"),
        RiskRule(r"\bshutdown\b", "high", "匹配关机规则：shutdown"),
        RiskRule(r"\breboot\b", "high", "匹配重启规则：reboot"),
        RiskRule(r"\bpoweroff\b", "high", "匹配关机规则：poweroff"),
        RiskRule(r"\bchmod\s+.*777\b", "high", "匹配危险权限修改规则：chmod 777"),
        RiskRule(r"\brm\b", "medium", "匹配删除命令：rm"),
        RiskRule(r"\bmv\b", "medium", "匹配移动/重命名命令：mv"),
        RiskRule(r"\bcp\b", "medium", "匹配复制命令：cp"),
        RiskRule(r"\bmkdir\b", "medium", "匹配目录创建命令：mkdir"),
        RiskRule(r"\btouch\b", "medium", "匹配文件创建命令：touch"),
    ]

    def __init__(
        self,
        *,
        llm_client: OpenAI | None = None,
        model: str | None = None,
    ) -> None:
        self.llm_client = llm_client or get_llm_client()
        self.model = model or DEFAULT_MODEL

    @staticmethod
    def _emit_workflow(ui: "OrchestratorOutput", message: str, state: str = "info") -> None:
        ui.output_workflow(message, state=state)

    @staticmethod
    def _shorten_command(command: str, max_length: int = 60) -> str:
        """压缩命令展示长度；多行命令仅显示首行并用省略号提示。"""
        lines = [line.strip() for line in command.strip().splitlines() if line.strip()]
        preview = lines[0] if lines else ""
        if len(lines) > 1:
            preview += " ..."
        if len(preview) <= max_length:
            return preview
        return preview[: max_length - 3].rstrip() + "..."

    @staticmethod
    def _risk_rank(level: RiskLevel) -> int:
        return {"low": 0, "medium": 1, "high": 2}[level]

    @classmethod
    def _max_risk(cls, *levels: RiskLevel) -> RiskLevel:
        return max(levels, key=cls._risk_rank)

    @staticmethod
    def _runtime_context() -> dict[str, str]:
        files_preview = "无法获取文件列表"
        try:
            if platform.system() == "Windows":
                output = subprocess.check_output(["cmd", "/c", "dir"], text=True, stderr=subprocess.DEVNULL)
            else:
                output = subprocess.check_output(["ls", "-la"], text=True, stderr=subprocess.DEVNULL)
            files_preview = "\n".join(output.splitlines()[:30])
        except Exception:
            pass

        return {
            "os_info": platform.system(),
            "shell_info": os.environ.get("SHELL", "unknown"),
            "cwd": os.getcwd(),
            "files_preview": files_preview,
        }

    @staticmethod
    def _list_cwd_entries() -> list[tuple[str, bool]]:
        entries: list[tuple[str, bool]] = []
        try:
            with os.scandir(os.getcwd()) as iterator:
                for entry in iterator:
                    entries.append((entry.name, entry.is_dir()))
        except OSError:
            return []
        return sorted(entries, key=lambda item: (item[1], item[0].lower()))

    @staticmethod
    def _parse_json(raw_text: str) -> ShellPlan | None:
        raw_text = raw_text.strip()
        if not raw_text:
            return None
        decoder = json.JSONDecoder()

        for candidate in (
            raw_text,
            raw_text.removeprefix("```json").removeprefix("```").removesuffix("```").strip(),
        ):
            try:
                parsed = json.loads(candidate)
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                try:
                    parsed, _ = decoder.raw_decode(candidate)
                    if isinstance(parsed, dict):
                        return parsed
                except json.JSONDecodeError:
                    pass

        match = re.search(r"\{", raw_text)
        if not match:
            return None

        try:
            parsed, _ = decoder.raw_decode(raw_text[match.start() :])
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            return None

    @staticmethod
    def _normalize_plan(parsed: ShellPlan | None) -> ShellPlan | None:
        if not parsed:
            return None

        plan = dict(parsed)
        if not plan.get("intent"):
            if isinstance(plan.get("command"), str) and plan["command"].strip():
                plan["intent"] = "run_command"
                plan.setdefault("reason", "模型已直接给出命令，自动补全 intent。")
                plan.setdefault("risk_level", "low")
                plan.setdefault("question", None)
                plan.setdefault("options", None)
            elif isinstance(plan.get("question"), str) and plan["question"].strip():
                plan["intent"] = "ask_clarification"
                plan.setdefault("reason", "模型给出了追问内容，自动补全 intent。")
                plan.setdefault("risk_level", "low")
                plan.setdefault("command", None)
            else:
                plan["intent"] = "refuse"
                plan.setdefault("reason", "模型未返回可执行命令，自动降级为 refuse。")
                plan.setdefault("risk_level", "high")
                plan.setdefault("command", None)
                plan.setdefault("question", None)
                plan.setdefault("options", None)

        return plan

    @staticmethod
    def _validate_plan(plan: ShellPlan) -> tuple[bool, str]:
        try:
            validate(instance=plan, schema=SHELL_AGENT_OUTPUT_SCHEMA)
            intent = plan["intent"]
            if intent == "run_command":
                if not isinstance(plan.get("command"), str) or not plan["command"].strip():
                    return False, "run_command 必须提供非空 command"
            elif intent == "ask_clarification":
                if not isinstance(plan.get("question"), str) or not plan["question"].strip():
                    return False, "ask_clarification 必须提供非空 question"
            return True, ""
        except ValidationError as exc:
            return False, exc.message

    @staticmethod
    def _fallback_plan(user_input: str, raw_text: str = "") -> ShellPlan:
        cleaned = raw_text.strip()
        return {
            "intent": "ask_clarification",
            "reason": f"Shell Agent 未能稳定生成结构化命令。原始输出: {cleaned[:160]}" if cleaned else "Shell Agent 未能稳定生成结构化命令。",
            "risk_level": "low",
            "command": None,
            "question": f"我理解你是想执行 shell 操作：{user_input}\n\n但我还不能安全地确定具体命令，请补充更明确的目标、路径或参数。",
            "options": None,
        }

    def plan_command(self, user_input: str, routed_task: ShellPlan) -> ShellPlan:
        context = self._runtime_context()
        system_prompt = self.SYSTEM_PROMPT.format(
            json_schema=json.dumps(SHELL_AGENT_OUTPUT_SCHEMA, ensure_ascii=False, indent=2),
            user_input=user_input,
            task_description=routed_task.get("task_description", user_input),
            context_passed=json.dumps(routed_task.get("context_passed", []), ensure_ascii=False),
            orchestrator_risk_level=routed_task.get("risk_level", "low"),
            **context,
        )

        try:
            raw_text = generate_text_response(
                system_prompt,
                routed_task.get("task_description", user_input),
                llm_client=self.llm_client,
                model=self.model,
                temperature=0.1,
                json_schema=SHELL_AGENT_OUTPUT_SCHEMA,
                json_schema_name=SHELL_AGENT_SCHEMA_NAME,
            )
        except Exception as exc:
            return {
                "intent": "refuse",
                "reason": f"调用 Shell Agent 所需 LLM 失败：{exc}",
                "risk_level": "high",
                "command": None,
                "question": None,
                "options": None,
            }

        parsed = self._normalize_plan(self._parse_json(raw_text))
        if parsed is None:
            return self._fallback_plan(user_input, raw_text)

        valid, error_message = self._validate_plan(parsed)
        if not valid:
            return self._fallback_plan(user_input, f"{raw_text}\n[SchemaError] {error_message}")

        return parsed

    def assess_command_risk(
        self,
        command: str,
        llm_risk_level: RiskLevel,
        orchestrator_risk_level: RiskLevel = "low",
    ) -> CommandRiskAssessment:
        matched_rules = [
            rule
            for rule in self.LOCAL_RISK_RULES
            if re.search(rule.pattern, command, re.IGNORECASE)
        ]
        local_risk: RiskLevel = "low"
        if matched_rules:
            local_risk = self._max_risk(*(rule.risk_level for rule in matched_rules))

        final_risk = self._max_risk(llm_risk_level, orchestrator_risk_level, local_risk)
        return CommandRiskAssessment(
            command=command,
            final_risk=final_risk,
            llm_risk=llm_risk_level,
            orchestrator_risk=orchestrator_risk_level,
            local_risk=local_risk,
            matched_rules=matched_rules,
        )

    @staticmethod
    def _looks_like_log_request(text: str) -> bool:
        lowered = text.lower()
        return "日志" in text or "log" in lowered

    @staticmethod
    def _extract_filename_hints(*texts: str) -> list[str]:
        hints: set[str] = set()
        for text in texts:
            hints.update(match.lower() for match in re.findall(r"[A-Za-z0-9_.-]+\.[A-Za-z0-9]+", text))
            hints.update(match.lower() for match in re.findall(r"`([^`]+)`", text))
            hints.update(match.lower() for match in re.findall(r'"([^"]+)"', text))
            hints.update(match.lower() for match in re.findall(r"'([^']+)'", text))
        return [hint for hint in hints if hint]

    def _contextual_clarification_options(
        self,
        user_input: str,
        routed_task: ShellPlan,
        plan: ShellPlan,
        *,
        limit: int = 6,
    ) -> list[str]:
        entries = self._list_cwd_entries()
        if not entries:
            return []

        combined = "\n".join(
            filter(
                None,
                [
                    user_input,
                    routed_task.get("task_description", ""),
                    plan.get("question", ""),
                    " ".join(str(item) for item in (routed_task.get("context_passed") or [])),
                ],
            )
        )
        lowered = combined.lower()
        wants_directory = any(keyword in lowered for keyword in ["目录", "文件夹", "folder", "directory"]) and not any(
            keyword in lowered for keyword in ["文件", "日志", "log", "."]
        )

        filename_hints = self._extract_filename_hints(combined)
        extension_hints = {match.lower() for match in re.findall(r"\.[A-Za-z0-9]+", combined)}
        keyword_tokens = {
            token.lower()
            for token in re.findall(r"[A-Za-z0-9_-]{2,}", combined)
            if token.lower() not in {"shell", "agent", "with", "from", "that"}
        }
        if self._looks_like_log_request(combined):
            keyword_tokens.add("log")
            extension_hints.add(".log")

        scored: list[tuple[int, str]] = []
        for name, is_dir in entries:
            if wants_directory and not is_dir:
                continue
            if not wants_directory and is_dir and (filename_hints or extension_hints or self._looks_like_log_request(combined)):
                continue

            lowered_name = name.lower()
            score = 0
            if lowered_name in filename_hints:
                score += 10
            if any(lowered_name.endswith(ext) for ext in extension_hints):
                score += 6
            if self._looks_like_log_request(combined) and ("log" in lowered_name or lowered_name.endswith(".log")):
                score += 8
            for token in keyword_tokens:
                if token in lowered_name:
                    score += 2

            if score > 0:
                scored.append((score, name))

        if not scored:
            return []

        scored.sort(key=lambda item: (-item[0], item[1].lower()))
        return [name for _, name in scored[:limit]]

    @staticmethod
    def _merge_options(*option_groups: list[str] | None) -> list[str]:
        merged: list[str] = []
        seen: set[str] = set()
        for group in option_groups:
            if not group:
                continue
            for option in group:
                normalized = str(option).strip()
                if not normalized or normalized in seen:
                    continue
                merged.append(normalized)
                seen.add(normalized)
        return merged

    @staticmethod
    def _augment_user_input(user_input: str, question: str, answer: str) -> str:
        return f"{user_input}\n\n补充信息：针对“{question}”，用户选择/输入了：{answer}"

    @staticmethod
    def _augment_routed_task(routed_task: ShellPlan, answer: str) -> ShellPlan:
        enriched = dict(routed_task)
        context_passed = list(enriched.get("context_passed") or [])
        context_passed.append(answer)
        enriched["context_passed"] = context_passed
        return enriched

    async def _prompt_for_clarification(
        self,
        user_input: str,
        routed_task: ShellPlan,
        plan: ShellPlan,
        ui: "OrchestratorOutput",
    ) -> str | None:
        question = plan.get("question") or plan.get("reason") or "请补充更多信息。"
        context_options = self._contextual_clarification_options(user_input, routed_task, plan)
        llm_options = plan.get("options") if isinstance(plan.get("options"), list) else None
        options = self._merge_options(context_options, llm_options)

        if not options:
            fallback_entries = [
                name
                for name, is_dir in self._list_cwd_entries()
                if is_dir is False
            ][:4]
            options = self._merge_options(fallback_entries, ["补充路径", "补充文件名"])

        return await ui.prompt_clarification(
            question=question,
            options=options,
            allow_manual=True,
            manual_prompt="请输入更明确的文件名、路径或参数...",
        )

    async def _review_and_execute_command(
        self,
        command: str,
        ui: "OrchestratorOutput",
        shell_executor: ShellExecutor | None,
        *,
        llm_risk_level: RiskLevel,
        orchestrator_risk_level: RiskLevel,
        reason: str,
    ) -> ShellPlan:
        self._emit_workflow(ui, "正在进行风险审查...", state="running")
        assessment = self.assess_command_risk(command, llm_risk_level, orchestrator_risk_level)
        result: ShellPlan = {
            "intent": "run_command",
            "command": command,
            "reason": reason,
            "risk_level": assessment.final_risk,
            "question": None,
            "options": None,
            "risk_assessment": {
                "final_risk": assessment.final_risk,
                "llm_risk": assessment.llm_risk,
                "orchestrator_risk": assessment.orchestrator_risk,
                "local_risk": assessment.local_risk,
                "matched_rules": [rule.description for rule in assessment.matched_rules],
            },
        }

        if assessment.final_risk in {"medium", "high"}:
            if assessment.final_risk == "high":
                prompt_reason = "是否执行以下高风险命令？"
                confirm_label = "Yes，执行高风险命令"
                detail_lines = assessment.detail_lines()
                self._emit_workflow(ui, "检测到高风险命令，等待用户确认", state="warn")
            else:
                prompt_reason = "是否执行以下命令？"
                confirm_label = "Yes，继续执行"
                detail_lines = None
                self._emit_workflow(ui, "检测到中风险命令，等待用户确认", state="warn")
            confirmed = await ui.confirm_shell_command(
                command=command,
                risk_level=assessment.final_risk,
                reason=prompt_reason,
                details=detail_lines,
                confirm_label=confirm_label,
            )
            if not confirmed:
                self._emit_workflow(ui, "用户已取消执行", state="warn")
                result["status"] = "cancelled_by_user"
                return result
            self._emit_workflow(ui, "已收到确认，准备执行命令", state="done")

        if shell_executor is None:
            self._emit_workflow(
                ui,
                f"已生成命令，但当前未配置执行器：{self._shorten_command(command)}",
                state="warn",
            )
            result["status"] = "command_generated"
            return result

        self._emit_workflow(
            ui,
            f"正在执行命令：{self._shorten_command(command)}",
            state="running",
        )
        await shell_executor(command, ui)
        self._emit_workflow(ui, "命令执行流程结束", state="done")
        result["status"] = "executed"
        return result

    async def run_direct_command(
        self,
        command: str,
        ui: "OrchestratorOutput",
        shell_executor: ShellExecutor | None,
    ) -> ShellPlan:
        """对 /command 直输命令执行本地风险审查。"""
        self._emit_workflow(ui, "收到直接命令，正在进行本地风险审查...", state="running")
        return await self._review_and_execute_command(
            command,
            ui,
            shell_executor,
            llm_risk_level="low",
            orchestrator_risk_level="low",
            reason="这是用户直接输入的命令，未经过 LLM 规划；已按本地规则引擎进行风险审查。",
        )

    async def run(
        self,
        routed_task: ShellPlan,
        ui: "OrchestratorOutput",
        shell_executor: ShellExecutor | None,
        *,
        user_input: str,
    ) -> ShellPlan:
        working_input = user_input
        working_task = dict(routed_task)

        for _ in range(self.MAX_CLARIFICATION_ROUNDS + 1):
            self._emit_workflow(ui, "正在分析任务并生成命令...", state="running")
            plan = await asyncio.to_thread(self.plan_command, working_input, working_task)
            intent = plan.get("intent")

            if intent == "ask_clarification":
                self._emit_workflow(ui, "当前信息不足，需要补充信息", state="warn")
                answer = await self._prompt_for_clarification(working_input, working_task, plan, ui)
                if not answer:
                    self._emit_workflow(ui, "用户取消了澄清流程，未执行任何命令", state="warn")
                    plan["status"] = "cancelled_by_user"
                    return plan

                self._emit_workflow(ui, f"已收到补充信息：{answer}", state="info")
                working_input = self._augment_user_input(working_input, plan.get("question") or "补充信息", answer)
                working_task = self._augment_routed_task(working_task, answer)
                continue

            if intent == "refuse":
                self._emit_workflow(ui, "已拒绝执行该请求", state="warn")
                await ui.stream_llm(plan.get("reason", "该 shell 任务被拒绝执行。"))
                plan["status"] = "refused"
                return plan

            command = str(plan.get("command", "") or "").strip()
            if not command:
                fallback = self._fallback_plan(user_input, "缺少 command 字段")
                fallback["status"] = "clarification"
                return fallback

            self._emit_workflow(
                ui,
                f"已生成命令：{self._shorten_command(command)}",
                state="done",
            )
            reviewed = await self._review_and_execute_command(
                command,
                ui,
                shell_executor,
                llm_risk_level=plan.get("risk_level", "low"),
                orchestrator_risk_level=working_task.get("risk_level", "low"),
                reason=plan.get("reason", ""),
            )
            reviewed.update(
                {
                    "original_plan": plan,
                    "clarified_input": working_input,
                }
            )
            return reviewed

        self._emit_workflow(ui, "多轮澄清后仍无法确定命令", state="warn")
        await ui.stream_llm("我仍无法安全确定要执行的命令，请提供更明确的文件名、路径或参数。")
        return {
            "intent": "ask_clarification",
            "reason": "Shell Agent 多次澄清后仍无法确定命令。",
            "risk_level": "low",
            "command": None,
            "question": "我仍无法安全确定要执行的命令，请提供更明确的文件名、路径或参数。",
            "options": ["手动输入更具体信息", "改用 /命令 直接执行"],
            "status": "clarification_limit_reached",
        }


__all__ = ["SHELL_AGENT_OUTPUT_SCHEMA", "SHELL_AGENT_SCHEMA_NAME", "ShellAgent"]
