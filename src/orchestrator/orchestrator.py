"""
Task 2.3 - 总控 Agent（Orchestrator）
功能：对接前端输入，完成命令直通 / 意图分类 / 结果输出与路由。
"""

from __future__ import annotations

import asyncio
import sys
from typing import Any, Awaitable, Callable, Protocol

try:
    from .intent_classifier import (
        DEFAULT_MODEL,
        _make_fallback,
        classify_intent,
        get_advanced_context,
        handle_intent,
        parse_llm_json,
        validate_intent_result,
    )
    from .llm_client import generate_text_response, get_api_mode, get_llm_client
except ImportError:
    from intent_classifier import (  # type: ignore
        DEFAULT_MODEL,
        _make_fallback,
        classify_intent,
        get_advanced_context,
        handle_intent,
        parse_llm_json,
        validate_intent_result,
    )
    from llm_client import generate_text_response, get_api_mode, get_llm_client  # type: ignore


class OrchestratorOutput(Protocol):
    """前端输出适配接口。"""

    def output_system(self, text: str, style: str = "") -> None:
        ...

    def output_llm(
        self,
        content: str,
        markdown: bool | None = None,
        language: str | None = None,
    ) -> None:
        ...


ShellExecutor = Callable[[str, OrchestratorOutput], Awaitable[None]]
OrchestratorResult = dict[str, Any]

client = get_llm_client()


class ConsoleOutput:
    """命令行调试用输出适配器。"""

    def output_system(self, text: str, style: str = "") -> None:
        del style
        print(text)

    def output_llm(
        self,
        content: str,
        markdown: bool | None = None,
        language: str | None = None,
    ) -> None:
        del markdown, language
        print(content)


async def execute_command_async(command: str, ui: OrchestratorOutput | None = None) -> None:
    """异步执行 shell 命令，流式输出 stdout 和 stderr。"""

    def emit_system(text: str, style: str = "") -> None:
        if ui is not None:
            ui.output_system(text, style=style)
        else:
            print(text)

    emit_system(f"[直接执行] $ {command}", style="bold cyan")
    emit_system("-" * 40)

    try:
        process = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        async def read_stream(stream, prefix: str = "", style: str = "") -> None:
            while True:
                line = await stream.readline()
                if not line:
                    break
                emit_system(f"{prefix}{line.decode().rstrip()}", style=style)

        await asyncio.gather(
            read_stream(process.stdout),
            read_stream(process.stderr, prefix="[stderr] ", style="bold red"),
        )

        await process.wait()
        emit_system("-" * 40)
        emit_system(f"[完成] 退出码: {process.returncode}", style="dim")

    except Exception as exc:
        emit_system(f"[执行错误] {exc}", style="bold red")


class OrchestratorAgent:
    """可直接对接前端的后端总控 Agent。"""

    def __init__(
        self,
        *,
        shell_executor: ShellExecutor | None = None,
        llm_client=None,
        model: str | None = None,
        verbose: bool = False,
    ) -> None:
        self.shell_executor = shell_executor
        self.llm_client = llm_client or client
        self.model = model or DEFAULT_MODEL
        self.verbose = verbose

    def _emit_route_summary(self, result: OrchestratorResult, ui: OrchestratorOutput) -> None:
        intent = result.get("intent", "unknown")
        confidence = float(result.get("confidence", 0.0))
        risk_level = result.get("risk_level", "unknown")

        ui.output_system(
            f"[orchestrator] intent={intent} confidence={confidence:.2f} risk={risk_level}",
            style="dim",
        )

        if intent in {"shell_agent", "tool_agent"}:
            ui.output_system(
                result.get("task_description", "未生成任务描述"),
                style="bold cyan" if intent == "shell_agent" else "bold magenta",
            )
            context_passed = result.get("context_passed", [])
            if context_passed:
                ui.output_system(f"context_passed: {context_passed}", style="dim")

    @staticmethod
    def _build_clarification_markdown(result: OrchestratorResult) -> str:
        question = result.get("question", "请补充更多信息")
        options = result.get("options", [])
        if not options:
            return question

        option_lines = "\n".join(f"- {option}" for option in options)
        return f"{question}\n\n{option_lines}"

    async def handle_input(self, user_input: str, ui: OrchestratorOutput) -> OrchestratorResult:
        """后端统一输入入口：接收用户输入并回写前端。"""
        user_input = user_input.strip()
        if not user_input:
            return {"status": "ignored", "input": user_input}

        if user_input.startswith("/"):
            command = user_input[1:].strip()
            if not command:
                ui.output_system("[提示] / 后面请输入要执行的命令", style="yellow")
                return {"status": "invalid_command", "input": user_input}

            if self.shell_executor is None:
                ui.output_system(
                    f"[orchestrator] 收到直接命令但未配置 shell_executor: {command}",
                    style="bold yellow",
                )
                return {
                    "status": "shell_command_pending",
                    "intent": "shell_command",
                    "command": command,
                }

            await self.shell_executor(command, ui)
            return {
                "status": "shell_command_executed",
                "intent": "shell_command",
                "command": command,
            }

        result = await asyncio.to_thread(
            handle_intent,
            user_input,
            self.llm_client,
            self.model,
            1,
            self.verbose,
        )
        self._emit_route_summary(result, ui)

        intent = result.get("intent")
        if intent == "direct_answer":
            reply = result.get("reply", "")
            if reply:
                ui.output_llm(reply)
            else:
                reply = await asyncio.to_thread(
                    generate_text_response,
                    "你是一个有用的 AI 助手，请直接回答用户的问题。",
                    user_input,
                    llm_client=self.llm_client,
                    model=self.model,
                    temperature=0.7,
                )
                ui.output_llm(reply)
        elif intent == "clarification":
            ui.output_llm(self._build_clarification_markdown(result), markdown=True)

        return result

# 调试用：总控Agent前端接口适配器和命令行主循环
def build_controller(
    *,
    shell_executor: ShellExecutor | None = None,
    llm_client=None,
    model: str | None = None,
    verbose: bool = False,
) -> Callable[[str, OrchestratorOutput], Awaitable[OrchestratorResult]]:
    """构造符合前端 input_handler 签名的 controller。"""
    agent = OrchestratorAgent(
        shell_executor=shell_executor,
        llm_client=llm_client,
        model=model,
        verbose=verbose,
    )
    return agent.handle_input


_default_agent = OrchestratorAgent(verbose=False)


async def main_controller(user_input: str, ui: OrchestratorOutput) -> OrchestratorResult:
    """默认总控入口，可直接作为前端 command_handler 使用。"""
    return await _default_agent.handle_input(user_input, ui)


async def orchestrator_loop() -> None:
    """命令行模式下的总控 Agent 主循环。"""
    io = ConsoleOutput()
    agent = OrchestratorAgent(shell_executor=execute_command_async, verbose=True)

    print("=" * 60)
    print("  Task 2.3 - 总控 Agent (Orchestrator)")
    print("  输入自然语言指令，或用 / 开头直接执行命令")
    print("  输入 exit 或 quit 退出")
    print("=" * 60)
    print(f"[LLM 接口模式] {get_api_mode(client)}")

    ctx = get_advanced_context()
    print(f"\n[环境上下文]")
    print(f"  操作系统: {ctx['os']}")
    print(f"  Shell: {ctx['shell']}")
    print(f"  工作目录: {ctx['pwd']}")
    print(f"  文件数量: {len(ctx['files'].splitlines())} 项")
    print(f"  Git 状态: {ctx['git_status'][:50]}")
    print()

    while True:
        try:
            user_input = input(">>> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n再见！")
            break

        if not user_input:
            continue

        if user_input.lower() in ("exit", "quit"):
            print("再见！")
            break

        await agent.handle_input(user_input, io)


def classify_without_context(user_input: str) -> OrchestratorResult:
    """不注入环境上下文的意图分类，用于报告对比。"""
    bare_prompt = """你是一个多 Agent 命令行系统的\"总控大脑\"。你的唯一任务是分析用户的自然语言输入，判断其真实意图，并输出严格的结构化 JSON 进行路由。

支持的 intent：
1. \"shell_agent\": 确定性的本地系统操作。
2. \"tool_agent\": 需要认知能力、语义分析或调用外部资源的复杂任务。
3. \"direct_answer\": 纯知识问答或闲聊，无需系统执行任何操作。
4. \"clarification\": 指令模糊或存在安全隐患，必须追问。

输出字段顺序：
reasoning -> confidence -> intent -> risk_level -> 专属字段

如果 intent 是 \"shell_agent\" 或 \"tool_agent\"：
{
    \"reasoning\": \"...\",
    \"confidence\": 0.95,
    \"intent\": \"shell_agent\" | \"tool_agent\",
    \"risk_level\": \"low\" | \"medium\" | \"high\",
    \"task_description\": \"给下级 Agent 的清晰任务描述\",
    \"context_passed\": [\"提取出的关键参数\"]
}

如果 intent 是 \"direct_answer\"：
{
    \"reasoning\": \"...\",
    \"confidence\": 1.0,
    \"intent\": \"direct_answer\",
    \"risk_level\": \"low\",
    \"reply\": \"完整回答\"
}

如果 intent 是 \"clarification\"：
{
    \"reasoning\": \"...\",
    \"confidence\": 0.4,
    \"intent\": \"clarification\",
    \"risk_level\": \"low\",
    \"question\": \"追问问题\",
    \"options\": [\"候选项1\", \"候选项2\", \"其他\"]
}"""

    try:
        raw_text = generate_text_response(
            bare_prompt,
            user_input,
            llm_client=client,
            model=DEFAULT_MODEL,
            temperature=0.1,
            json_mode=True,
        )
        parsed = parse_llm_json(raw_text)
        if parsed:
            valid, _ = validate_intent_result(parsed)
            if valid:
                return parsed
        return _make_fallback(user_input, raw_text)
    except Exception as exc:
        return _make_fallback(user_input, str(exc))


def compare_context_effect(user_input: str) -> tuple[OrchestratorResult, OrchestratorResult]:
    """对比有/无上下文注入时 LLM 输出的差异。"""
    print(f"\n{'=' * 60}")
    print(f"  对比测试: \"{user_input}\"")
    print(f"{'=' * 60}")

    print(f"\n--- 无上下文注入 ---")
    result_bare = classify_without_context(user_input)
    print(f"  intent: {result_bare.get('intent')}")
    print(f"  confidence: {result_bare.get('confidence')}")
    print(f"  reasoning: {result_bare.get('reasoning', 'N/A')[:120]}")
    print(f"  task_description: {result_bare.get('task_description', 'N/A')}")

    print(f"\n--- 有上下文注入 ---")
    result_ctx = handle_intent(user_input, client, DEFAULT_MODEL, 1, False)
    print(f"  intent: {result_ctx.get('intent')}")
    print(f"  confidence: {result_ctx.get('confidence')}")
    print(f"  reasoning: {result_ctx.get('reasoning', 'N/A')[:120]}")
    print(f"  task_description: {result_ctx.get('task_description', 'N/A')}")

    print(f"\n{'=' * 60}\n")
    return result_bare, result_ctx


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--compare":
        compare_cases = [
            "帮我看看这个项目有哪些 Python 文件",
            "帮我删除那个文件",
            "这个项目用了什么框架",
        ]
        for case in compare_cases:
            compare_context_effect(case)
    else:
        asyncio.run(orchestrator_loop())
