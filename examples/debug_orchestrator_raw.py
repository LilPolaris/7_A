"""调试脚本：原样打印总控 Agent 的 LLM 原始输出。"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.orchestrator.intent_classifier import (
    INTENT_JSON_SCHEMA,
    INTENT_JSON_SCHEMA_NAME,
    get_advanced_context,
    get_system_prompt,
)
from src.orchestrator.llm_client import DEFAULT_MODEL, generate_text_response, get_api_mode, get_llm_client


def fetch_raw_orchestrator_output(user_input: str, *, model: str | None = None) -> str:
    """按总控当前逻辑请求 LLM，但不做 JSON 解析/校验/降级，直接返回原始文本。"""
    llm_client = get_llm_client()
    context = get_advanced_context()
    system_prompt = get_system_prompt(context)
    # print(f"[debug] system_prompt for intent classification:\n{system_prompt}\n{'-'*60}")

    return generate_text_response(
        system_prompt,
        user_input,
        llm_client=llm_client,
        model=model or DEFAULT_MODEL,
        temperature=0.1,
        json_schema=INTENT_JSON_SCHEMA,
        json_schema_name=INTENT_JSON_SCHEMA_NAME,
    )


def main() -> None:
    if len(sys.argv) > 1:
        user_input = " ".join(sys.argv[1:]).strip()
    else:
        user_input = input("请输入要调试的用户输入: ").strip()

    if not user_input:
        print("用户输入不能为空")
        raise SystemExit(1)

    llm_client = get_llm_client()
    model = DEFAULT_MODEL
    api_mode = get_api_mode(llm_client)

    print(f"[debug] model={model} api_mode={api_mode}")
    print("[debug] 以下为总控 Agent 调用 API 后拿到的原始文本：")
    print("-" * 60)

    raw_text = fetch_raw_orchestrator_output(user_input, model=model)
    print(raw_text, end="" if raw_text.endswith("\n") else "\n")

    print("-" * 60)


if __name__ == "__main__":
    main()
