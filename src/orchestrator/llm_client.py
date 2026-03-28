"""Orchestrator 通用 LLM Client 与调用封装。"""

from __future__ import annotations

import os
import sys
from typing import Callable, Literal

import httpx
from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

ApiMode = Literal["responses", "chat"]
API_MODE_ENV = "OPENAI_API_MODE"
DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-5")

try:
    _CLIENT = OpenAI()
except Exception as exc:
    print(f"初始化失败，请检查 .env 文件是否配置正确: {exc}")
    sys.exit(1)


def get_llm_client() -> OpenAI:
    """返回 orchestrator 共用的同步 LLM client。"""
    return _CLIENT


def get_default_model() -> str:
    """返回默认模型名。"""
    return DEFAULT_MODEL


def _probe_payload(mode: ApiMode) -> dict:
    """用于探测接口存在性的最小请求体。"""
    if mode == "responses":
        return {
            "model": "__route_probe__",
            "input": "ping",
            "max_output_tokens": 1,
        }

    return {
        "model": "__route_probe__",
        "messages": [{"role": "user", "content": "ping"}],
        "max_tokens": 1,
        "stream": False,
    }


def _supports_mode(llm_client: OpenAI, mode: ApiMode) -> bool:
    """判断当前网关是否支持某种接口。"""
    endpoint = "responses" if mode == "responses" else "chat/completions"
    url = str(llm_client.base_url).rstrip("/") + f"/{endpoint}"
    headers = {
        "Authorization": f"Bearer {llm_client.api_key}",
        "Content-Type": "application/json",
    }

    try:
        with httpx.Client(timeout=10.0) as http_client:
            response = http_client.post(url, headers=headers, json=_probe_payload(mode))
    except httpx.HTTPError:
        return False

    return response.status_code not in {404, 405}


def _detect_api_mode(llm_client: OpenAI | None = None) -> ApiMode:
    """自动检测并缓存应使用 responses 还是 chat.completions。"""
    configured = os.getenv(API_MODE_ENV, "auto").strip().lower()
    if configured in {"responses", "chat"}:
        return configured  # type: ignore[return-value]

    llm_client = llm_client or _CLIENT

    responses_ok = _supports_mode(llm_client, "responses")
    chat_ok = _supports_mode(llm_client, "chat")

    if responses_ok:
        os.environ[API_MODE_ENV] = "responses"
        return "responses"
    if chat_ok:
        os.environ[API_MODE_ENV] = "chat"
        return "chat"

    raise RuntimeError("未检测到可用接口：responses 与 chat/completions 均不可用。")


def get_api_mode(llm_client: OpenAI | None = None) -> ApiMode:
    """读取已缓存的接口类型；若无则检测一次并写入环境变量。"""
    configured = os.getenv(API_MODE_ENV, "auto").strip().lower()
    if configured in {"responses", "chat"}:
        return configured  # type: ignore[return-value]
    return _detect_api_mode(llm_client)


def _supports_json_mode(model: str) -> bool:
    """仅对已知兼容模型启用 JSON mode。"""
    model_name = model.lower()
    return "deepseek" in model_name or model_name.startswith("gpt-")


def _prepare_responses_input(user_input: str, *, json_mode: bool) -> str:
    """为 Responses API 准备输入；json_mode 下显式加入 JSON 约束。"""
    if not json_mode:
        return user_input
    return (
        "Return a valid JSON object only. "
        "The response must be JSON with no extra text.\n\n"
        f"User request:\n{user_input}"
    )


def generate_text_response(
    system_prompt: str,
    user_input: str,
    *,
    llm_client: OpenAI | None = None,
    model: str | None = None,
    temperature: float = 0.1,
    json_mode: bool = False,
) -> str:
    """根据已检测的接口类型发起非流式调用并返回文本。"""
    llm_client = llm_client or _CLIENT
    model = model or DEFAULT_MODEL
    api_mode = get_api_mode(llm_client)

    if api_mode == "responses":
        request_kwargs = {
            "model": model,
            "instructions": system_prompt,
            "input": _prepare_responses_input(user_input, json_mode=json_mode),
            "temperature": temperature,
        }
        if json_mode and _supports_json_mode(model):
            request_kwargs["text"] = {"format": {"type": "json_object"}}

        full_response = ""
        try:
            with llm_client.responses.stream(**request_kwargs) as stream:
                for event in stream:
                    if event.type == "response.output_text.delta":
                        full_response += event.delta

                stream.get_final_response()
        except Exception:
            if json_mode and "text" in request_kwargs:
                request_kwargs.pop("text", None)
                full_response = ""
                with llm_client.responses.stream(**request_kwargs) as stream:
                    for event in stream:
                        if event.type == "response.output_text.delta":
                            full_response += event.delta

                    stream.get_final_response()
            else:
                raise
        return full_response

    request_kwargs = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input},
        ],
        "temperature": temperature,
    }
    if json_mode and _supports_json_mode(model):
        request_kwargs["response_format"] = {"type": "json_object"}

    response = llm_client.chat.completions.create(**request_kwargs)
    return response.choices[0].message.content or ""


def stream_text_response(
    system_prompt: str,
    user_input: str,
    *,
    llm_client: OpenAI | None = None,
    model: str | None = None,
    temperature: float = 0.7,
    on_delta: Callable[[str], None] | None = None,
) -> str:
    """根据已检测的接口类型发起流式调用并同步打印。"""
    llm_client = llm_client or _CLIENT
    model = model or DEFAULT_MODEL
    api_mode = get_api_mode(llm_client)
    full_response = ""

    def emit(text: str) -> None:
        if on_delta is not None:
            on_delta(text)
        else:
            print(text, end="", flush=True)

    if api_mode == "responses":
        with llm_client.responses.stream(
            model=model,
            instructions=system_prompt,
            input=user_input,
            temperature=temperature,
        ) as stream:
            for event in stream:
                if event.type == "response.output_text.delta":
                    emit(event.delta)
                    full_response += event.delta

            stream.get_final_response()
        return full_response

    stream = llm_client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input},
        ],
        stream=True,
        temperature=temperature,
    )
    for chunk in stream:
        if len(chunk.choices) > 0 and chunk.choices[0].delta.content is not None:
            content = chunk.choices[0].delta.content
            emit(content)
            full_response += content

    return full_response


__all__ = [
    "API_MODE_ENV",
    "ApiMode",
    "DEFAULT_MODEL",
    "generate_text_response",
    "get_api_mode",
    "get_default_model",
    "get_llm_client",
    "stream_text_response",
]
