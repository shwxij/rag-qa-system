# -*- coding: utf-8 -*-
"""生成模块：调用 LLM 基于检索资料作答。

支持三种后端：
1. DeepSeek（兼容 OpenAI 接口）
2. 通义千问（兼容 OpenAI 接口）
3. mock：未配置 API Key 时降级为"检索片段拼接 + 模板应答"，保证离线可运行。

所有后端统一返回 dict：{answer, used_context, source_ids, backend, latency_ms}
"""
from __future__ import annotations

import time
import json
from typing import List, Dict, Any

from . import config

try:
    from openai import OpenAI
    _OAI_AVAILABLE = True
except Exception:
    _OAI_AVAILABLE = False


def _build_context(chunks: List[Dict[str, Any]]) -> str:
    """把检索片段拼成带编号的上下文。"""
    blocks = []
    for i, c in enumerate(chunks, 1):
        meta = c.get("metadata", {})
        title = meta.get("title", "")
        doc_id = meta.get("doc_id", "")
        blocks.append(f"[{i}] 来源：{title}（{doc_id}）\n{c['text']}")
    return "\n\n".join(blocks)


def _llm_call_deepseek(question: str, context: str) -> str:
    client = OpenAI(api_key=config.DEEPSEEK_API_KEY, base_url=config.DEEPSEEK_BASE_URL)
    resp = client.chat.completions.create(
        model=config.DEEPSEEK_MODEL,
        temperature=config.LLM_TEMPERATURE,
        max_tokens=config.LLM_MAX_TOKENS,
        messages=[
            {"role": "system", "content": config.SYSTEM_PROMPT},
            {"role": "user", "content": config.RETRIEVAL_PROMPT_TEMPLATE.format(
                context=context, question=question)},
        ],
    )
    return resp.choices[0].message.content


def _llm_call_qwen(question: str, context: str) -> str:
    client = OpenAI(api_key=config.QWEN_API_KEY, base_url=config.QWEN_BASE_URL)
    resp = client.chat.completions.create(
        model=config.QWEN_MODEL,
        temperature=config.LLM_TEMPERATURE,
        max_tokens=config.LLM_MAX_TOKENS,
        messages=[
            {"role": "system", "content": config.SYSTEM_PROMPT},
            {"role": "user", "content": config.RETRIEVAL_PROMPT_TEMPLATE.format(
                context=context, question=question)},
        ],
    )
    return resp.choices[0].message.content


def _mock_answer(question: str, context: str, chunks: List[Dict[str, Any]]) -> str:
    """离线兜底应答：直接给出检索片段并模板化。

    用于无 API Key 时仍可演示检索效果与评估流程（诚实度会受影响，报告中专门讨论）。
    """
    lines = ["[离线模式·未配置 LLM API] 以下为检索到的最相关资料片段，供参考：", ""]
    for i, c in enumerate(chunks, 1):
        meta = c.get("metadata", {})
        lines.append(f"[{i}] {meta.get('title', '')}\n{c['text']}")
    lines.append("")
    lines.append("提示：配置 DEEPSEEK_API_KEY 后可获得 LLM 生成的自然语言回答。")
    return "\n".join(lines)


def generate(question: str, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    context = _build_context(chunks)
    source_ids = [c.get("metadata", {}).get("doc_id", "") for c in chunks]

    t0 = time.time()
    backend = config.LLM_PROVIDER
    try:
        if config.is_llm_available():
            if backend == "deepseek" and _OAI_AVAILABLE:
                answer = _llm_call_deepseek(question, context)
            elif backend == "qwen" and _OAI_AVAILABLE:
                answer = _llm_call_qwen(question, context)
            else:
                answer = _mock_answer(question, context, chunks)
                backend = "mock"
        else:
            answer = _mock_answer(question, context, chunks)
            backend = "mock"
    except Exception as e:
        answer = f"[生成失败·回退至离线模式] {e}\n\n" + _mock_answer(question, context, chunks)
        backend = "mock(fallback)"

    latency = round((time.time() - t0) * 1000, 1)
    return {
        "answer": answer,
        "used_context": context,
        "source_ids": source_ids,
        "backend": backend,
        "latency_ms": latency,
    }
