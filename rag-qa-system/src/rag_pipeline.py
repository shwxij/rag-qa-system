# -*- coding: utf-8 -*-
"""RAG 完整流水线：检索 → 生成。

封装为 RAGPipeline 类，统一对外提供 ask() 接口，
便于在 Web 服务和评估脚本中复用。
"""
from __future__ import annotations

import time
from typing import List, Dict, Any

from . import config
from .vector_store import VectorStore
from .retrieval import retrieve
from .generation import generate
from .ingestion import build_index


class RAGPipeline:
    def __init__(self, vector_store: "VectorStore | None" = None, build: bool = True):
        if vector_store is None and build:
            config.ensure_dirs()
            self.vs, _ = build_index()
        else:
            self.vs = vector_store

    def ask(self, question: str, rerank: "bool | None" = None,
            top_k: "int | None" = None) -> Dict[str, Any]:
        # 端到端计时：检索（含重排）与生成分开统计，才能体现 cross-encoder 的时间代价
        t0 = time.time()
        chunks = retrieve(question, self.vs, top_k=top_k, rerank=rerank)
        retrieval_ms = round((time.time() - t0) * 1000, 2)

        t1 = time.time()
        result = generate(question, chunks)
        generation_ms = round((time.time() - t1) * 1000, 2)

        result["question"] = question
        result["retrieval_ms"] = retrieval_ms
        result["generation_ms"] = generation_ms
        result["latency_ms"] = round(retrieval_ms + generation_ms, 2)  # 端到端耗时
        result["retrieved"] = [
            {"text": c["text"], "metadata": c.get("metadata", {}),
             "score": c.get("score", 0.0)} for c in chunks
        ]
        return result

    def ask_naive(self, question: str) -> Dict[str, Any]:
        """朴素 RAG：不重排，仅稠密检索 top-3。用于消融对比。"""
        return self.ask(question, rerank=False, top_k=3)


# 模块级单例（不使用注解，避免与 global 冲突，兼容低版本 Python）
_pipeline = None


def get_pipeline() -> "RAGPipeline":
    """全局单例。"""
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline()
    return _pipeline
