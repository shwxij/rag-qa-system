# -*- coding: utf-8 -*-
"""检索模块：稠密检索 + 重排序。

设计体现 RAG 检索增强的核心思想：
- Stage 1：基于向量相似度的稠密初检，快速从大库中召回 top-K；
- Stage 2（可选）：交叉编码器重排，对 query-doc 对做更精细的相关性打分，
  提升最终保留片段的精度。

重排序使用 cross-encoder 模型（bge-reranker-base 中文友好）。
未安装时自动跳过重排，仅用稠密检索结果。
"""
from __future__ import annotations

from typing import List, Dict, Any

from . import config

try:
    from sentence_transformers import CrossEncoder
    _CE_AVAILABLE = True
except Exception:
    _CE_AVAILABLE = False


# 模块级单例
_reranker = None


def get_reranker():
    global _reranker
    if _reranker is None and _CE_AVAILABLE and config.USE_RERANK:
        # BAAI/bge-reranker-base 对中英文均友好，体量小
        _reranker = CrossEncoder("BAAI/bge-reranker-base", max_length=512)
    return _reranker


def retrieve(query: str, vector_store, top_k: int | None = None,
             rerank: bool | None = None) -> List[Dict[str, Any]]:
    """执行 检索 → （可选）重排 流水线。"""
    top_k = top_k or config.RETRIEVE_TOPK
    do_rerank = (rerank if rerank is not None else config.USE_RERANK)

    # Stage 1：稠密初检
    candidates = vector_store.search(query, top_k=top_k)

    # Stage 2：交叉编码器重排
    if do_rerank and len(candidates) > config.RERANK_TOPK and get_reranker() is not None:
        pairs = [(query, c["text"]) for c in candidates]
        scores = get_reranker().predict(pairs)
        for c, s in zip(candidates, scores):
            c["rerank_score"] = float(s)
        candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
        candidates = candidates[: config.RERANK_TOPK]
    else:
        candidates = candidates[: config.RERANK_TOPK]

    return candidates
