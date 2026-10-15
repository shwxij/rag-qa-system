# -*- coding: utf-8 -*-
"""数据入库模块：文档加载、分块、向量化、写入向量库。

体现 RAG 索引构建流程（Ingestion 阶段）。采用递归字符分块器，
保留 chunk_overlap 以在块边界保留上下文，避免语义被切断。
"""
from __future__ import annotations

import json
from typing import List, Dict, Any

from . import config
from .vector_store import VectorStore

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    _SPLITTER_AVAILABLE = True
except Exception:  # pragma: no cover
    _SPLITTER_AVAILABLE = False


def load_documents(path=None) -> List[Dict[str, Any]]:
    """从 JSON 加载教务知识库文档。"""
    path = path or config.KNOWLEDGE_FILE
    with open(path, "r", encoding="utf-8") as f:
        docs = json.load(f)
    return docs


def chunk_documents(docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """对每篇文档分块。

    返回 [{"id": "doc-001#0", "text": "...", "metadata": {...}}]
    metadata 保留文档级信息，便于检索溯源。
    """
    if _SPLITTER_AVAILABLE:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            separators=["\n\n", "\n", "。", "；", "，", " ", ""],
        )
    else:  # 简易兜底分块
        class _Dummy:
            def __init__(s, size, overlap):
                s.size, s.overlap = size, overlap
            def split_text(s, t):
                return [t[i:i + s.size] for i in range(0, len(t), s.size - s.overlap)]
        splitter = _Dummy(config.CHUNK_SIZE, config.CHUNK_OVERLAP)

    chunks: List[Dict[str, Any]] = []
    for doc in docs:
        pieces = splitter.split_text(doc["content"])
        for i, piece in enumerate(pieces):
            meta = {
                "doc_id": doc["id"],
                "title": doc.get("title", ""),
                "source": doc.get("source", ""),
                "chunk_index": i,
            }
            chunks.append({
                "id": f"{doc['id']}#{i}",
                "text": piece,
                "metadata": meta,
            })
    return chunks


def build_index(force_rebuild: bool = False) -> tuple[VectorStore, int]:
    """构建向量索引。若库已存在且非强制重建则直接复用。"""
    vs = VectorStore()
    if force_rebuild:
        vs.reset()
    if vs.count() == 0:
        docs = load_documents()
        chunks = chunk_documents(docs)
        n = vs.upsert(chunks)
        print(f"[ingestion] 已索引 {n} 个分块（来自 {len(docs)} 篇文档）。")
        return vs, n
    print(f"[ingestion] 向量库已有 {vs.count()} 个分块，跳过重建。")
    return vs, vs.count()


if __name__ == "__main__":
    build_index(force_rebuild=True)
