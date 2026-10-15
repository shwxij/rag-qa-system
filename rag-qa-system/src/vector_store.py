# -*- coding: utf-8 -*-
"""向量存储模块：基于 ChromaDB 的持久化向量库。

ChromaDB 是面向 LLM 应用的开源向量数据库，纯本地运行、无服务依赖，
适合教学实验。我们用 sentence-transformers 向量入库，并保存原始文档元数据，
便于检索后溯源（这是 RAG 可解释性的重要基础）。

线程模型说明（踩坑记录）：ChromaDB 的 PersistentClient/System 对访问线程与同进程内
多实例较为敏感——在 Flask 默认多线程（threaded=True）下，或在同进程中反复新建
多个 PersistentClient 时，查询可能抛 "Collection ... does not exist"。
因此本系统在 app.py 中以**单线程**方式运行 Flask，并在进程内**复用同一个**
PersistentClient / collection（创建与使用都在主线程），与独立脚本的运行模型一致。
"""
from __future__ import annotations

from typing import List, Dict, Any

import chromadb
from chromadb.config import Settings

from . import config
from .embeddings import get_embedder


class VectorStore:
    COLLECTION_NAME = "academic_affairs"

    def __init__(self, persist_dir=None, collection_name=None):
        self.persist_dir = str(persist_dir or config.PERSIST_DIR)
        self._coll_name = collection_name or self.COLLECTION_NAME
        config.ensure_dirs()
        self._embedder = get_embedder()
        self._client = chromadb.PersistentClient(
            path=self.persist_dir,
            settings=Settings(anonymized_telemetry=False, allow_reset=True),
        )
        self._collection = self._client.get_or_create_collection(
            name=self._coll_name, metadata={"hnsw:space": "cosine"}
        )

    # ---------- 写入 ----------
    def upsert(self, chunks: List[Dict[str, Any]]):
        """批量写入/更新分块。chunks: list of {"id","text","metadata"}"""
        ids = [c["id"] for c in chunks]
        texts = [c["text"] for c in chunks]
        metadatas = [c.get("metadata", {}) for c in chunks]
        vectors = self._embedder.encode(texts).tolist()
        self._collection.upsert(
            ids=ids, embeddings=vectors, documents=texts, metadatas=metadatas
        )
        return len(ids)

    # ---------- 查询 ----------
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        qvec = self._embedder.encode_one(query).tolist()
        res = self._collection.query(
            query_embeddings=[qvec], n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        dists = res.get("distances", [[]])[0]
        results = []
        for i in range(len(docs)):
            results.append({
                "text": docs[i],
                "metadata": metas[i] if i < len(metas) else {},
                "score": 1 - float(dists[i]) if i < len(dists) else 0.0,  # cosine 相似度
            })
        return results

    # ---------- 统计 / 清空 ----------
    def count(self) -> int:
        return self._collection.count()

    def reset(self):
        self._client.delete_collection(self._coll_name)
        self._collection = self._client.get_or_create_collection(
            name=self._coll_name, metadata={"hnsw:space": "cosine"}
        )
