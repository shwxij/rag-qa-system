# -*- coding: utf-8 -*-
"""嵌入模块：统一封装本地 sentence-transformers 嵌入。

使用本地模型以保证实验可复现、无需 API Key。模型在首次运行时自动下载，
之后从本地缓存加载。对长文本采用分批编码以控制内存。
"""
from __future__ import annotations

from typing import List
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
    _ST_AVAILABLE = True
except Exception:  # pragma: no cover - 依赖未安装时的兜底
    _ST_AVAILABLE = False

from . import config


class Embedder:
    """句子级向量编码器。"""

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or config.EMBED_MODEL_NAME
        self._model = None
        self._dim = None

    @property
    def dim(self) -> int:
        if self._dim is None:
            self._lazy_load()
        return self._dim

    def _lazy_load(self):
        if self._model is not None:
            return  # 已加载则复用，避免每次 encode 重建模型（会触发 hub 检查，慢达秒级）
        if not _ST_AVAILABLE:
            raise RuntimeError(
                "sentence-transformers 未安装，请先 `pip install -r requirements.txt`。"
            )
        self._model = SentenceTransformer(self.model_name)
        # 兼容 sentence-transformers 新旧版本的维度接口
        if hasattr(self._model, "get_embedding_dimension"):
            self._dim = self._model.get_embedding_dimension()
        else:  # pragma: no cover - 旧版本兼容
            self._dim = self._model.get_sentence_embedding_dimension()

    def encode(self, texts: List[str], batch_size: int | None = None) -> np.ndarray:
        """对一组文本进行向量化，返回 [N, dim] 的 float32 矩阵。"""
        self._lazy_load()
        batch_size = batch_size or config.EMBED_BATCH_SIZE
        vecs = self._model.encode(
            texts, batch_size=batch_size, show_progress_bar=True,
            convert_to_numpy=True, normalize_embeddings=True,
        )
        return vecs.astype(np.float32)

    def encode_one(self, text: str) -> np.ndarray:
        return self.encode([text])[0]


# 模块级单例，避免重复加载模型
_default_embedder: Embedder | None = None


def get_embedder() -> Embedder:
    global _default_embedder
    if _default_embedder is None:
        _default_embedder = Embedder()
    return _default_embedder
