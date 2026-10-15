# -*- coding: utf-8 -*-
"""对比实验脚本：朴素 RAG vs 重排 RAG。

一次进程内顺序运行两组配置，复用向量模型与向量库（重排器懒加载一次），
将汇总指标与逐题结果写入 results/eval_result.json，便于复现报告中的实验表。

用法：
    py -3.13 run_compare.py             # 复用已有索引
    py -3.13 run_compare.py --rebuild   # 改过分块参数后强制重建索引
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src import config
from src.ingestion import build_index
from src.evaluate import run_eval


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true", help="强制重建向量索引")
    args = ap.parse_args()

    if args.rebuild:
        build_index(force_rebuild=True)

    # 预热向量模型：提前加载权重，避免首次查询的模型加载耗时污染延迟统计
    print(">>> 预热向量模型（加载权重，不计入计时）...")
    from src.embeddings import get_embedder
    get_embedder().encode_one("warmup")

    print(">>> 运行朴素 RAG（稠密 top-3，不重排）...")
    naive = run_eval(rerank=False, verbose=True)

    # 预热重排器：提前加载 cross-encoder 权重，避免首次懒加载耗时污染延迟统计
    print("\n>>> 预热重排器（加载权重，不计入计时）...")
    from src.retrieval import get_reranker
    get_reranker()

    print(">>> 运行重排 RAG（稠密 top-5 → cross-encoder 重排 top-3）...")
    advanced = run_eval(rerank=True, verbose=True)

    out = {
        "config": {
            "chunk_size": config.CHUNK_SIZE,
            "chunk_overlap": config.CHUNK_OVERLAP,
            "retrieve_topk": config.RETRIEVE_TOPK,
            "rerank_topk": config.RERANK_TOPK,
            "embed_model": config.EMBED_MODEL_NAME,
        },
        "naive": naive["summary"],
        "advanced": advanced["summary"],
        "naive_details": naive["details"],
        "advanced_details": advanced["details"],
    }

    out_dir = config.BASE_DIR / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "eval_result.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 60)
    print("汇总对比：")
    for key in ("naive", "advanced"):
        s = out[key]
        print(f"\n[{key}] hit={s['hit_rate']} mrr={s['mrr']} ctx={s['context_precision']} "
              f"rouge_l={s['rouge_l']} faith={s['faithfulness']} "
              f"retr={s['avg_retrieval_ms']}ms gen={s['avg_generation_ms']}ms "
              f"total={s['avg_latency_ms']}ms")
    print(f"\n详细结果已写入：{out_path}")


if __name__ == "__main__":
    main()
