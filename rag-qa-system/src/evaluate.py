# -*- coding: utf-8 -*-
"""评估模块：检索命中、答案忠实度、回答相关性。

对应报告中三个评估维度：
1. 检索命中率（Hit Rate / Recall@K）：top-K 是否覆盖 gold_doc_ids
2. 答案忠实度：生成答案是否包含未被资料支持的内容（启发式：与检索片段字面重叠率）
3. 回答相关性：答案与参考答案的字面相似度（BM25 加权 Jaccard + 字符 n-gram ROUGE-L）

注：这些指标为教学用轻量度量，适合定性比较消融配置；生产评估建议引入 LLM-as-Judge
或人工标注，报告讨论部分对此局限作专门说明。
"""
from __future__ import annotations

import json
import re
from typing import List, Dict, Any
from collections import Counter

from . import config
from .rag_pipeline import get_pipeline


# ---------- 分词工具 ----------
def _tokenize(text: str) -> List[str]:
    """中文按字切，英文按词切，统一小写。"""
    text = text.lower()
    # 抽取英文词
    words = re.findall(r"[a-z]+", text)
    # 中文按字
    words += list(re.sub(r"[^\u4e00-\u9fa5]", "", text))
    return words


def _ngrams(tokens: List[str], n: int = 2) -> Counter:
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


# ---------- 指标 ----------
def hit_rate(retrieved_ids: List[str], gold_ids: List[str]) -> float:
    """top-K 命中率：只要命中一个即算 1。"""
    s = set(retrieved_ids)
    return 1.0 if any(g in s for g in gold_ids) else 0.0


def recall_at_k(retrieved_ids: List[str], gold_ids: List[str]) -> float:
    """召回率：命中的 gold 占比。"""
    if not gold_ids:
        return 0.0
    s = set(retrieved_ids)
    hit = sum(1 for g in gold_ids if g in s)
    return hit / len(gold_ids)


def mrr(retrieved_ids: List[str], gold_ids: List[str]) -> float:
    """平均倒数排名（单题为 RR）：gold 首次出现位置的倒数；未命中为 0。

    相比命中率只看"在不在 top-K"，MRR 能反映"排在第几"，
    是衡量重排能否把正确片段提到更前位置的关键指标。
    """
    gold = set(gold_ids)
    for i, rid in enumerate(retrieved_ids, start=1):
        if rid in gold:
            return 1.0 / i
    return 0.0


def context_precision(retrieved_ids: List[str], gold_ids: List[str]) -> float:
    """上下文精度@N：返回片段中属于 gold 文档的比例。

    值越高代表喂给生成器的上下文越"纯净"、噪声越少。
    """
    if not retrieved_ids:
        return 0.0
    gold = set(gold_ids)
    relevant = sum(1 for rid in retrieved_ids if rid in gold)
    return relevant / len(retrieved_ids)


def rouge_l(pred: str, ref: str) -> float:
    """字符级 ROUGE-L F1。"""
    p_tok = list(re.sub(r"[^\u4e00-\u9fa5a-z0-9]", "", pred.lower()))
    r_tok = list(re.sub(r"[^\u4e00-\u9fa5a-z0-9]", "", ref.lower()))
    if not p_tok or not r_tok:
        return 0.0
    m, n = len(p_tok), len(r_tok)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if p_tok[i - 1] == r_tok[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    lcs = dp[m][n]
    if lcs == 0:
        return 0.0
    prec, rec = lcs / m, lcs / n
    return 2 * prec * rec / (prec + rec)


def faithfulness(answer: str, context: str) -> float:
    """忠实度：答案与上下文字符重叠率（1-gram overlap ratio）。

    值越高代表答案越"贴近"检索资料，越低越可能是编造内容。
    """
    a = set(_tokenize(answer))
    c = set(_tokenize(context))
    if not a:
        return 0.0
    return len(a & c) / len(a)


def run_eval(rerank: bool = True, eval_file=None, verbose: bool = True) -> Dict[str, Any]:
    """在评测集上跑全流程并汇总指标。"""
    eval_file = eval_file or config.EVAL_FILE
    with open(eval_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    pipe = get_pipeline()
    results = []
    hit_sum = rec_sum = rl_sum = faith_sum = mrr_sum = ctx_sum = 0.0
    retr_sum = gen_sum = 0.0

    for item in data:
        q = item["question"]
        gold = item["gold_doc_ids"]
        ref = item["reference_answer"]

        r = pipe.ask(q, rerank=rerank)
        retrieved_ids = r["source_ids"]
        h = hit_rate(retrieved_ids, gold)
        rec = recall_at_k(retrieved_ids, gold)
        rr = mrr(retrieved_ids, gold)
        ctx = context_precision(retrieved_ids, gold)
        rl = rouge_l(r["answer"], ref)
        fi = faithfulness(r["answer"], r["used_context"])

        hit_sum += h
        rec_sum += rec
        mrr_sum += rr
        ctx_sum += ctx
        rl_sum += rl
        faith_sum += fi
        retr_sum += r.get("retrieval_ms", 0.0)
        gen_sum += r.get("generation_ms", 0.0)

        results.append({
            "id": item["id"],
            "question": q,
            "hit": h, "recall": rec,
            "mrr": round(rr, 4),
            "context_precision": round(ctx, 4),
            "rouge_l": round(rl, 4),
            "faithfulness": round(fi, 4),
            "backend": r["backend"],
            "latency_ms": r["latency_ms"],
            "retrieval_ms": r.get("retrieval_ms", 0.0),
            "generation_ms": r.get("generation_ms", 0.0),
            "answer": r["answer"],
        })
        if verbose:
            print(f"[{item['id']}] hit={h} mrr={rr:.2f} ctx={ctx:.2f} "
                  f"rouge_l={rl:.3f} faith={fi:.3f} retr={r.get('retrieval_ms',0):.1f}ms")

    n = len(data)
    summary = {
        "n": n,
        "rerank": rerank,
        "hit_rate": round(hit_sum / n, 4),
        "recall_at_k": round(rec_sum / n, 4),
        "mrr": round(mrr_sum / n, 4),
        "context_precision": round(ctx_sum / n, 4),
        "rouge_l": round(rl_sum / n, 4),
        "faithfulness": round(faith_sum / n, 4),
        "avg_retrieval_ms": round(retr_sum / n, 2),
        "avg_generation_ms": round(gen_sum / n, 2),
        "avg_latency_ms": round(sum(r["latency_ms"] for r in results) / n, 2),
    }
    return {"summary": summary, "details": results}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-rerank", action="store_true", help="关闭重排（朴素 RAG 对照）")
    args = ap.parse_args()
    out = run_eval(rerank=not args.no_rerank)
    print("\n=== 汇总 ===")
    print(json.dumps(out["summary"], ensure_ascii=False, indent=2))
