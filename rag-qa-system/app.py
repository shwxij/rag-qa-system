# -*- coding: utf-8 -*-
"""RAG 教务问答系统 - Flask Web 服务

接口：
- GET  /              主页（Vue3 + ECharts 单页应用）
- POST /api/ask       提问，返回答案与检索溯源
- POST /api/eval      在评测集上运行评估，返回指标与可视化数据
- GET  /api/kb        返回知识库文档列表
"""
from __future__ import annotations

import sys
from pathlib import Path

# 让 app.py 直接运行时能 import src
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from flask import Flask, request, jsonify, Response
from flask_cors import CORS

from src import config
from src.rag_pipeline import get_pipeline
from src.evaluate import run_eval
from src.ingestion import load_documents
from src.retrieval import get_reranker

app = Flask(__name__, static_folder="static", template_folder="templates")
CORS(app)

# 启动时构建向量索引
pipe = get_pipeline()

# 前端是纯 Vue3 SPA，直接以静态文件返回，绕开 Jinja2（避免与 Vue {{ }} 冲突）
_INDEX_HTML = (BASE_DIR / "templates" / "index.html")


@app.route("/")
def index():
    return Response(_INDEX_HTML.read_text(encoding="utf-8"),
                    mimetype="text/html; charset=utf-8")


@app.route("/api/kb")
def kb():
    docs = load_documents()
    return jsonify([{
        "id": d["id"], "title": d.get("title", ""), "source": d.get("source", ""),
        "len": len(d.get("content", "")),
    } for d in docs])


@app.route("/api/ask", methods=["POST"])
def ask():
    data = request.get_json(force=True)
    question = (data.get("question") or "").strip()
    rerank = data.get("rerank", True)
    if not question:
        return jsonify({"error": "问题不能为空"}), 400
    try:
        r = pipe.ask(question, rerank=rerank)
        return jsonify({
            "answer": r["answer"],
            "backend": r["backend"],
            "latency_ms": r["latency_ms"],
            "source_ids": r["source_ids"],
            "retrieved": [
                {"text": c["text"][:200] + ("..." if len(c["text"]) > 200 else ""),
                 "metadata": c["metadata"], "score": round(c["score"], 4)}
                for c in r["retrieved"]
            ],
        })
    except Exception as e:
        # 完整堆栈打印到服务端日志，便于排查；客户端只收到错误信息
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/eval", methods=["POST"])
def eval_route():
    data = request.get_json(silent=True) or {}
    rerank = data.get("rerank", True)
    try:
        # 预热重排器，排除首次加载权重对延迟统计的干扰
        if rerank:
            get_reranker()
        out = run_eval(rerank=rerank, verbose=False)
        # 同时跑一次对照（关闭重排）
        out_naive = run_eval(rerank=False, verbose=False)
        return jsonify({
            "advanced": out["summary"],
            "naive": out_naive["summary"],
            "details": out["details"],
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print(f"[app] LLM 后端: {config.LLM_PROVIDER} | 可用: {config.is_llm_available()}")
    # 单线程运行：ChromaDB PersistentClient 对多线程访问敏感（会抛 Collection does not
    # exist）。教学系统为单用户、CPU 推理秒级，串行处理完全够用，且更稳定可复现。
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=False)
