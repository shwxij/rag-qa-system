# -*- coding: utf-8 -*-
"""RAG 教务问答系统 - 全局配置

设计说明：
- 优先使用本地 sentence-transformers 做向量编码，无需 API Key 即可运行（保证可复现）。
- LLM 生成支持国产大模型 API（DeepSeek / 通义千问，均兼容 OpenAI 接口）；
  当未配置 API Key 时自动降级为"检索片段拼接 + 规则式模板应答"，保证系统离线可用。
- 所有可调参数集中于此，便于消融实验。
"""
import os
from pathlib import Path

# ---------- HuggingFace 镜像（国内网络无法直连 huggingface.co） ----------
# 优先尊重用户已设环境变量；未设则走国内镜像，避免下载超时
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
# 关闭 HF 在线遥测，减少不必要的网络请求
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
# 允许 sentence-transformers 在受限网络下走镜像拉取
os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", str(Path.home() / ".cache" / "sentence-transformers"))

# ---------- 路径 ----------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
KNOWLEDGE_FILE = DATA_DIR / "knowledge_base.json"
EVAL_FILE = DATA_DIR / "eval_dataset.json"
PERSIST_DIR = BASE_DIR / "vectordb"        # ChromaDB 持久化目录
LOG_DIR = BASE_DIR / "logs"

# ---------- 嵌入模型 ----------
# 多语言 MiniLM，对中文支持良好，模型体量小（约 470MB），适合教学实验
EMBED_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
EMBED_BATCH_SIZE = 32

# ---------- 检索参数 ----------
CHUNK_SIZE = 120          # 文本分块字符数（细化分块以增大检索空间、考验定位精度）
CHUNK_OVERLAP = 30        # 分块重叠（保留上下文）
RETRIEVE_TOPK = 5         # 初检召回数
RERANK_TOPK = 3           # 重排后保留数
USE_RERANK = True         # 是否启用重排（消融对比用）

# ---------- LLM 生成 ----------
# 1) 优先读取环境变量；2) 兼容 DeepSeek 与通义千问
LLM_PROVIDER = os.getenv("RAG_LLM_PROVIDER", "deepseek")  # deepseek | qwen | mock
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

QWEN_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
QWEN_BASE_URL = os.getenv("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
QWEN_MODEL = os.getenv("QWEN_MODEL", "qwen-plus")

LLM_TEMPERATURE = 0.2     # 低温度以减少幻觉
LLM_MAX_TOKENS = 600

# ---------- 提示词 ----------
SYSTEM_PROMPT = (
    "你是一名高校教务知识问答助手。请仅依据下面给出的【检索资料】作答，"
    "不要编造未在资料中出现的信息。若资料不足以回答，请明确说明'根据现有资料无法回答'。"
    "回答应准确、简洁、分条理，并在末尾标注引用的资料编号。"
)

RETRIEVAL_PROMPT_TEMPLATE = (
    "【检索资料】\n{context}\n\n"
    "【问题】{question}\n\n"
    "请依据上述资料回答问题，并在末尾用[资料编号]标注引用来源。"
)


def ensure_dirs():
    """确保所需目录存在。"""
    for d in (DATA_DIR, PERSIST_DIR, LOG_DIR):
        d.mkdir(parents=True, exist_ok=True)


def is_llm_available() -> bool:
    """判断是否有可用的 LLM API。"""
    if LLM_PROVIDER == "mock":
        return False
    if LLM_PROVIDER == "deepseek":
        return bool(DEEPSEEK_API_KEY)
    if LLM_PROVIDER == "qwen":
        return bool(QWEN_API_KEY)
    return False
