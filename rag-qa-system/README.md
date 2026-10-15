# RAG 教务知识问答系统

> 检索增强生成（RAG）技术演进与高校教务知识问答系统实践 —— 课程报告配套实验

一个面向高校教务知识库的 RAG 问答原型，演示"**检索 → 重排 → 生成 → 溯源**"的完整流程，
并提供检索命中率、答案忠实度、回答相关性的多维评估与可视化。

## 特性

- **本地向量检索**：基于 `sentence-transformers` 多语言模型 + `ChromaDB`，无需 API Key 即可运行
- **交叉编码器重排**：`BAAI/bge-reranker-base` 对初检结果做精排，可开关对比
- **国产大模型生成**：支持 DeepSeek / 通义千问（兼容 OpenAI 接口）；未配置时自动降级为离线兜底应答
- **可解释溯源**：每条回答附带引用文档编号、相关度评分与原文片段
- **多维评估**：命中率、Recall@K、ROUGE-L、忠实度，自动对比朴素 RAG 与重排 RAG
- **Web 界面**：Flask + Vue3 + ECharts，浅色清新风格，支持交互式问答与指标可视化

## 技术栈

| 层 | 选型 |
|----|------|
| 文档分块 | LangChain `RecursiveCharacterTextSplitter` |
| 向量编码 | `sentence-transformers` paraphrase-multilingual-MiniLM-L12-v2 |
| 向量库 | ChromaDB（持久化、cosine） |
| 重排序 | `sentence-transformers` CrossEncoder（bge-reranker-base） |
| 生成 | DeepSeek / 通义千问（OpenAI 兼容接口） |
| Web | Flask + Vue3(CDN) + ECharts(CDN) |

## 目录结构

```
rag-qa-system/
├── data/
│   ├── knowledge_base.json    # 教务知识库（12 篇文档）
│   └── eval_dataset.json      # 评测集（12 题，含 gold 文档与参考答案）
├── src/
│   ├── config.py             # 全局配置
│   ├── embeddings.py         # 嵌入模块
│   ├── vector_store.py       # ChromaDB 封装
│   ├── ingestion.py          # 加载/分块/入库
│   ├── retrieval.py          # 检索 + 重排
│   ├── generation.py         # LLM 生成
│   ├── rag_pipeline.py       # 完整流水线
│   └── evaluate.py           # 评估指标
├── templates/index.html      # Web 前端
├── app.py                    # Flask 服务
├── requirements.txt
└── README.md
```

## 环境要求

- **Python 3.10+**（依赖 chromadb / sentence-transformers 需 3.8+，推荐 3.13）
- Windows 下若默认 `python` 版本过低，可用 `py -3.13` 调用高版本，或用 `py -3.13 -m pip install -r requirements.txt`

## 快速开始

### 1. 安装依赖

```bash
cd rag-qa-system
pip install -r requirements.txt
```

### 2. 启动服务（离线模式，无需 API Key）

```bash
python app.py
```

打开浏览器访问 `http://127.0.0.1:5000`。

首次启动会自动下载向量模型（约 470MB）并构建索引。

### 3. 接入国产大模型（可选，获得自然语言回答）

DeepSeek：

```bash
# Windows PowerShell
$env:DEEPSEEK_API_KEY="你的Key"
$env:RAG_LLM_PROVIDER="deepseek"
python app.py
```

通义千问：

```bash
$env:DASHSCOPE_API_KEY="你的Key"
$env:RAG_LLM_PROVIDER="qwen"
python app.py
```

未配置时，系统自动降级为"检索片段拼接"离线模式，仍可完成检索与评估实验。

### 4. 对比实验（推荐，复现报告表 2）

```powershell
# 重建索引并在一次进程内跑完 朴素/重排 两组（自动预热，结果写入 results/eval_result.json）
py -3.13 run_compare.py --rebuild
# 之后再次运行可省略 --rebuild 以复用已有索引
py -3.13 run_compare.py
```

### 5. 单独评估 / 重建索引

```powershell
py -3.13 -m src.evaluate              # 重排 RAG
py -3.13 -m src.evaluate --no-rerank  # 朴素 RAG 对照
py -3.13 -m src.ingestion             # 重建向量索引
```

## 评估维度说明

| 指标 | 含义 | 价值 |
|------|------|------|
| Hit@3 | top-3 是否覆盖 gold 文档 | 检索能否找到正确资料 |
| MRR | gold 首次出现位置的倒数 | 排在第几，对重排最敏感 |
| 上下文精度@3 | top-3 中属于 gold 文档的分块占比 | 喂给生成器的上下文是否纯净 |
| Recall@K | 命中 gold 占比 | 多 gold 时召回完整性 |
| ROUGE-L | 答案与参考答案的字符级 LCS F1 | 回答与标准答案契合度（mock 模式下仅为下限参考） |
| Faithfulness | 答案与检索上下文的 1-gram 重叠率 | 答案是否"贴近"资料（反幻觉参考） |

## 局限说明

- 评测集与参考答案为作者基于教务规定构造（12 题），规模有限，结论为教学演示性质
- 报告主结果（`results/eval_result.json`）由 DeepSeek 真实生成测得；离线 mock 组（`results/eval_result_mock.json`）
  只做检索片段拼接、不产生真实 LLM 文本，其 ROUGE-L（0.273，不足真实生成 0.586 的一半）仅作下限参考
- 忠实度采用字面重叠启发式，生产环境建议引入 LLM-as-Judge 或人工标注
- 知识库内容为通用教务规定示例，不对应任何高校正式文件
