# Enterprise RAG — 企业知识库问答系统

基于 RAG（检索增强生成）架构的企业知识库问答系统，支持文档上传、混合检索、LLM 答案生成与来源溯源。

## 技术栈

| 层级 | 技术 |
|------|------|
| 前端 | Vue 3 + Vite + Element Plus + SSE 流式响应 |
| 后端 | FastAPI + LangChain |
| 向量库 | Milvus 2.x（混合检索 + RRF 融合） |
| 嵌入模型 | BGE-M3（Dense + Sparse 双向量） |
| 重排序 | BGE-Reranker-v2-M3 |
| LLM | DeepSeek API |
| 文档解析 | PDF（pdfplumber）、DOCX（python-docx）、TXT（chardet） |
| 基础设施 | Docker Compose（etcd + MinIO + Milvus） |

## 检索流程

```
Query → BGE-M3 编码（Dense + Sparse 向量）
      → Milvus hybrid_search（IVF_FLAT + SPARSE_WAND → RRF 融合）
      → BGE-Reranker-v2-m3 重排序
      → LLM（DeepSeek）结合父块上下文 → 答案 + 来源引用
```

## 快速开始

### 1. 启动基础设施

```bash
docker compose up -d
```

### 2. 启动后端

```bash
cd backend
python -m venv venv
source venv/Scripts/activate   # Git Bash / WSL
# venv\Scripts\activate        # PowerShell
pip install -r requirements.txt
cp ../.env.example ../.env     # 编辑 .env 填入 DEEPSEEK_API_KEY
SKIP_MODELS=0 uvicorn app.main:app --reload --port 8000
```

> 模型首次运行时会自动从 HuggingFace 下载到 `cache/` 目录。
> 设置 `SKIP_MODELS=1` 仅启动 API 服务，不加载嵌入/重排序模型。

### 3. 启动前端

```bash
cd frontend
npm install
npm run dev                     # → http://localhost:5173
```

## 项目结构

```
├── backend/
│   ├── app/
│   │   ├── main.py                  # FastAPI 入口，生命周期管理
│   │   ├── config.py                # 配置加载（.env）
│   │   ├── api/
│   │   │   ├── documents.py         # 文档上传/列表/删除
│   │   │   └── query.py             # 问答接口（同步 + SSE 流式）
│   │   ├── services/
│   │   │   ├── milvus_store.py      # Milvus 客户端（Schema、混合检索）
│   │   │   ├── embedding_service.py # BGE-M3 嵌入
│   │   │   ├── reranker_service.py  # BGE-Reranker 重排序
│   │   │   ├── llm_service.py       # DeepSeek LLM 调用
│   │   │   ├── chunker.py           # 父-子块分割
│   │   │   └── document_parser.py   # PDF/DOCX/TXT 解析
│   │   └── models/
│   │       └── schemas.py           # Pydantic 模型
│   ├── Dockerfile
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── views/
│   │   │   ├── ChatView.vue         # 问答界面（SSE 流式）
│   │   │   └── DocumentManager.vue  # 文档管理
│   │   ├── components/
│   │   │   ├── ChatMessage.vue      # 消息组件
│   │   │   ├── FileUpload.vue       # 文件上传
│   │   │   └── DocList.vue          # 文档列表
│   │   ├── api/index.js             # API 封装（Axios + SSE Fetch）
│   │   └── router/index.js          # Vue Router
│   ├── Dockerfile
│   └── nginx.conf
├── docker-compose.yml               # etcd + MinIO + Milvus
├── .env.example                     # 环境变量模板
└── cache/                           # HuggingFace 模型缓存
```

## 分块策略（Parent-Child）

文档经过两次分割：
- **父块**（800 字符）：为 LLM 提供完整上下文
- **子块**（200 字符）：索引到 Milvus 用于检索

每个子块关联一个父块，LLM 上下文构建时按 `parent_doc_id` 去重，确保每段父块在 Prompt 中只出现一次。

## 环境变量

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `DEEPSEEK_API_KEY` | DeepSeek API 密钥（必填） | - |
| `DEEPSEEK_MODEL` | LLM 模型 | `deepseek-chat` |
| `MILVUS_URI` | Milvus 地址 | `http://localhost:19530` |
| `BGE_DEVICE` | 嵌入模型设备 | `cpu` |
| `RERANKER_DEVICE` | 重排序模型设备 | `cpu` |
| `HYBRID_DENSE_TOP_K` | Dense 检索候选数 | 50 |
| `HYBRID_SPARSE_TOP_K` | Sparse 检索候选数 | 50 |
| `RERANKER_TOP_N` | 重排序后文档数 | 5 |
| `PARENT_CHUNK_SIZE` | 父块大小 | 800 |
| `CHILD_CHUNK_SIZE` | 子块大小 | 200 |

完整配置见 `.env.example`。

## API 文档

启动后端后访问：http://localhost:8000/docs
