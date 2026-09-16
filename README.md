# Enterprise RAG — 企业知识库问答系统

基于 Agentic RAG 的企业知识库问答系统：文档上传 → 混合检索（Dense + Sparse + RRF）→ 重排序 → LLM 生成带来源引用的答案。内置意图路由、查询改写、多跳分解、检索充分性评估与纠错重试，并支持多会话与流式输出。

## 技术栈

| 层级 | 技术 |
|------|------|
| 前端 | Vue 3 + Vite + Element Plus + SSE 流式响应 |
| 后端 | FastAPI + LangGraph（Agent 编排）+ LangChain |
| 向量库 | Milvus 2.4（Dense + Sparse 混合检索 + RRF 融合） |
| 嵌入模型 | BGE-M3（Dense + Sparse 双向量） |
| 重排序 | BGE-Reranker-v2-M3 |
| LLM | DeepSeek API（`langchain-openai` 兼容接口） |
| 会话记忆 | LangGraph Checkpointer（SQLite，可切内存） |
| 文档解析 | PDF（pdfplumber）、DOCX（python-docx）、TXT / Markdown（chardet） |
| 基础设施 | Docker Compose（etcd + MinIO + Milvus） |

## 检索流程

```
Query → BGE-M3 编码（Dense + Sparse 向量）
      → Milvus hybrid_search（IVF_FLAT + SPARSE_WAND → RRF 融合）
      → BGE-Reranker-v2-m3 重排序
      → LLM（DeepSeek）结合父块上下文 → 答案 + 来源引用
```

## Agent 流程

```
route ─(direct)───────────────────────────────────────────▶ generate
      ├(knowledge)─▶ rewrite ──▶ agent_retrieve（工具循环）─┐
      └(multihop)──▶ decompose ─▶ parallel_retrieve ────────┤
                                                            ▼
                                            grade ─┬─ sufficient ──▶ generate
                                                   └─ insufficient ─▶ refine ─▶ agent_retrieve
```

- `route`：意图分类（闲聊直答 / 单主题检索 / 多跳）
- `rewrite`：结合会话历史消解指代，生成独立检索查询
- `decompose`：多跳问题拆成 2–4 个子问题，`parallel_retrieve` 一次批量嵌入后并发检索
- `agent_retrieve`：受 `AGENT_MAX_TOOL_CALLS` 约束的工具循环（`search_knowledge_base` / `list_documents`），保证回答 grounded
- `grade` / `refine`：评估检索是否足以回答；不足时改写查询重试（重试时自动扩大召回窗口）
- `generate`：流式生成答案，并按答案中实际引用的编号输出来源

## 快速开始

### 1. 启动基础设施

```bash
docker compose up -d     # etcd + MinIO + Milvus
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

- `.env` 统一放在仓库根目录，程序按仓库根解析，与启动目录无关。
- 模型默认按 HuggingFace 模型 ID 加载（首次自动下载）；已本地下载时可用 `BGE_MODEL_PATH` / `RERANKER_MODEL_PATH` 指向本地目录。
- `SKIP_MODELS=1` 仅启动 API，不加载嵌入/重排序模型；此时问答与上传会返回 503（用于纯前端联调）。
- 启动完成后可访问 http://localhost:8000/api/health 查看各服务就绪状态。

### 3. 启动前端

```bash
cd frontend
npm install
npm run dev                     # → http://localhost:5173（/api 由 Vite 代理到 8000）
```

## 功能说明

### 文档管理

- 支持格式：PDF / DOCX / TXT / Markdown；旧版 `.doc` 请先另存为 `.docx`。
- 单文件默认上限 50MB（`MAX_UPLOAD_SIZE_MB`）。
- 同名文件重复上传会**覆盖**旧版本（旧分块与磁盘文件一并清理）。
- 删除文档按标题删除全部关联分块，并同步删除服务器上的原始文件。

### 问答与会话

- 流式输出，前端实时展示「Agent 轨迹」（路由、改写、工具调用、评估、纠错等步骤）。
- 答案中的 `[n]` 标记用于来源归属，界面只展示来源文档名。
- 多会话：前端会话列表存于浏览器 `localStorage`；每个会话用独立 `threadId` 对应后端会话记忆（默认 SQLite `backend/data/checkpoints.sqlite`，重启不丢）。删除前端会话不会清理后端记忆，「清空对话」会轮换线程从零开始。

## API

| 接口 | 说明 |
|------|------|
| `GET /api/health` | 健康检查（Milvus / 嵌入 / 重排 / LLM / checkpointer 状态） |
| `POST /api/documents/upload` | 上传并索引文档（multipart，字段名 `file`） |
| `GET /api/documents/list` | 文档标题列表 |
| `DELETE /api/documents/{title}` | 按标题删除文档（不存在返回 404） |
| `POST /api/query` | 同步问答，返回 `{answer, sources}` |
| `POST /api/query/stream` | SSE 流式问答 |

`/api/query` 与 `/api/query/stream` 请求体：

```json
{
  "question": "省外出差住宿标准是多少？",
  "session_id": "会话 threadId",
  "history": [{"role": "user", "content": "..."}]
}
```

`history` 仅在新线程时用于播种上下文，之后由服务端记忆接管。SSE 事件类型：

| type | 含义 |
|------|------|
| `status` | 处理进度提示 |
| `step` | Agent 轨迹（`node` + `detail`） |
| `content` | 答案增量片段 |
| `sources` | 引用来源列表 |
| `error` | 错误信息 |
| `[DONE]` | 流结束标记 |

完整交互文档：启动后端后访问 http://localhost:8000/docs

## 分块策略（Parent-Child）

文档经 `RecursiveCharacterTextSplitter` 两次分割：

- **父块**（默认 1000 字符）：作为 LLM 上下文，保证语义完整
- **子块**（默认 250 字符）：索引到 Milvus 用于检索，定位更精确

子块一定包含于所属父块，通过 `parent_doc_id` 关联；构建 Prompt 时按父块去重，且最多取 `CONTEXT_MAX_DOCS` 个不同文档的分块，避免跨文档噪声。

## 项目结构

```
├── backend/
│   ├── app/
│   │   ├── main.py                  # FastAPI 入口、生命周期与健康检查
│   │   ├── config.py                # 配置加载（仓库根 .env，按仓库根解析相对路径）
│   │   ├── api/
│   │   │   ├── documents.py         # 文档上传/列表/删除
│   │   │   └── query.py             # 问答接口（同步 + SSE 流式）
│   │   ├── agent/
│   │   │   ├── graph.py             # LangGraph 图编排
│   │   │   ├── nodes.py             # route/rewrite/decompose/retrieve/grade/refine/generate
│   │   │   ├── tools.py             # 检索工具层（工具循环 + 并行检索）
│   │   │   ├── prompts.py           # 各节点提示词
│   │   │   ├── state.py             # 图状态定义
│   │   │   └── checkpointer.py      # 会话记忆（SQLite / 内存）
│   │   ├── services/
│   │   │   ├── milvus_store.py      # Milvus 客户端（Schema、混合检索、增删查）
│   │   │   ├── retriever.py         # HybridRetriever（LangChain BaseRetriever）
│   │   │   ├── embedding_service.py # BGE-M3 嵌入（带查询缓存、线程安全锁）
│   │   │   ├── reranker_service.py  # BGE-Reranker 重排序
│   │   │   ├── llm_service.py       # DeepSeek LLM 调用、上下文编号与来源抽取
│   │   │   ├── chunker.py           # 父-子块分割
│   │   │   ├── document_parser.py   # PDF/DOCX/TXT/MD 解析
│   │   │   └── tokenizer_patch.py   # transformers 5.x 兼容补丁
│   │   └── models/
│   │       └── schemas.py           # Pydantic 模型
│   ├── tests/                       # 单元测试（假服务，不依赖 Milvus/模型）
│   ├── ragas_eval/                  # 评测工具（测试集生成、检索/生成/拒答指标）
│   ├── pytest.ini
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── views/
│   │   │   ├── ChatView.vue         # 问答界面（多会话 + SSE 流式 + Agent 轨迹）
│   │   │   └── DocumentManager.vue  # 文档管理
│   │   ├── components/
│   │   │   ├── ChatMessage.vue      # 消息组件
│   │   │   ├── AgentTrace.vue       # Agent 轨迹面板
│   │   │   ├── FileUpload.vue       # 文件上传
│   │   │   └── DocList.vue          # 文档列表
│   │   ├── composables/
│   │   │   └── useSessions.js       # 多会话状态（localStorage 持久化）
│   │   ├── api/index.js             # API 封装（Axios + SSE Fetch）
│   │   └── router/index.js          # Vue Router
│   └── package.json
├── docker-compose.yml               # etcd + MinIO + Milvus
├── .env.example                     # 环境变量模板
└── cache/                           # HuggingFace 模型缓存（可选）
```

## 环境变量

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `DEEPSEEK_API_KEY` | DeepSeek API 密钥（必填） | - |
| `DEEPSEEK_MODEL` | LLM 模型 | `deepseek-chat` |
| `DEEPSEEK_BASE_URL` | API 地址 | `https://api.deepseek.com` |
| `DEEPSEEK_TEMPERATURE` | 采样温度 | `0.1` |
| `DEEPSEEK_MAX_TOKENS` | 最大输出长度 | `1024` |
| `DEEPSEEK_TIMEOUT` | 请求超时（秒） | `60` |
| `DEEPSEEK_MAX_RETRIES` | 失败重试次数 | `2` |
| `MILVUS_URI` | Milvus 地址 | `http://localhost:19530` |
| `MILVUS_COLLECTION` | 集合名 | `knowledge_base` |
| `MILVUS_NPROBE` | IVF_FLAT 检索探测数 | `32` |
| `BGE_MODEL_PATH` | 嵌入模型路径/ID | `BAAI/bge-m3` |
| `BGE_DEVICE` | 嵌入模型设备 | `cuda` |
| `RERANKER_MODEL_PATH` | 重排序模型路径/ID | `BAAI/bge-reranker-v2-m3` |
| `RERANKER_DEVICE` | 重排序模型设备 | `cuda` |
| `RERANKER_TOP_N` | 重排序后保留的分块数 | `10` |
| `HYBRID_DENSE_TOP_K` | Dense 召回候选数 | `50` |
| `HYBRID_SPARSE_TOP_K` | Sparse 召回候选数 | `50` |
| `RRF_K` | RRF 融合平滑参数 | `60` |
| `HYBRID_FUSION_TOP_K` | 融合后候选数 | `30` |
| `PARENT_CHUNK_SIZE` | 父块大小 | `1000` |
| `CHILD_CHUNK_SIZE` | 子块大小 | `250` |
| `CHUNK_OVERLAP` | 分块重叠 | `100` |
| `UPLOAD_DIR` | 上传目录（相对路径按仓库根解析） | `backend/uploads` |
| `MAX_UPLOAD_SIZE_MB` | 单文件大小上限 | `50` |
| `SKIP_MODELS` | `1` 时不加载嵌入/重排序模型 | `0` |
| `AGENT_HISTORY_TURNS` | 供查询改写使用的历史轮数 | `6` |
| `AGENT_MAX_ITERATIONS` | 纠错检索最大轮数 | `2` |
| `AGENT_MAX_TOOL_CALLS` | 工具循环最大调用次数 | `3` |
| `AGENT_RETRY_FUSION_TOP_K` | 纠错重试时的召回窗口（不小于 `HYBRID_FUSION_TOP_K`） | `50` |
| `CONTEXT_MAX_DOCS` | 进入评估/生成 Prompt 的最大文档数 | `5` |
| `AGENT_CHECKPOINTER_BACKEND` | 会话记忆后端：`sqlite` / `memory` | `sqlite` |
| `AGENT_CHECKPOINTER_PATH` | SQLite 记忆文件（相对仓库根） | `backend/data/checkpoints.sqlite` |

## 测试

```bash
cd backend
pytest            # 单元测试使用假服务，无需 Milvus、模型或网络
```

## 评测工具（可选）

`backend/ragas_eval/` 提供合成测试集生成与指标评测（检索 Hit@k / recall / MRR / evidence、Ragas 生成指标、拒答/幻觉率、判卷）：

```bash
cd backend
python -m ragas_eval.run_eval generate --num-questions 3 --output testset.json
python -m ragas_eval.run_eval eval --testset testset.json --mode full
```

## 备注

- 当前 `docker-compose.yml` 仅编排 etcd / MinIO / Milvus 基础设施，前后端以本地开发方式运行。
- 首次启动会加载嵌入与重排序模型（GPU 上约数秒，CPU 上较慢），请以 `/api/health` 为准。
