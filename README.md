# RAG 知识问答系统

基于 TF-IDF 向量检索 + 通义千问大模型的本地知识库问答系统。上传 TXT 文档即可通过网页界面进行智能问答。

---

## 项目结构

```
├── rag_app.py      # 主程序（包含全部逻辑 + 内嵌网页前端）
├── .env            # API 密钥与配置
└── data/           # 运行后自动生成，存放向量索引和聊天记录
```

> `config.py` 和 `text_processor.py` 已整合进 `rag_app.py`，**只需上述两个文件即可运行**。

---

## 快速开始

### 1. 环境要求

- Python 3.8+
- 已安装 `numpy` 和 `scikit-learn`

```bash
pip install numpy scikit-learn
```

### 2. 配置 API Key（可选）

编辑 `.env` 文件，填入通义千问 API Key：

```env
QWEN_API_KEY=sk-你的真实key
QWEN_MODEL_NAME=qwen-max
PORT=6666
```

- **有 Key**：检索后调用千问生成自然语言回答
- **无 Key**：直接返回检索到的相关文档片段（同样可用）

API Key 从 [阿里云百炼平台](https://dashscope.console.aliyun.com/) 获取。

### 3. 启动

```bash
python3 rag_app.py
```

浏览器打开 **http://localhost:6666** 即可使用。

---

## 功能说明

| 功能 | 说明 |
|------|------|
| 上传文档 | 支持 TXT 格式，自动分块并建立向量索引 |
| 智能问答 | 输入问题，检索知识库中最相关的内容并生成回答 |
| 聊天记录 | 自动保存最近 100 条对话，支持查看和清空 |
| 清空知识库 | 一键清除所有已索引的文档数据 |

---

## 技术方案

```
用户提问
   │
   ▼
TF-IDF 向量化（字符级 n-gram，适配中文）
   │
   ▼
余弦相似度检索 Top-K 片段
   │
   ▼
┌──────────────────────────────┐
│ 有 QWEN_API_KEY？            │
│  是 → 调用千问 API 生成回答  │
│  否 → 直接返回检索片段       │
└──────────────────────────────┘
   │
   ▼
返回结果 + 来源标注
```

- **向量化**：`sklearn.TfidfVectorizer`，使用 `char_wb` 分析器 + (2,4)-gram，无需分词器即可处理中文
- **检索**：`sklearn.metrics.pairwise.cosine_similarity`
- **存储**：JSON 文件持久化（`data/` 目录下）
- **服务**：Python 标准库 `http.server`，零额外依赖

---

## API 接口

系统同时提供 REST API，可供其他程序调用：

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/` | 网页前端 |
| `POST` | `/api/upload` | 上传 TXT 文档（multipart/form-data） |
| `POST` | `/api/ask` | 提问，body: `{"query": "问题"}` |
| `GET` | `/api/stats` | 获取知识库统计信息 |
| `GET` | `/api/history` | 获取聊天记录 |
| `POST` | `/api/clear-knowledge` | 清空知识库 |
| `POST` | `/api/clear-history` | 清空聊天记录 |

---

## 常见问题

**Q：不配置 API Key 能用吗？**
可以。系统会直接返回检索到的原文片段和相关度评分，只是不会用大模型重新组织语言。

**Q：支持什么格式的文档？**
目前支持 TXT 纯文本文件。如需扩展 PDF/DOCX 支持，可在 `extract_text()` 函数中添加对应解析逻辑。

**Q：数据存在哪里？**
运行后自动在同目录下生成 `data/` 文件夹，包含 `vector_store.json`（向量索引）和 `chat_history.json`（聊天记录）。

