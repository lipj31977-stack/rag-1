# rag-1
第一个简单的rag项目
# 简化版RAG问答系统 - 项目结构

## 📁 完整目录结构

```
RAG_Simplified_Project/
├── README.md                    # 项目说明文档
├── config.py                   # 配置文件
├── main.py                     # FastAPI主程序
├── requirements.txt            # Python依赖列表
├── start.bat                   # Windows启动脚本
└── start.sh                    # Linux/Mac启动脚本
│
├── core/
│   └── qa_system.py          # RAG问答核心逻辑
│
├── utils/
│   ├── chat_history.py       # 聊天记录管理
│   ├── embeddings.py         # 向量化工具
│   ├── text_processor.py     # 文本处理工具
│   └── vector_store.py       # FAISS向量存储管理
│
└── data/                     # 数据存储目录
    ├── documents/           # 上传的TXT文档
    ├── vector_store.faiss   # FAISS索引文件
    └── chat_history.json    # 聊天记录JSON文件
```

## 🚀 核心功能

- **📄 TXT文档处理** - 支持TXT文件上传和自动分块向量化
- **🧠 Qwen云端模型** - Embedding和生成都使用Qwen-max
- **💾 本地FAISS存储** - 向量持久化保存到磁盘
- **💬 聊天记录保存** - JSON格式保存对话历史
- **⚡ FastAPI前端** - 简洁REST接口，无进度条显示
- **🎯 一键启动** - 自动环境检查和依赖安装

## 📋 快速开始

1. **进入项目目录**
   ```bash
   cd "RAG_Simplified_Project"
   ```

2. **安装依赖**
   ```bash
   pip install -r requirements.txt
   ```

3. **配置API密钥**
   - 编辑 `config.py` 或 `.env` 文件
   - 填入你的 Qwen API Key

4. **启动服务**
   ```bash
   python main.py
   ```
   或运行启动脚本：
   - Windows: `start.bat`
   - Linux/Mac: `./start.sh`

5. **访问API**
   - 服务地址: `http://localhost:8000`
   - API文档: `http://localhost:8000/docs`

## 🔌 API接口

### 上传文档
```bash
curl -X POST "http://localhost:8000/upload" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@your_document.txt"
```

### 提问
```bash
curl -X POST "http://localhost:8000/ask" \
     -H "Content-Type: application/json" \
     -d '{"query": "你是什么类型的模型？"}'
```

### 获取聊天历史
```bash
curl "http://localhost:8000/history?limit=10"
```

### 清空聊天记录
```bash
curl -X DELETE "http://localhost:8000/clear-history"
```

### 获取系统统计
```bash
curl "http://localhost:8000/stats"
```

## ⚙️ 配置文件

`config.py` 包含以下配置项：

- `QWEN_API_KEY`: Qwen API密钥
- `QWEN_MODEL_NAME`: Qwen模型名称（默认：qwen-max）
- `EMBEDDING_MODEL_NAME`: 向量化模型（默认：nghuyong/ernie-3.0-nano-zh）
- `VECTOR_DIMENSION`: 向量维度（默认：768）
- `CHUNK_SIZE`: 文本分块大小（默认：500字符）
- `CHUNK_OVERLAP`: 分块重叠大小（默认：50字符）
- `TOP_K_RESULTS`: 搜索结果返回数量（默认：5）

## 📊 数据持久化

- **向量存储**: FAISS索引保存到 `data/vector_store.faiss`
- **聊天记录**: JSON格式保存到 `data/chat_history.json`
- **上传文档**: 保存到 `data/documents/` 目录

## 🛠️ 技术栈

- **后端框架**: FastAPI + Uvicorn
- **向量化**: Sentence Transformers (Ernie-3.0-nano-zh)
- **向量检索**: FAISS
- **语言模型**: Qwen Cloud API
- **数据存储**: JSON + FAISS
- **配置管理**: 环境变量

## 🎯 项目特点

- ✅ **纯TXT支持** - 专注TXT文档处理
- ✅ **云端模型统一** - Embedding和生成都用Qwen
- ✅ **本地向量存储** - 高效的相似度搜索
- ✅ **聊天记录功能** - 完整的对话历史保存
- ✅ **简洁API设计** - 无复杂UI，纯REST接口
- ✅ **一键部署** - 自动环境检测和依赖安装
