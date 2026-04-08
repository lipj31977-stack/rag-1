"""
RAG系统配置文件
"""

import os
from pathlib import Path

# 基础路径
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

# 向量存储配置
VECTOR_STORE_PATH = DATA_DIR / "vector_store.faiss"
VECTOR_DIMENSION = 384  # all-MiniLM-L6-v2 维度
DOCUMENTS_PATH = DATA_DIR / "documents"
DOCUMENTS_PATH.mkdir(exist_ok=True)

# 文本分块配置
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# 检索配置
TOP_K_RESULTS = 5

# 嵌入模型
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# 聊天记录
CHAT_HISTORY_PATH = DATA_DIR / "chat_history.json"

# Qwen API 配置
QWEN_API_KEY = os.getenv("QWEN_API_KEY", "")
QWEN_MODEL_NAME = os.getenv("QWEN_MODEL_NAME", "qwen-max")
