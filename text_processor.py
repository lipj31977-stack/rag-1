"""
文本处理工具 - 分块和提取
"""

import re
from typing import List, Tuple
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CHUNK_SIZE, CHUNK_OVERLAP


def extract_text_from_txt(file_path: str) -> str:
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        return text.strip()
    except Exception as e:
        raise Exception(f"读取TXT文件失败: {e}")


def split_text(text: str, chunk_size: int = None, overlap: int = None) -> List[str]:
    if chunk_size is None:
        chunk_size = CHUNK_SIZE
    if overlap is None:
        overlap = CHUNK_OVERLAP

    sentences = re.split(r'[。！？\n]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]

    chunks = []
    current_chunk = ""
    current_length = 0

    for sentence in sentences:
        sentence_length = len(sentence)
        if current_length + sentence_length > chunk_size and current_chunk:
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
                current_chunk = ""
                current_length = 0

        if not current_chunk:
            current_chunk = sentence
            current_length = sentence_length
        else:
            current_chunk += "。" + sentence
            current_length += sentence_length + 1

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks


def create_chunks_with_metadata(chunks: List[str], filename: str) -> Tuple[List[str], List[dict]]:
    metadatas = []
    for i, chunk in enumerate(chunks):
        metadatas.append({
            'source': filename,
            'chunk_id': f"{filename}_{i}",
            'position': i
        })
    return chunks, metadatas
