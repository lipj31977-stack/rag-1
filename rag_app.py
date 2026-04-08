#!/usr/bin/env python3
"""
完整RAG问答系统 - 单文件版本
使用 TF-IDF 向量化 + 内置HTTP服务器 + 内嵌网页前端
运行: python3 rag_app.py
然后浏览器打开 http://localhost:8080
"""

import os
import re
import json
import numpy as np
import tempfile
import io
from pathlib import Path

# -------- 加载 .env 文件 --------
def _load_dotenv(path=None):
    env_path = path or Path(__file__).parent / ".env"
    if not Path(env_path).exists():
        return
    with open(env_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if '=' in line:
                key, _, value = line.partition('=')
                key, value = key.strip(), value.strip().strip('"').strip("'")
                os.environ.setdefault(key, value)

_load_dotenv()
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import parse_qs, urlparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ======================== 配置 ========================
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K_RESULTS = 5
CHAT_HISTORY_PATH = DATA_DIR / "chat_history.json"
VECTOR_STORE_PATH = DATA_DIR / "vector_store.json"

# ======================== 文本处理 ========================

def extract_text(file_path: str) -> str:
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read().strip()


def split_text(text: str) -> list:
    sentences = re.split(r'[。！？\n]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]

    chunks, current, length = [], "", 0
    for s in sentences:
        sl = len(s)
        if length + sl > CHUNK_SIZE and current:
            chunks.append(current.strip())
            current, length = "", 0
        current = s if not current else current + "。" + s
        length = len(current)
    if current.strip():
        chunks.append(current.strip())
    return chunks


# ======================== 向量存储 ========================

class VectorStore:
    def __init__(self):
        self.chunks = []
        self.metadata = []
        self.vectorizer = TfidfVectorizer(
            analyzer='char', ngram_range=(1, 3), max_features=10000
        )
        self.tfidf_matrix = None
        self._load()

    def _load(self):
        if VECTOR_STORE_PATH.exists():
            try:
                with open(VECTOR_STORE_PATH, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                self.chunks = data.get('chunks', [])
                self.metadata = data.get('metadata', [])
                if self.chunks:
                    self.tfidf_matrix = self.vectorizer.fit_transform(self.chunks)
                print(f"✅ 已加载 {len(self.chunks)} 个文本块")
            except Exception as e:
                print(f"⚠️ 加载失败: {e}")

    def add(self, chunks, metadatas):
        self.chunks.extend(chunks)
        self.metadata.extend(metadatas)
        self.tfidf_matrix = self.vectorizer.fit_transform(self.chunks)
        print(f"✅ 已添加 {len(chunks)} 个文本块，总计 {len(self.chunks)}")

    def search(self, query, k=5):
        if not self.chunks:
            return []
        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        top_idx = scores.argsort()[::-1][:k]
        results = []
        for i in top_idx:
            if scores[i] > 0.001:
                results.append({
                    'chunk': self.chunks[i],
                    'metadata': self.metadata[i],
                    'score': float(scores[i])
                })
        return results

    def save(self):
        with open(VECTOR_STORE_PATH, 'w', encoding='utf-8') as f:
            json.dump({'chunks': self.chunks, 'metadata': self.metadata}, f, ensure_ascii=False, indent=2)

    def clear(self):
        self.chunks, self.metadata, self.tfidf_matrix = [], [], None
        if VECTOR_STORE_PATH.exists():
            VECTOR_STORE_PATH.unlink()

    def stats(self):
        return {'total_chunks': len(self.chunks)}


store = VectorStore()


# ======================== 聊天记录 ========================

def save_chat(query, answer, sources):
    history = load_history()
    history.insert(0, {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'user_query': query,
        'assistant_response': answer,
        'sources': sources
    })
    history = history[:100]
    with open(CHAT_HISTORY_PATH, 'w', encoding='utf-8') as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


def load_history():
    if not CHAT_HISTORY_PATH.exists():
        return []
    try:
        with open(CHAT_HISTORY_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    except:
        return []


def clear_history():
    if CHAT_HISTORY_PATH.exists():
        CHAT_HISTORY_PATH.unlink()


# ======================== RAG 核心 ========================

def process_document(file_path, filename="unknown"):
    text = extract_text(file_path)
    print(f"📄 提取文本: {len(text)} 字符")
    if not text.strip():
        print("⚠️ 文件内容为空")
        return 0
    chunks = split_text(text)
    print(f"📄 分块结果: {len(chunks)} 个块")
    metadatas = [{'source': filename, 'chunk_id': f"{filename}_{i}", 'position': i} for i, _ in enumerate(chunks)]
    store.add(chunks, metadatas)
    store.save()
    return len(chunks)


def rag_answer(query):
    print(f"🔍 查询: {query}, 知识库大小: {len(store.chunks)} 块")
    results = store.search(query, k=TOP_K_RESULTS)
    print(f"🔍 检索到 {len(results)} 条结果")
    if not results:
        return "抱歉，知识库中没有找到相关信息。请先上传文档。", []

    # 构建回答（无外部LLM时，直接展示检索结果作为回答）
    parts = []
    sources = set()
    for i, r in enumerate(results, 1):
        src = r['metadata'].get('source', '未知')
        sources.add(src)
        score_pct = f"{r['score']*100:.1f}%"
        parts.append(f"**片段{i}** (相关度: {score_pct}, 来源: {src}):\n{r['chunk']}")

    answer = f"根据知识库检索，以下是与您问题最相关的内容：\n\n" + "\n\n---\n\n".join(parts)
    answer += f"\n\n📚 参考来源: {', '.join(sources)}"

    # 提示：如果配置了Qwen API Key，可以用LLM生成更自然的回答
    if os.getenv('QWEN_API_KEY'):
        answer = _call_qwen(query, results) or answer

    return answer, list(sources)


def _call_qwen(query, results):
    """可选：调用Qwen API生成回答"""
    try:
        import requests
        context = "\n\n".join([f"[来源{i+1}] {r['chunk']}" for i, r in enumerate(results)])
        prompt = f"基于以下参考内容回答问题。\n\n参考内容：\n{context}\n\n问题：{query}\n\n请用中文回答："
        resp = requests.post(
            "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
            headers={"Authorization": f"Bearer {os.getenv('QWEN_API_KEY')}", "Content-Type": "application/json"},
            json={"model": os.getenv("QWEN_MODEL_NAME", "qwen-max"), "messages": [{"role": "user", "content": prompt}]},
            timeout=30
        )
        if resp.status_code == 200:
            data = resp.json()
            return data["choices"][0]["message"]["content"]
    except:
        pass
    return None


# ======================== 网页前端 ========================

HTML_PAGE = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>RAG 知识问答系统</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+SC:wght@300;400;500;700&display=swap');

  * { margin: 0; padding: 0; box-sizing: border-box; }

  :root {
    --bg: #0f1117;
    --surface: #1a1d27;
    --surface2: #242836;
    --border: #2e3345;
    --text: #e4e6f0;
    --text2: #8b8fa3;
    --accent: #6c8aff;
    --accent2: #4a6aff;
    --green: #4ecb71;
    --orange: #f0a050;
    --red: #f06060;
  }

  body {
    font-family: 'Noto Sans SC', -apple-system, sans-serif;
    background: var(--bg);
    color: var(--text);
    height: 100vh;
    display: flex;
    flex-direction: column;
    overflow: hidden;
  }

  header {
    background: var(--surface);
    border-bottom: 1px solid var(--border);
    padding: 16px 24px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-shrink: 0;
  }

  header h1 {
    font-size: 18px;
    font-weight: 500;
    letter-spacing: 0.5px;
  }

  header h1 span { color: var(--accent); }

  .header-actions { display: flex; gap: 8px; align-items: center; }

  .badge {
    background: var(--surface2);
    border: 1px solid var(--border);
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 12px;
    color: var(--text2);
  }

  .badge.active { border-color: var(--green); color: var(--green); }

  .btn {
    background: var(--surface2);
    border: 1px solid var(--border);
    color: var(--text);
    padding: 8px 16px;
    border-radius: 8px;
    cursor: pointer;
    font-size: 13px;
    font-family: inherit;
    transition: all 0.2s;
  }

  .btn:hover { border-color: var(--accent); color: var(--accent); }
  .btn.primary { background: var(--accent2); border-color: var(--accent2); color: #fff; }
  .btn.primary:hover { background: var(--accent); }
  .btn.danger:hover { border-color: var(--red); color: var(--red); }

  .main-area {
    flex: 1;
    display: flex;
    flex-direction: column;
    overflow: hidden;
    max-width: 860px;
    width: 100%;
    margin: 0 auto;
    padding: 0 16px;
  }

  #chat-box {
    flex: 1;
    overflow-y: auto;
    padding: 24px 0;
    display: flex;
    flex-direction: column;
    gap: 16px;
    scrollbar-width: thin;
    scrollbar-color: var(--border) transparent;
  }

  .msg {
    max-width: 85%;
    padding: 14px 18px;
    border-radius: 16px;
    font-size: 14px;
    line-height: 1.7;
    white-space: pre-wrap;
    word-break: break-word;
    animation: fadeIn 0.3s ease;
  }

  @keyframes fadeIn { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: none; } }

  .msg.user {
    align-self: flex-end;
    background: var(--accent2);
    color: #fff;
    border-bottom-right-radius: 4px;
  }

  .msg.bot {
    align-self: flex-start;
    background: var(--surface2);
    border: 1px solid var(--border);
    border-bottom-left-radius: 4px;
  }

  .msg.system {
    align-self: center;
    background: transparent;
    border: 1px dashed var(--border);
    color: var(--text2);
    font-size: 12px;
    padding: 8px 16px;
    border-radius: 8px;
  }

  .input-area {
    padding: 16px 0 24px;
    display: flex;
    gap: 8px;
    flex-shrink: 0;
  }

  .input-area input[type="text"] {
    flex: 1;
    background: var(--surface);
    border: 1px solid var(--border);
    color: var(--text);
    padding: 14px 18px;
    border-radius: 12px;
    font-size: 14px;
    font-family: inherit;
    outline: none;
    transition: border-color 0.2s;
  }

  .input-area input[type="text"]:focus { border-color: var(--accent); }
  .input-area input[type="text"]::placeholder { color: var(--text2); }

  /* Upload overlay */
  #upload-overlay {
    display: none;
    position: fixed; inset: 0;
    background: rgba(0,0,0,0.6);
    backdrop-filter: blur(4px);
    z-index: 100;
    justify-content: center;
    align-items: center;
  }

  #upload-overlay.show { display: flex; }

  .upload-box {
    background: var(--surface);
    border: 2px dashed var(--border);
    border-radius: 16px;
    padding: 48px;
    text-align: center;
    min-width: 400px;
    transition: border-color 0.2s;
  }

  .upload-box.dragover { border-color: var(--accent); }

  .upload-box h2 { font-size: 16px; font-weight: 500; margin-bottom: 8px; }
  .upload-box p { font-size: 13px; color: var(--text2); margin-bottom: 20px; }

  #file-input { display: none; }

  .loading { color: var(--orange); }
</style>
</head>
<body>

<header>
  <h1><span>◆</span> RAG 知识问答系统</h1>
  <div class="header-actions">
    <span class="badge" id="kb-badge">知识库: 0 块</span>
    <button class="btn" onclick="showUpload()">上传文档</button>
    <button class="btn" onclick="loadHistory()">历史记录</button>
    <button class="btn danger" onclick="clearKB()">清空知识库</button>
  </div>
</header>

<div class="main-area">
  <div id="chat-box">
    <div class="msg system">欢迎使用RAG知识问答系统。请先上传TXT文档，然后输入问题进行提问。</div>
  </div>
  <div class="input-area">
    <input type="text" id="query-input" placeholder="输入您的问题..." onkeydown="if(event.key==='Enter')ask()">
    <button class="btn primary" onclick="ask()">发送</button>
  </div>
</div>

<div id="upload-overlay">
  <div class="upload-box" id="upload-box">
    <h2>上传 TXT 文档</h2>
    <p>将文件拖放到此处，或点击选择文件</p>
    <button class="btn primary" onclick="document.getElementById('file-input').click()">选择文件</button>
    <button class="btn" onclick="hideUpload()" style="margin-left:8px">取消</button>
    <input type="file" id="file-input" accept=".txt" onchange="uploadFile(this.files[0])">
    <p id="upload-status" style="margin-top:16px"></p>
  </div>
</div>

<script>
const chatBox = document.getElementById('chat-box');

function addMsg(text, cls) {
  const d = document.createElement('div');
  d.className = 'msg ' + cls;
  d.textContent = text;
  chatBox.appendChild(d);
  chatBox.scrollTop = chatBox.scrollHeight;
}

async function ask() {
  const input = document.getElementById('query-input');
  const q = input.value.trim();
  if (!q) return;
  input.value = '';
  addMsg(q, 'user');
  addMsg('正在检索知识库...', 'system loading');

  try {
    const res = await fetch('/api/ask', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({query: q})
    });
    const data = await res.json();
    // Remove loading msg
    chatBox.removeChild(chatBox.lastChild);
    addMsg(data.answer, 'bot');
  } catch(e) {
    chatBox.removeChild(chatBox.lastChild);
    addMsg('请求失败: ' + e.message, 'system');
  }
}

function showUpload() { document.getElementById('upload-overlay').classList.add('show'); }
function hideUpload() { document.getElementById('upload-overlay').classList.remove('show'); document.getElementById('upload-status').textContent = ''; }

async function uploadFile(file) {
  if (!file) return;
  const status = document.getElementById('upload-status');
  status.textContent = '上传中...';
  status.className = 'loading';

  const formData = new FormData();
  formData.append('file', file);

  try {
    const res = await fetch('/api/upload', { method: 'POST', body: formData });
    const data = await res.json();
    status.textContent = data.message || '上传成功';
    status.className = '';
    refreshStats();
    addMsg('📄 文档 "' + file.name + '" 已上传并处理完成', 'system');
    setTimeout(hideUpload, 1500);
  } catch(e) {
    status.textContent = '上传失败: ' + e.message;
  }
}

async function refreshStats() {
  try {
    const res = await fetch('/api/stats');
    const data = await res.json();
    document.getElementById('kb-badge').textContent = '知识库: ' + data.total_chunks + ' 块';
    if (data.total_chunks > 0) document.getElementById('kb-badge').classList.add('active');
    else document.getElementById('kb-badge').classList.remove('active');
  } catch(e) {}
}

async function loadHistory() {
  try {
    const res = await fetch('/api/history');
    const data = await res.json();
    if (!data.messages || data.messages.length === 0) {
      addMsg('暂无历史记录', 'system');
      return;
    }
    addMsg(`--- 最近 ${data.messages.length} 条历史记录 ---`, 'system');
    data.messages.slice(0, 10).reverse().forEach(m => {
      addMsg(m.user_query, 'user');
      addMsg(m.assistant_response, 'bot');
    });
  } catch(e) {}
}

async function clearKB() {
  if (!confirm('确定要清空知识库吗？')) return;
  await fetch('/api/clear-knowledge', { method: 'POST' });
  refreshStats();
  addMsg('🗑️ 知识库已清空', 'system');
}

// Drag and drop
const uploadBox = document.getElementById('upload-box');
uploadBox.addEventListener('dragover', e => { e.preventDefault(); uploadBox.classList.add('dragover'); });
uploadBox.addEventListener('dragleave', () => uploadBox.classList.remove('dragover'));
uploadBox.addEventListener('drop', e => { e.preventDefault(); uploadBox.classList.remove('dragover'); if(e.dataTransfer.files.length) uploadFile(e.dataTransfer.files[0]); });

refreshStats();
</script>
</body>
</html>"""


# ======================== HTTP 服务器 ========================

class RAGHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass  # 静默日志

    def _json_response(self, data, code=200):
        self.send_response(code)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.end_headers()
        self.wfile.write(json.dumps(data, ensure_ascii=False).encode('utf-8'))

    def _html_response(self, html):
        self.send_response(200)
        self.send_header('Content-Type', 'text/html; charset=utf-8')
        self.end_headers()
        self.wfile.write(html.encode('utf-8'))

    def do_GET(self):
        path = urlparse(self.path).path

        if path == '/' or path == '':
            self._html_response(HTML_PAGE)

        elif path == '/api/stats':
            self._json_response(store.stats())

        elif path == '/api/history':
            history = load_history()
            self._json_response({'total_messages': len(history), 'messages': history[:20]})

        else:
            self.send_error(404)

    def do_POST(self):
        path = urlparse(self.path).path

        if path == '/api/ask':
            length = int(self.headers.get('Content-Length', 0))
            body = json.loads(self.rfile.read(length))
            query = body.get('query', '').strip()
            if not query:
                self._json_response({'error': '问题不能为空'}, 400)
                return
            answer, sources = rag_answer(query)
            save_chat(query, answer, sources)
            self._json_response({'answer': answer, 'sources': sources})

        elif path == '/api/upload':
            content_type = self.headers.get('Content-Type', '')
            if 'multipart/form-data' not in content_type:
                self._json_response({'error': '请使用multipart上传'}, 400)
                return

            # Parse multipart
            boundary = content_type.split('boundary=')[1].encode()
            length = int(self.headers.get('Content-Length', 0))
            data = self.rfile.read(length)

            # Simple multipart parse
            filename, file_data = self._parse_multipart(data, boundary)
            if not filename or not filename.lower().endswith('.txt'):
                self._json_response({'error': '只支持TXT文件'}, 400)
                return

            # Save temp and process
            if not file_data or len(file_data) == 0:
                self._json_response({'error': '文件内容为空'}, 400)
                return

            tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.txt', mode='wb')
            tmp.write(file_data)
            tmp.close()

            try:
                count = process_document(tmp.name, filename)
                if count == 0:
                    self._json_response({'error': '文档中未提取到有效文本块'}, 400)
                else:
                    print(f"✅ 上传完成: {filename}, {count} 个文本块, 知识库总计 {len(store.chunks)} 块")
                    self._json_response({'message': f'文档处理成功，生成 {count} 个文本块', 'chunks': count})
            except Exception as e:
                print(f"❌ 处理失败: {e}")
                self._json_response({'error': str(e)}, 500)
            finally:
                os.unlink(tmp.name)

        elif path == '/api/clear-knowledge':
            store.clear()
            self._json_response({'message': '知识库已清空'})

        elif path == '/api/clear-history':
            clear_history()
            self._json_response({'message': '聊天记录已清空'})

        else:
            self.send_error(404)

    def _parse_multipart(self, data, boundary):
        """健壮的 multipart 解析"""
        # 确保 boundary 是 bytes
        if isinstance(boundary, str):
            boundary = boundary.encode()

        delimiter = b'--' + boundary
        end_delimiter = delimiter + b'--'

        # 去掉结束标记
        if end_delimiter in data:
            data = data[:data.index(end_delimiter)]

        parts = data.split(delimiter)
        for part in parts:
            part = part.strip(b'\r\n')
            if not part or b'filename=' not in part:
                continue

            header_end = part.find(b'\r\n\r\n')
            if header_end < 0:
                continue

            header = part[:header_end].decode('utf-8', errors='ignore')
            body = part[header_end + 4:]

            # 去掉末尾可能的 \r\n
            if body.endswith(b'\r\n'):
                body = body[:-2]

            # 提取文件名（兼容双引号和没引号的情况）
            fn_match = re.search(r'filename="([^"]+)"', header)
            if not fn_match:
                fn_match = re.search(r"filename=([^\s;]+)", header)
            filename = fn_match.group(1) if fn_match else 'unknown.txt'

            print(f"📎 解析到文件: {filename}, 大小: {len(body)} 字节")
            return filename, body

        return None, None


def main():
    port = int(os.getenv('PORT', 8080))
    server = HTTPServer(('0.0.0.0', port), RAGHandler)
    print(f"""
╔══════════════════════════════════════════╗
║    RAG 知识问答系统已启动                ║
║    访问地址: http://localhost:{port}       ║
║                                          ║
║    功能:                                 ║
║      · 上传 TXT 文档建立知识库           ║
║      · 输入问题进行智能检索              ║
║      · 查看聊天历史记录                  ║
║                                          ║
║    按 Ctrl+C 停止服务                    ║
╚══════════════════════════════════════════╝
""")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n🛑 服务已停止")
        server.server_close()


if __name__ == "__main__":
    main()
