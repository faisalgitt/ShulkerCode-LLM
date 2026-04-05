"""
=====================================
 Shulker Code — FastAPI Web UI
 Chat-like coding interface
 Developed by @kopeedev / CyeroX
=====================================

Launch with:
    python main.py webui
    # or directly:
    uvicorn web.app:app --host 0.0.0.0 --port 8000
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, AsyncIterator
import asyncio
import json

app = FastAPI(
    title="Shulker Code API",
    description="Code Intelligence Engine by @kopeedev / CyeroX Development",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model reference (set at startup)
_engine = None
_tokenizer = None


class GenerateRequest(BaseModel):
    prompt: str
    language: str = "python"
    task: str = "gen"
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_k: int = 50
    top_p: float = 0.95
    stream: bool = False


class GenerateResponse(BaseModel):
    generated_code: str
    tokens_generated: int
    model: str


@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    import os
    global _engine, _tokenizer

    model_path = os.environ.get("SHULKER_MODEL_PATH", "checkpoints/final")
    device = os.environ.get("SHULKER_DEVICE", "auto")

    try:
        from model.transformer import ShulkerCodeModel
        from data.tokenizer import ShulkerTokenizer
        from inference.engine import ShulkerInferenceEngine

        model = ShulkerCodeModel.from_pretrained(model_path, device=device)
        tokenizer = ShulkerTokenizer.from_pretrained(model_path)
        _engine = ShulkerInferenceEngine(model, tokenizer, device=device)
        _tokenizer = tokenizer
        print(f"✅ Shulker Code Web UI ready at http://0.0.0.0:8000")
    except Exception as e:
        print(f"⚠️  Could not load model: {e}")
        print("   The API is running, but /generate will fail until a model is loaded.")


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the Web UI."""
    return HTML_UI


@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest):
    """Generate code from a prompt."""
    if _engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Train or download a model first.")

    from inference.engine import GenerationConfig
    config = GenerationConfig(
        max_new_tokens=req.max_new_tokens,
        temperature=req.temperature,
        top_k=req.top_k,
        top_p=req.top_p,
    )

    code = _engine.generate(req.prompt, config=config, language=req.language, task=req.task)
    tokens = len(_tokenizer.encode(code))

    return GenerateResponse(
        generated_code=code,
        tokens_generated=tokens,
        model=_engine.model.config.name,
    )


@app.post("/generate/stream")
async def generate_stream(req: GenerateRequest):
    """Stream generated tokens via Server-Sent Events."""
    if _engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    from inference.engine import GenerationConfig
    config = GenerationConfig(
        max_new_tokens=req.max_new_tokens,
        temperature=req.temperature,
        top_k=req.top_k,
        top_p=req.top_p,
    )

    async def token_stream() -> AsyncIterator[str]:
        loop = asyncio.get_event_loop()

        def blocking_generate():
            tokens = []
            for token in _engine.generate_streaming(req.prompt, config=config,
                                                     language=req.language, task=req.task):
                tokens.append(token)
            return tokens

        # Run blocking generation in thread pool
        tokens = await loop.run_in_executor(None, blocking_generate)
        for token in tokens:
            yield f"data: {json.dumps({'token': token})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(token_stream(), media_type="text/event-stream")


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "ok",
        "model_loaded": _engine is not None,
        "model": _engine.model.config.name if _engine else None,
    }


# ─── Embedded Web UI ────────────────────────
HTML_UI = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Shulker Code — by @kopeedev</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    background: #0d1117; color: #c9d1d9;
    display: flex; flex-direction: column; height: 100vh;
  }
  header {
    background: #161b22; border-bottom: 1px solid #30363d;
    padding: 12px 20px; display: flex; align-items: center; gap: 12px;
  }
  header h1 { font-size: 1.1rem; color: #58a6ff; }
  header span { font-size: 0.75rem; color: #8b949e; }
  .container { display: flex; flex: 1; overflow: hidden; }
  .sidebar {
    width: 240px; background: #161b22;
    border-right: 1px solid #30363d; padding: 16px; overflow-y: auto;
  }
  .sidebar label { font-size: 0.7rem; color: #8b949e; display: block; margin-bottom: 4px; margin-top: 12px; }
  .sidebar select, .sidebar input[type=range], .sidebar input[type=number] {
    width: 100%; background: #0d1117; border: 1px solid #30363d;
    color: #c9d1d9; padding: 6px 8px; border-radius: 6px; font-size: 0.8rem;
  }
  .sidebar input[type=range] { padding: 0; }
  .range-val { font-size: 0.75rem; color: #58a6ff; text-align: right; }
  main { flex: 1; display: flex; flex-direction: column; overflow: hidden; }
  .messages { flex: 1; overflow-y: auto; padding: 16px; display: flex; flex-direction: column; gap: 12px; }
  .msg { border-radius: 8px; padding: 12px 14px; max-width: 90%; }
  .msg.user { background: #1f6feb22; border: 1px solid #1f6feb55; align-self: flex-end; }
  .msg.assistant { background: #161b22; border: 1px solid #30363d; align-self: flex-start; }
  .msg.assistant pre { background: #0d1117; border-radius: 6px; padding: 10px; overflow-x: auto; margin-top: 8px; }
  .msg-label { font-size: 0.65rem; color: #8b949e; margin-bottom: 4px; }
  .input-area {
    padding: 12px 16px; background: #161b22; border-top: 1px solid #30363d;
    display: flex; gap: 8px;
  }
  textarea {
    flex: 1; background: #0d1117; border: 1px solid #30363d; color: #c9d1d9;
    padding: 10px; border-radius: 8px; font-family: inherit; font-size: 0.85rem;
    resize: none; height: 80px;
  }
  textarea:focus { outline: none; border-color: #58a6ff; }
  button {
    background: #238636; color: white; border: none; border-radius: 8px;
    padding: 10px 18px; cursor: pointer; font-size: 0.85rem; font-family: inherit;
    align-self: flex-end;
  }
  button:hover { background: #2ea043; }
  button:disabled { background: #2d333b; color: #8b949e; cursor: not-allowed; }
  .typing { color: #58a6ff; font-size: 0.8rem; }
  .cursor { animation: blink 0.8s step-end infinite; }
  @keyframes blink { 50% { opacity: 0; } }
</style>
</head>
<body>
<header>
  <span style="font-size:1.4rem">🧱</span>
  <h1>Shulker Code</h1>
  <span>Code Intelligence Engine · by @kopeedev / CyeroX Dev</span>
</header>
<div class="container">
  <div class="sidebar">
    <label>Language</label>
    <select id="lang">
      <option value="python">Python</option>
      <option value="javascript">JavaScript</option>
      <option value="typescript">TypeScript</option>
      <option value="cpp">C++</option>
      <option value="java">Java</option>
      <option value="go">Go</option>
      <option value="rust">Rust</option>
    </select>
    <label>Task</label>
    <select id="task">
      <option value="gen">Generate</option>
      <option value="fix">Debug / Fix</option>
      <option value="explain">Explain</option>
      <option value="optimize">Optimize</option>
    </select>
    <label>Temperature: <span class="range-val" id="temp-val">0.7</span></label>
    <input type="range" id="temperature" min="0.1" max="1.5" step="0.1" value="0.7"
      oninput="document.getElementById('temp-val').textContent=this.value">
    <label>Top-K: <span class="range-val" id="topk-val">50</span></label>
    <input type="range" id="top_k" min="1" max="100" step="1" value="50"
      oninput="document.getElementById('topk-val').textContent=this.value">
    <label>Max Tokens</label>
    <input type="number" id="max_tokens" value="512" min="64" max="4096" step="64">
  </div>
  <main>
    <div class="messages" id="messages">
      <div class="msg assistant">
        <div class="msg-label">Shulker Code</div>
        👋 Hello! I'm <strong>Shulker Code</strong>, your local code intelligence engine.<br>
        Describe what you want to build, paste code to debug, or ask me to explain anything.
      </div>
    </div>
    <div class="input-area">
      <textarea id="prompt" placeholder="Write a Python function to sort a list of dicts by a key...
Paste code here to debug or explain..." onkeydown="handleKey(event)"></textarea>
      <button id="send-btn" onclick="sendMessage()">Generate ▶</button>
    </div>
  </main>
</div>
<script>
function handleKey(e) {
  if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) sendMessage();
}
async function sendMessage() {
  const prompt = document.getElementById('prompt').value.trim();
  if (!prompt) return;
  const btn = document.getElementById('send-btn');
  btn.disabled = true;
  addMessage('user', prompt);
  document.getElementById('prompt').value = '';
  const assistantDiv = addMessage('assistant', '<span class="typing">Generating<span class="cursor">▌</span></span>');
  const payload = {
    prompt,
    language: document.getElementById('lang').value,
    task: document.getElementById('task').value,
    temperature: parseFloat(document.getElementById('temperature').value),
    top_k: parseInt(document.getElementById('top_k').value),
    max_new_tokens: parseInt(document.getElementById('max_tokens').value),
  };
  try {
    const res = await fetch('/generate', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(payload),
    });
    const data = await res.json();
    const code = data.generated_code || data.detail || 'No output.';
    assistantDiv.innerHTML = `<div class="msg-label">Shulker Code · ${data.model || ''} · ${data.tokens_generated || 0} tokens</div><pre><code>${escapeHtml(code)}</code></pre>`;
  } catch(e) {
    assistantDiv.innerHTML = `<div class="msg-label">Error</div>❌ ${e.message}`;
  }
  btn.disabled = false;
  document.getElementById('messages').scrollTop = 9999;
}
function addMessage(role, html) {
  const el = document.createElement('div');
  el.className = `msg ${role}`;
  el.innerHTML = role === 'user' ? `<div class="msg-label">You</div>${escapeHtml(html)}` : html;
  document.getElementById('messages').appendChild(el);
  document.getElementById('messages').scrollTop = 9999;
  return el;
}
function escapeHtml(s) {
  return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}
</script>
</body>
</html>"""
