import os
import subprocess
import atexit
from dotenv import load_dotenv
import threading
from datetime import datetime
import time
load_dotenv()

SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT", "You are a helpful assistant who answers questions clearly and concisely based only on the provided documents. Avoid using the phrase 'According to the context' wherever possible.")

LOG_FILE = "rag_server.log"

def log(msg):
    timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    line = f"{timestamp} {msg}"
    print(line)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")

from flask import Flask, request, session, redirect, url_for, render_template
from flask_session import Session
from datetime import timedelta
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
import json
import markdown2

# === App setup ===
SECRET = os.getenv("FLASK_SECRET_KEY")
if not SECRET:
    raise RuntimeError("❌ FLASK_SECRET_KEY not set in .env!")

app = Flask(__name__)
app.config["SECRET_KEY"] = SECRET
app.config["SESSION_TYPE"] = "filesystem"
Session(app)
app.permanent_session_lifetime = timedelta(days=1)

# === Data sources ===
CHROMA_DIR = "./chroma_store"
HASH_DB = "known_hashes.json"

retriever = None

def load_hash_db():
    if os.path.exists(HASH_DB):
        with open(HASH_DB, "r") as f:
            return json.load(f)
    return {}

@app.route("/")
def index():
    known_hashes = load_hash_db()
    file_list = list(known_hashes.keys())
    chat = session.get("chat_history", [])
    refreshed = session.pop("refresh_success", None)
    return render_template("chat-ui.html", files=file_list, chat=chat, refreshed=refreshed)

@app.route("/ask", methods=["POST"])
def ask_question():
    query = request.form.get("q", "").strip()
    if not query:
        return redirect(url_for("index"))

    embedding = OllamaEmbeddings(model=OLLAMA_EMBED_MODEL, base_url=OLLAMA_BASE_URL)
    db = Chroma(persist_directory=CHROMA_DIR, embedding_function=embedding)
    global retriever
    if retriever is None:
        retriever = db.as_retriever()
    llm = OllamaLLM(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL)

    from langchain.prompts import PromptTemplate

    prompt_template = PromptTemplate.from_template(f"""{SYSTEM_PROMPT}

Context:
{{context}}

Question:
{{question}}

Answer:""")

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt_template}
    )
    response = qa_chain.invoke(query)

    # If the response is a dict (some chains return structured output), extract the answer string
    if isinstance(response, dict):
        raw_answer = response.get("result") or response.get("output") or str(response)
    else:
        raw_answer = response

    answer = markdown2.markdown(raw_answer)

    chat = session.get("chat_history", [])
    chat.append({ "q": query, "a": answer })
    session["chat_history"] = chat

    return redirect(url_for("index"))

@app.route("/refresh", methods=["POST"])
def refresh_retriever():
    try:
        log("🔁 Manual retriever refresh triggered via web UI")
        embedding = OllamaEmbeddings(model=OLLAMA_EMBED_MODEL, base_url=OLLAMA_BASE_URL)
        db = Chroma(persist_directory=CHROMA_DIR, embedding_function=embedding)
        global retriever
        retriever = db.as_retriever()
        session["refresh_success"] = True
    except Exception as e:
        session["refresh_success"] = False
        log(f"⚠️ Web-triggered retriever refresh failed: {e}")
    return redirect(url_for("index"))

ingest_process = None
stop_stream = threading.Event()

def shutdown_ingest():
    if ingest_process:
        log("🛑 Stopping rag_ingest.py...")
        stop_stream.set()
        ingest_process.terminate()
        ingest_process.wait()
        log("✅ rag_ingest.py has stopped.")

def on_exit():
    log("👋 rag_server is now closing.")

def refresh_retriever_periodically(interval=60):
    global retriever
    while not stop_stream.is_set():
        time.sleep(interval)
        try:
            log("♻️ Refreshing vector store retriever...")
            embedding = OllamaEmbeddings(model=OLLAMA_EMBED_MODEL, base_url=OLLAMA_BASE_URL)
            db = Chroma(persist_directory=CHROMA_DIR, embedding_function=embedding)
            retriever = db.as_retriever()
        except Exception as e:
            log(f"⚠️ Error refreshing retriever: {e}")

if __name__ == "__main__":
    ingest_process = subprocess.Popen(
        ["python", "-u", "rag_ingest.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )

    def stream_ingest_output():
        while not stop_stream.is_set():
            if ingest_process.stdout is None:
                break
            line = ingest_process.stdout.readline()
            if not line:
                break
            log(f"🛰️ [ingest] {line.strip()}")

    threading.Thread(target=stream_ingest_output, daemon=True).start()
    atexit.register(shutdown_ingest)
    atexit.register(on_exit)
    threading.Thread(target=refresh_retriever_periodically, daemon=True).start()
    app.run(host="0.0.0.0", port=8001, debug=False)
