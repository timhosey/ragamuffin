import os
import subprocess
import atexit
from dotenv import load_dotenv
import threading
load_dotenv()

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")

from flask import Flask, request, session, redirect, url_for, render_template
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
app.permanent_session_lifetime = timedelta(days=1)

# === Data sources ===
CHROMA_DIR = "./chroma_store"
HASH_DB = "known_hashes.json"

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
    return render_template("chat-ui.html", files=file_list, chat=chat)

@app.route("/ask", methods=["POST"])
def ask_question():
    query = request.form.get("q", "").strip()
    if not query:
        return redirect(url_for("index"))

    embedding = OllamaEmbeddings(model=OLLAMA_EMBED_MODEL, base_url=OLLAMA_BASE_URL)
    db = Chroma(persist_directory=CHROMA_DIR, embedding_function=embedding)
    retriever = db.as_retriever()
    llm = OllamaLLM(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL)

    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
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

ingest_process = None
stop_stream = threading.Event()

def shutdown_ingest():
    if ingest_process:
        print("🛑 Stopping rag_ingest.py...")
        stop_stream.set()
        ingest_process.terminate()
        ingest_process.wait()
        print("✅ rag_ingest.py has stopped.")

def on_exit():
    print("👋 rag_server is now closing.")

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
            print("🛰️ [ingest]", line.strip())

    threading.Thread(target=stream_ingest_output, daemon=True).start()
    atexit.register(shutdown_ingest)
    atexit.register(on_exit)
    app.run(host="0.0.0.0", port=8001, debug=False)