import os
import json
import subprocess
import threading
import atexit
import time
from datetime import datetime, timedelta
from flask import Flask, request, session, redirect, url_for, render_template
from flask_session import Session
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_core.documents import Document
from langchain.prompts import PromptTemplate
import markdown2
from dotenv import load_dotenv
import re

load_dotenv()

# === Config ===
LOG_FILE = "ragamuffin.log"
FAISS_DIR = "./faiss_store"
HASH_DB = "known_hashes.json"

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT", "You are a helpful assistant who answers questions clearly and concisely based only on the provided documents. Avoid using the phrase 'According to the context' wherever possible.")
SECRET = os.getenv("FLASK_SECRET_KEY")

# === Logging ===
def log(msg):
    timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    line = f"{timestamp} {msg}"
    print(line)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")

# === App Setup ===
if not SECRET:
    raise RuntimeError("❌ FLASK_SECRET_KEY not set in .env!")

app = Flask(__name__)
app.config["SECRET_KEY"] = SECRET
app.config["SESSION_TYPE"] = "filesystem"
Session(app)
app.permanent_session_lifetime = timedelta(days=1)

retriever = None
qa_chain = None
ingest_process = None
stop_stream = threading.Event()

def stream_ingest_output():
    while not stop_stream.is_set():
        if ingest_process.stdout is None:
            break
        line = ingest_process.stdout.readline()
        if not line:
            break
        log(f"🛰️ [ingest] {line.strip()}")

def start_ingest():
    global ingest_process
    ingest_process = subprocess.Popen(
        ["python", "-u", "ingest.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )
    threading.Thread(target=stream_ingest_output, daemon=True).start()

def shutdown_ingest():
    if ingest_process:
        log("🛑 Stopping ingest...")
        stop_stream.set()
        ingest_process.terminate()
        ingest_process.wait()
        log("✅ ingest has stopped.")

def load_hash_db():
    if os.path.exists(HASH_DB):
        with open(HASH_DB, "r") as f:
            return json.load(f)
    return {}

def build_qa_chain():
    global qa_chain
    llm = OllamaLLM(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL)
    prompt = PromptTemplate.from_template(f"""{SYSTEM_PROMPT}

Context:
{{context}}

Question:
{{question}}

Answer:""")
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt}
    )

def refresh_vectorstore():
    global retriever
    embedding = OllamaEmbeddings(model=OLLAMA_EMBED_MODEL, base_url=OLLAMA_BASE_URL)
    index_file_path = os.path.join(FAISS_DIR, "index.faiss")
    if not os.path.exists(index_file_path):
        log(f"⚠️ FAISS index file not found at {index_file_path}. Skipping reload.")
        return
    db = FAISS.load_local(FAISS_DIR, embedding, allow_dangerous_deserialization=True)
    retriever = db.as_retriever()
    try:
        all_docs = db.docstore._dict.values()
        log(f"🧠 Vectorstore rehydrated with {len(list(all_docs))} total chunks")
    except Exception as e:
        log(f"⚠️ Failed to inspect vectorstore after refresh: {e}")
    # Rebuild the QA chain after refreshing the vectorstore
    build_qa_chain()

@app.route("/")
def index():
    import glob
    file_list = [os.path.basename(f) for f in glob.glob("docs/*.md")]
    chat = session.get("chat_history", [])
    refreshed = session.pop("refresh_success", None)
    return render_template("chat-ui.html", files=file_list, chat=chat, refreshed=refreshed)

@app.route("/ask", methods=["POST"])
def ask_question():
    query = request.form.get("q", "").strip()
    if not query:
        return redirect(url_for("index"))

    try:
        results = retriever.vectorstore.similarity_search_with_score(query)
        docs = []
        for result in results:
            if isinstance(result, tuple) and isinstance(result[0], Document):
                doc = result[0]
                if isinstance(doc.page_content, str) and doc.page_content.strip():
                    docs.append(doc)
                else:
                    log(f"⚠️ Skipping invalid document: {doc.metadata}")
            else:
                log(f"❌ Invalid result skipped during similarity search: {result}")
        if not docs:
            log("⚠️ No relevant documents retrieved.")

        log(f"🔍 Retrieved {len(docs)} docs for: '{query}'")
        for doc in docs:
            log(f"• {doc.metadata.get('source', 'unknown')} — {doc.page_content[:100]}")

        response_data = qa_chain.combine_documents_chain.invoke({"input_documents": docs, "question": query})
        response_text = response_data.get("output_text", str(response_data))
        answer = markdown2.markdown(
            response_text,
            extras=["link-patterns", "target-blank-links"],
            link_patterns=[(re.compile(r'(https?://[^\s]+)', re.IGNORECASE), r'\1')]
        )
    except Exception as e:
        log(f"❌ Failed to answer question: {e}")
        answer = markdown2.markdown("⚠️ Sorry, there was an error answering your question.")

    chat = session.get("chat_history", [])
    chat.append({ "q": query, "a": answer })
    session["chat_history"] = chat

    return redirect(url_for("index"))

@app.route("/refresh", methods=["POST"])
def refresh_retriever():
    global qa_chain
    try:
        log("🔁 Manual retriever refresh triggered via web UI")
        if ingest_process:
            ingest_process.terminate()
            ingest_process.wait()
            log("♻️ Restarting ingest process...")
        start_ingest()
        refresh_vectorstore()
        build_qa_chain()
        session["refresh_success"] = True
    except Exception as e:
        session["refresh_success"] = False
        log(f"⚠️ Web-triggered retriever refresh failed: {e}")
    return redirect(url_for("index"))

atexit.register(shutdown_ingest)

if __name__ == "__main__":
    start_ingest()
    refresh_vectorstore()
    if retriever is not None:
        build_qa_chain()
    # Periodic refresh every 5 minutes
    threading.Thread(target=lambda: (lambda interval: [time.sleep(interval) or refresh_vectorstore() or build_qa_chain() for _ in iter(int, 1)])(300), daemon=True).start()
    app.run(host="0.0.0.0", port=8001, debug=False)
