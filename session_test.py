import os
import time
import hashlib
import threading
import json
from datetime import timedelta
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA

# === Load environment and config ===
load_dotenv()
SECRET = os.getenv("FLASK_SECRET_KEY")
if not SECRET:
    raise RuntimeError("❌ FLASK_SECRET_KEY not set!")

print("✅ Loaded Flask secret key:", SECRET)

from flask import Flask, session

app = Flask(__name__)
app.config["SECRET_KEY"] = SECRET
app.secret_key = SECRET

UPLOAD_DIR = "./docs"
CHROMA_DIR = "./chroma_store"
HASH_DB = "known_hashes.json"
SCAN_INTERVAL = 15  # seconds

# === Flask setup ===
app = Flask(__name__)
app.config["SECRET_KEY"] = SECRET
app.secret_key = SECRET
app.permanent_session_lifetime = timedelta(days=1)

# === Ensure watch directory exists ===
os.makedirs(UPLOAD_DIR, exist_ok=True)

# === Utility Functions ===
def load_document(filepath):
    if filepath.endswith(".pdf"):
        return PyPDFLoader(filepath).load()
    elif filepath.endswith(".md"):
        return UnstructuredMarkdownLoader(filepath).load()
    else:
        return TextLoader(filepath).load()

def file_hash(filepath):
    with open(filepath, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()

def load_hash_db():
    if os.path.exists(HASH_DB):
        with open(HASH_DB, "r") as f:
            return json.load(f)
    return {}

def save_hash_db(hashes):
    with open(HASH_DB, "w") as f:
        json.dump(hashes, f, indent=2)

def ingest_file(filepath):
    docs = load_document(filepath)
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    embedding = OllamaEmbeddings(model="nomic-embed-text")
    db = Chroma.from_documents(chunks, embedding, persist_directory=CHROMA_DIR)

def scan_and_ingest():
    known_hashes = load_hash_db()
    print("📂 Starting folder watch on:", UPLOAD_DIR)

    while True:
        for fname in os.listdir(UPLOAD_DIR):
            if not (fname.endswith(".pdf") or fname.endswith(".md")):
                continue
            path = os.path.join(UPLOAD_DIR, fname)
            h = file_hash(path)

            if known_hashes.get(fname) != h:
                print(f"🆕 Ingesting: {fname}")
                try:
                    ingest_file(path)
                    known_hashes[fname] = h
                    save_hash_db(known_hashes)
                    print(f"✅ Done ingesting: {fname}")
                except Exception as e:
                    print(f"⚠️ Failed to ingest {fname}: {e}")
            else:
                print(f"⏩ Skipping unchanged: {fname}")

        time.sleep(SCAN_INTERVAL)

# === Flask Routes ===
@app.before_request
def check_secret():
    print("✅ Flask sees secret key:", bool(app.secret_key))

@app.route("/", methods=["GET"])
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

    embedding = OllamaEmbeddings(model="nomic-embed-text")
    db = Chroma(persist_directory=CHROMA_DIR, embedding_function=embedding)
    retriever = db.as_retriever()
    llm = OllamaLLM(model="llama3")

    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    answer = qa_chain.invoke(query)

    chat = session.get("chat_history", [])
    chat.append({"q": query, "a": answer})
    session["chat_history"] = chat

    return redirect(url_for("index"))

def run_file_watcher():
    import threading
    threading.Thread(target=scan_and_ingest, daemon=True).start()

if __name__ == "__main__":
    run_file_watcher()
    app.run(host="0.0.0.0", port=8001, debug=False, use_reloader=False)