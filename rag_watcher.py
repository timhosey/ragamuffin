import os, time, hashlib, threading, json
from dotenv import load_dotenv
load_dotenv()

from flask import Flask, request, jsonify, redirect, url_for, render_template, render_template_string, session
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from datetime import timedelta

# Load secret before initializing Flask
SECRET = os.getenv("FLASK_SECRET_KEY")
if not SECRET:
    raise RuntimeError("❌ FLASK_SECRET_KEY is not set!")

print("✅ Loaded Flask secret key:", SECRET)

app = Flask(__name__)
app.config["SECRET_KEY"] = SECRET  # 👈 this is the one Flask uses internally
app.secret_key = SECRET  # 👈 for compatibility with old Flask extensions
app.permanent_session_lifetime = timedelta(days=1)

HASH_DB = "known_hashes.json"

def load_hash_db():
    if os.path.exists(HASH_DB):
        with open(HASH_DB, "r") as f:
            return json.load(f)
    return {}

def save_hash_db(hashes):
    with open(HASH_DB, "w") as f:
        json.dump(hashes, f, indent=2)

# Config
WATCH_DIR = "./docs"
CHROMA_DIR = "./chroma_store"
SCAN_INTERVAL = 15  # seconds

os.makedirs(WATCH_DIR, exist_ok=True)

app = Flask(__name__)
known_hashes = {}

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

def ingest_file(filepath):
    docs = load_document(filepath)
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    embedding = OllamaEmbeddings(model="nomic-embed-text")
    db = Chroma.from_documents(chunks, embedding, persist_directory=CHROMA_DIR)

def scan_and_ingest():
    global known_hashes
    known_hashes = load_hash_db()
    print("📂 Starting folder watch on:", WATCH_DIR)

    while True:
        for fname in os.listdir(WATCH_DIR):
            if not (fname.endswith(".pdf") or fname.endswith(".md")):
                continue
            path = os.path.join(WATCH_DIR, fname)
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

@app.before_request
def check_session():
    print("✅ Flask sees the secret key:", bool(app.secret_key))

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

    # Save to chat history
    chat = session.get("chat_history", [])
    chat.append({"q": query, "a": answer})
    session["chat_history"] = chat

    return redirect(url_for("index"))

if __name__ == "__main__":
    print("🧠 RAG server starting...")
    threading.Thread(target=scan_and_ingest, daemon=True).start()
    app.run(host="0.0.0.0", port=8001, debug=False, use_reloader=False)