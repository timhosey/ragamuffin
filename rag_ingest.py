import os
import time
import hashlib
import json
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

load_dotenv()

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")

WATCH_DIR = "./docs"
CHROMA_DIR = "./chroma_store"
HASH_DB = "known_hashes.json"
SCAN_INTERVAL = 15  # seconds

os.makedirs(WATCH_DIR, exist_ok=True)

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
    embedding = OllamaEmbeddings(model=OLLAMA_EMBED_MODEL, base_url=OLLAMA_BASE_URL)
    Chroma.from_documents(chunks, embedding, persist_directory=CHROMA_DIR)

    print("📄 Embedded chunks:")
    for chunk in chunks:
        print("-", chunk.page_content[:150])

def main():
    known_hashes = load_hash_db()
    print("📂 Watching for files in:", WATCH_DIR)

    try:
        while True:
            for root, _, files in os.walk(WATCH_DIR):
                for fname in files:
                    if not (fname.endswith(".pdf") or fname.endswith(".md")):
                        continue
                    path = os.path.join(root, fname)
                    h = file_hash(path)

                    if known_hashes.get(path) != h:
                        print(f"🆕 Ingesting: {path}")
                        try:
                            ingest_file(path)
                            known_hashes[path] = h
                            save_hash_db(known_hashes)
                            print(f"✅ Done ingesting: {path}")
                        except Exception as e:
                            print(f"⚠️ Failed to ingest {path}: {e}")
            time.sleep(SCAN_INTERVAL)
    except KeyboardInterrupt:
        print("\n👋 Ingest watcher stopped. Goodbye!")

if __name__ == "__main__":
    main()