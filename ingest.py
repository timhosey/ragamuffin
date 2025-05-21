import os
import time
import hashlib
import json
import requests
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredMarkdownLoader, UnstructuredEPubLoader
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from datetime import datetime
import pandas as pd

load_dotenv()

# Install pypandoc if it's not installed
try:
    import pypandoc
    pypandoc.get_pandoc_path()
except (ImportError, OSError):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 📦 Installing pypandoc and Pandoc...")
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pypandoc"], text=True)
    import pypandoc
    pypandoc.download_pandoc()

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
    elif filepath.endswith(".xlsx"):
        df = pd.read_excel(filepath)
        text = df.to_string(index=False)
        return [Document(page_content=text, metadata={"source": filepath})]
    elif filepath.endswith(".epub"):
        return UnstructuredEPubLoader(filepath).load()
    else:
        return TextLoader(filepath).load()
    
def load_gsheet_csv(url):
    resp = requests.get(url)
    resp.raise_for_status()
    df = pd.read_csv(BytesIO(resp.content))
    text = df.to_string(index=False)
    return [Document(page_content=text, metadata={"source": url})]

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

def cleanup_orphaned_files(existing_paths):
    embedding = OllamaEmbeddings(model=OLLAMA_EMBED_MODEL, base_url=OLLAMA_BASE_URL)
    db = Chroma(persist_directory=CHROMA_DIR, embedding_function=embedding)
    collection = db.get()
    seen_sources = set(existing_paths)

    to_delete = []
    for metadata in collection["metadatas"]:
        if metadata and "source" in metadata and metadata["source"] not in seen_sources:
            to_delete.append(metadata["source"])

    # Track last cleanup result using a static variable
    if not hasattr(cleanup_orphaned_files, "notified") or cleanup_orphaned_files.notified:
        if not to_delete:
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 🧼 No stale files found to clean up.")
            cleanup_orphaned_files.notified = False
    if to_delete:
        for source in set(to_delete):
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 🧹 Removing stale vectors for: {source}")
            db.delete(where={"source": source})
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ✅ Removed {len(set(to_delete))} stale file(s) from vector store.")
        cleanup_orphaned_files.notified = True

def ingest_file(filepath):
    filepath = os.path.abspath(filepath)
    # Remove old vectors for this file before re-ingesting
    embedding = OllamaEmbeddings(model=OLLAMA_EMBED_MODEL, base_url=OLLAMA_BASE_URL)
    db = Chroma(embedding_function=embedding, persist_directory=CHROMA_DIR)
    db.delete(where={"source": filepath})
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 🤖 Using embedding model: {OLLAMA_EMBED_MODEL}")
    docs = load_document(filepath)
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    # Debug: Identify any chunks with missing content
    for i, c in enumerate(chunks):
        if not c.page_content or not c.page_content.strip():
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ❗️ Skipping empty chunk {i} in {filepath}")
    filtered_chunks = [c for c in chunks if c.page_content and c.page_content.strip()]
    skipped = len(chunks) - len(filtered_chunks)
    if skipped:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ⚠️ Skipped {skipped} empty chunk(s) for {filepath}")
    chunks = filtered_chunks

    for chunk in chunks:
        chunk.metadata["source"] = filepath

    assert all(c.page_content for c in chunks), f"Found chunk with None content in: {filepath}"
    # Add the updated chunks
    db.add_documents(documents=chunks)
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ✅ Added {len(chunks)} chunks to Chroma")

    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 📄 Embedded chunks:")
    for chunk in chunks:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] - {chunk.page_content[:150]}")

def main():
    known_hashes = load_hash_db()
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 📂 Watching for files in: {WATCH_DIR}")

    try:
        while True:
            all_current_paths = []
            for root, _, files in os.walk(WATCH_DIR):
                for fname in files:
                    if fname.endswith(".pdf") or fname.endswith(".md") or fname.endswith(".epub") or fname.endswith(".xlsx"):
                        all_current_paths.append(os.path.join(root, fname))
            cleanup_orphaned_files([os.path.abspath(p) for p in all_current_paths])
            # Prune known_hashes for missing files
            missing = [path for path in known_hashes if path not in all_current_paths]
            if missing:
                for path in missing:
                    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 🗑️ Removing hash for deleted file: {path}")
                    del known_hashes[path]
                save_hash_db(known_hashes)

            for root, _, files in os.walk(WATCH_DIR):
                for fname in files:
                    if not (fname.endswith(".pdf") or fname.endswith(".md") or fname.endswith(".epub") or fname.endswith(".xlsx")):
                        continue
                    path = os.path.join(root, fname)
                    h = file_hash(path)

                    if known_hashes.get(path) != h:
                        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 🆕 Ingesting: {path}")
                        try:
                            ingest_file(path)
                            known_hashes[path] = h
                            save_hash_db(known_hashes)
                            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ✅ Done ingesting: {path}")
                        except Exception as e:
                            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ⚠️ Failed to ingest {path}: {e}")
            
            time.sleep(SCAN_INTERVAL)
    except KeyboardInterrupt:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] \n👋 Ingest watcher stopped. Goodbye!")

if __name__ == "__main__":
    main()