import os
import time
import hashlib
from datetime import datetime
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")

WATCH_DIR = "docs"
FAISS_DIR = "faiss_store"
SCAN_INTERVAL = 15  # seconds

os.makedirs(WATCH_DIR, exist_ok=True)

def load_markdown_file(filepath):
    try:
        loader = TextLoader(filepath)
        docs = loader.load()
        print(f"[{datetime.now()}] ✅ Loaded {len(docs)} documents/pages from: {filepath}")
        return docs
    except Exception as e:
        print(f"[{datetime.now()}] ❌ Failed to load {filepath}: {e}")
        return []

def file_hash(filepath):
    with open(filepath, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()

def ingest_file(filepath, known_hashes):
    if not (filepath.lower().endswith(".md") or filepath.lower().endswith(".pdf")):
        print(f"[{datetime.now()}] ⏭️ Skipping non-markdown/pdf file: {filepath}")
        return

    new_hash = file_hash(filepath)
    if known_hashes.get(filepath) == new_hash:
        return

    print(f"[{datetime.now()}] 🔍 Ingesting updated file: {filepath}")

    if filepath.lower().endswith(".md"):
        docs = load_markdown_file(filepath)
    elif filepath.lower().endswith(".pdf"):
        try:
            loader = PyPDFLoader(filepath)
            docs = loader.load()
            print(f"[{datetime.now()}] ✅ Loaded {len(docs)} documents from: {filepath}")
        except Exception as e:
            print(f"[{datetime.now()}] ❌ Failed to load {filepath}: {e}")
            return
    else:
        return

    if not docs:
        print(f"[{datetime.now()}] ⚠️ No documents loaded from: {filepath}")
        return

    for doc in docs:
        doc.metadata["source"] = filepath

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    chunks = [chunk for chunk in chunks if chunk.page_content.strip()]

    if not chunks:
        print(f"[{datetime.now()}] ⚠️ No valid chunks extracted from: {filepath}")
        return

    embedding = OllamaEmbeddings(model=OLLAMA_EMBED_MODEL, base_url=OLLAMA_BASE_URL)

    if os.path.exists(os.path.join(FAISS_DIR, "index.faiss")):
        try:
            db = FAISS.load_local(FAISS_DIR, embedding, allow_dangerous_deserialization=True)
            print(f"[{datetime.now()}] 🧠 FAISS loaded from disk.")
            # Remove all existing documents from the same source
            ids_to_delete = [doc_id for doc_id, doc in db.docstore._dict.items()
                             if doc.metadata.get("source") == filepath]
            if ids_to_delete:
                db.delete(ids_to_delete)
                print(f"[{datetime.now()}] 🧹 Removed {len(ids_to_delete)} old chunks for: {filepath}")
            else:
                print(f"[{datetime.now()}] ℹ️ Ingesting as no existing vectors found for: {filepath}...")
        except Exception as e:
            print(f"[{datetime.now()}] ❌ Failed to load FAISS: {e}")
            db = None
    else:
        db = None

    if db is None:
        if chunks:
            db = FAISS.from_documents(chunks, embedding)
        else:
            print(f"[{datetime.now()}] ⚠️ No chunks available to initialize FAISS.")
            return
    else:
        db.add_documents(chunks)

    db.save_local(FAISS_DIR)
    known_hashes[filepath] = new_hash
    print(f"[{datetime.now()}] ✅ Ingested {len(chunks)} chunks from: {filepath}")

def main():
    known_hashes = {}

    try:
        # Preload known_hashes from existing FAISS vectorstore, if present
        if os.path.exists(os.path.join(FAISS_DIR, "index.faiss")):
            embedding = OllamaEmbeddings(model=OLLAMA_EMBED_MODEL, base_url=OLLAMA_BASE_URL)
            try:
                db = FAISS.load_local(FAISS_DIR, embedding, allow_dangerous_deserialization=True)
                print(f"[{datetime.now()}] 🔄 FAISS vectorstore successfully loaded during startup.")
                seen = set()
                for doc in db.docstore._dict.values():
                    source = doc.metadata.get("source")
                    if source in seen:
                        continue
                    seen.add(source)
                    if source and os.path.exists(source):
                        known_hashes[source] = file_hash(source)

                # Detect and clean up removed files
                current_sources = set(doc.metadata.get("source") for doc in db.docstore._dict.values())
                removed_sources = [src for src in current_sources if not os.path.exists(src)]
                for removed in removed_sources:
                    ids_to_delete = [doc_id for doc_id, doc in db.docstore._dict.items()
                                     if doc.metadata.get("source") == removed]
                    if ids_to_delete:
                        db.delete(ids_to_delete)
                        print(f"[{datetime.now()}] 🧹 Removed {len(ids_to_delete)} chunks for deleted file on startup: {removed}")
                        db.save_local(FAISS_DIR)
                    if removed in known_hashes:
                        del known_hashes[removed]
            except Exception as e:
                print(f"[{datetime.now()}] ❌ Failed to preload FAISS vectorstore: {e}")
        
        print(f"[{datetime.now()}] ✅ Finished preloading FAISS vectorstore. Beginning scan loop.")
        print(f"[{datetime.now()}] 📂 Watching for markdown/pdf files in: {WATCH_DIR}")

        while True:
            all_md_files = []
            for root, _, files in os.walk(WATCH_DIR):
                for fname in files:
                    if fname.lower().endswith(".md") or fname.lower().endswith(".pdf"):
                        full_path = os.path.join(root, fname)
                        all_md_files.append(full_path)

            for fpath in all_md_files:
                ingest_file(fpath, known_hashes)

            # Remove files that were previously ingested but are now deleted
            removed_files = set(known_hashes.keys()) - set(all_md_files)
            if removed_files:
                print(f"[{datetime.now()}] 🗑️ Found removed files: {removed_files}")
                if os.path.exists(os.path.join(FAISS_DIR, "index.faiss")):
                    embedding = OllamaEmbeddings(model=OLLAMA_EMBED_MODEL, base_url=OLLAMA_BASE_URL)
                    try:
                        db = FAISS.load_local(FAISS_DIR, embedding, allow_dangerous_deserialization=True)
                        for removed in removed_files:
                            ids_to_delete = [doc_id for doc_id, doc in db.docstore._dict.items()
                                             if doc.metadata.get("source") == removed]
                            if ids_to_delete:
                                db.delete(ids_to_delete)
                                print(f"[{datetime.now()}] 🧹 Removed {len(ids_to_delete)} chunks for deleted file: {removed}")
                                db.save_local(FAISS_DIR)
                            del known_hashes[removed]
                    except Exception as e:
                        print(f"[{datetime.now()}] ❌ Failed to load FAISS for cleanup: {e}")

            time.sleep(SCAN_INTERVAL)
    except KeyboardInterrupt:
        print(f"[{datetime.now()}] 👋 Ingest watcher stopped gracefully.")

if __name__ == "__main__":
    main()