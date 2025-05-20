# RAGamuffin

RAGamuffin is a simple Python-based RAG (Retrieval-Augmented Generation) server.

## Installation

Create a .env file, with the following entries:
```
FLASK_SECRET_KEY=secret-goes-here
OLLAMA_MODEL=granite3.3:8b
OLLAMA_EMBED_MODEL=nomic-embed-text
OLLAMA_BASE_URL=http://ollama-server-url:11434
```

Run `pip -r requirements.txt` to install required dependencies.

For this version, ensure you have a .doc folder in the same directory as the scripts. This is where you'll load your files. (TODO: Add customizeable paths)

Run `python rag_server.py`, which will kick off the server and ingest. `chroma_store` and `flask_session` directories will be created as well. You should see docs ingesting.

Once some files are ingested, visit `http://localhost:8001` to access the webui.

## Supported files
So far, RAGamuffin supports:

* TXT
* PDF
* XLSX
* Google Sheets (via URL, untested)
* EPUB

I'm interested in adding more in the near future. Would also be cool as hell to support vision model descriptions for images, so you feed images, get their descriptions from a vision model, and then feed that into the RAG.