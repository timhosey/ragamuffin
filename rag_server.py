import os
from dotenv import load_dotenv
load_dotenv()

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

    embedding = OllamaEmbeddings(model="nomic-embed-text")
    db = Chroma(persist_directory=CHROMA_DIR, embedding_function=embedding)
    retriever = db.as_retriever()
    llm = OllamaLLM(model="llama3")

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

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8001, debug=False)