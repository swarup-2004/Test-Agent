from flask import Flask, request, jsonify
from langchain_community.vectorstores import Qdrant
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from qdrant_client import QdrantClient

# Config
COLLECTION_NAME = "knowledge_base"
QDRANT_URL = "http://localhost:6333"

# 1. Load embeddings (same model used in ingestion)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 2. Connect to existing Qdrant collection
client = QdrantClient(url=QDRANT_URL)
vectorstore = Qdrant(
    client=client,
    collection_name=COLLECTION_NAME,
    embeddings=embeddings
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# 3. Load Ollama LLaMA 3.2
llm = OllamaLLM(model="llama3.2")

# 4. Setup Flask app
app = Flask(__name__)

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    if not data or "question" not in data:
        return jsonify({"error": "Missing 'question' field"}), 400

    query = data["question"]

    # Retrieve relevant docs
    docs = retriever.get_relevant_documents(query)
    context = "\n\n".join([doc.page_content for doc in docs])

    # Construct prompt
    prompt = f"""
    Answer the following question based only on the context provided.
    If you don't know, say you don't know.

    Question: {query}

    Context:
    {context}

    Answer:
    """

    answer = llm.invoke(prompt)

    return jsonify({
        "question": query,
        "answer": answer,
        "context_used": [doc.page_content for doc in docs]
    })

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
