from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from langchain_community.vectorstores import Qdrant
from langchain.embeddings import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

# -------------------------------
# Config
# -------------------------------
COLLECTION_NAME = "knowledge_base"
QDRANT_URL = "http://localhost:6333"

# -------------------------------
# 1. Load embeddings (same model used in ingestion)
# -------------------------------
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# -------------------------------
# 2. Connect to existing Qdrant collection
# -------------------------------
client = QdrantClient(url=QDRANT_URL)
vectorstore = Qdrant(
    client=client,
    collection_name=COLLECTION_NAME,
    embeddings=embeddings
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# -------------------------------
# 3. Load Ollama LLaMA 3.2
# -------------------------------
# Load environment variables
load_dotenv()

# Access the Groq API key
groq_api_key = os.getenv("GROQ_API_KEY")

# Initialize the LLM (same model name as Ollama)
llm = ChatGroq(
    api_key=groq_api_key,
    model_name="llama-3.1-8b-instant"  # or "llama3.2" if supported
)

# -------------------------------
# 4. FastAPI app setup
# -------------------------------
app = FastAPI(title="RAG Agent API")

# -------------------------------
# 5. Request & Response Models
# -------------------------------
class AskRequest(BaseModel):
    question: str

class AskResponse(BaseModel):
    question: str
    answer: str
    context_used: List[str]

# -------------------------------
# 6. Endpoint
# -------------------------------
@app.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest):
    query = request.question
    if not query.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    # Retrieve relevant documents
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

    # Call Ollama LLaMA
    answer = llm.invoke(prompt)

    print(f"[INFO] Question: {query}")
    print(f"[INFO] Answer: {answer}")

    return AskResponse(
        question=query,
        answer=answer.content if hasattr(answer, "content") else str(answer),
        context_used=[doc.page_content for doc in docs]
    )

# -------------------------------
# 7. Run the server
# -------------------------------
# Run with: uvicorn app:app --host 127.0.0.1 --port 8000 --reload
