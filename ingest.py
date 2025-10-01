from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Qdrant
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings

# Config
PDF_PATH = "docs\SBI_car_policy-1.pdf"
COLLECTION_NAME = "knowledge_base"
QDRANT_URL = "http://localhost:6333"

# 1. Load PDF
print("[INFO] Loading PDF...")
loader = PyPDFLoader(PDF_PATH)
documents = loader.load()

# 2. Split into chunks
print("[INFO] Splitting into chunks...")
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
docs = splitter.split_documents(documents)

# 3. Generate embeddings
print("[INFO] Generating embeddings...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 4. Store in Qdrant
print("[INFO] Storing documents in Qdrant...")
qdrant = Qdrant.from_documents(
    docs,
    embeddings,
    url=QDRANT_URL,
    prefer_grpc=False,
    collection_name=COLLECTION_NAME
)

print("[INFO] Ingestion completed. Knowledge base stored in Qdrant!")
