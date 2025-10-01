## 🚀 Setup Instructions

### 1. Clone this repository

```bash
git clone https://github.com/your-username/test-agent.git
cd test-agent
```

### 2. Start Qdrant with Docker

Run this inside your project folder to start Qdrant with persistent storage:

**Windows CMD**

```cmd
docker run -p 6333:6333 -v "%cd%/qdrant_data:/qdrant/storage" qdrant/qdrant
```

**PowerShell**

```powershell
docker run -p 6333:6333 -v ${PWD}/qdrant_data:/qdrant/storage qdrant/qdrant
```

**Linux/Mac**

```bash
docker run -p 6333:6333 -v $(pwd)/qdrant_data:/qdrant/storage qdrant/qdrant
```

Check that Qdrant is running:
👉 [http://localhost:6333/dashboard](http://localhost:6333/dashboard)

---

### 3. Install Dependencies

Make sure you have Python 3.9+ and create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows

pip install -r requirements.txt
```

---

### 4. Generate Embeddings (Knowledge Base Setup)

Run this script once to process your PDF and store embeddings in Qdrant:

```bash
python ingest.py
```

This script will:

* Read the PDF from `docs/your_file.pdf`
* Generate embeddings
* Store them in the `qdrant_data` collection

---

### 5. Start Flask Server

Run the agent backend:

```bash
fastapi dev app.py --host 127.0.0.1 --port 8001

```

The server will start at:
👉 `http://127.0.0.1:5000`

---

## 🔎 Example Request (cURL)

Send a query to the RAG Agent using `curl`:

```bash
curl -X POST http://127.0.0.1:5000/ask \
     -H "Content-Type: application/json" \
     -d '{"question": "What is the main topic of the PDF?"}'
```

### ✅ Example Response:

```json
{
  "answer": "The PDF mainly discusses X, Y, and Z...",
  "context": [
    "Paragraph 1 from the PDF...",
    "Relevant section from the PDF..."
  ]
}
```

---

## 📂 Project Structure

```
Test-Agent/
│── app.py                  # FastAPI server
│── generate_embeddings.py  # Script to process PDF and store embeddings
│── requirements.txt        # Python dependencies
│── data/                   # Folder containing the PDF knowledge base
│── qdrant_data/            # Persistent storage for Qdrant
│── README.md               # Project documentation
```

---

⚡ You now have a working **local RAG Agent** with FastAPI + Qdrant + Ollama!

---
