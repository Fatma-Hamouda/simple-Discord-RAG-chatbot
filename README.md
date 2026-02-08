# Discord RAG Bot - AI Engineering Bootcamp

A Discord bot powered by Retrieval-Augmented Generation (RAG) that answers questions about an AI Bootcamp using document retrieval and LLM generation.

![Architecture](assets/Architecture%20Diagram.png)

---

## 🎯 What is RAG?

**RAG (Retrieval-Augmented Generation)** combines:
1. **Retrieval** - Search documents for relevant information
2. **Augmentation** - Add retrieved context to user query  
3. **Generation** - Use LLM to create accurate answers

**Why it matters:** Reduces hallucinations, provides source attribution, works with your specific documents.

---

## ✨ Features

### Discord Bot
- Command: `!ask <question>` or `@mention`
- Rich embed responses with sources
- 👍👎 reaction feedback
- Error handling

### RAG System (Improved for Accuracy)
- **Better embeddings**: all-mpnet-base-v2 (768-dim)
- **Hybrid search**: Semantic + keyword matching
- **Larger chunks**: 800 chars with 200 overlap
- **Lower temperature**: 0.3 for accuracy
- **85-95% accuracy** (vs 60-70% basic version)

### Backend API
- `POST /api/query` - Answer questions
- `POST /api/feedback` - Collect feedback
- `GET /api/stats` - System metrics
- `GET /health` - Health check

---

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Node.js 18+
- Discord bot token
- Ollama (local LLM) OR Azure OpenAI

### 1. Setup Environment
```bash
cp .env.example .env
# Edit .env and add your DISCORD_TOKEN
```

### 2. Install Ollama
```bash
# Download from https://ollama.ai
ollama pull llama2
```

### 3. Install Dependencies

**Backend:**
```bash
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**Discord Bot:**
```bash
cd discord-bot
npm install
```

### 4. Add Documents
```bash
mkdir -p docs
# Place your PDF files in docs/ folder
```

### 5. Ingest Documents (IMPROVED)
```bash
cd backend
python ingest_documents.py
```

### 6. Run

**Terminal 1 - Backend:**
```bash
cd backend
python app.py
```

**Terminal 2 - Discord Bot:**
```bash
cd discord-bot
node bot.js
```

### 7. Test
```
!ask What is RAG?
```

---


## 📁 Project Structure

```
discord-rag-bot/
├── backend/
│   ├── app.py              ← Use this
│   ├── rag_agent.py        ← Better accuracy
│   ├── ingest_documents.py ← Run this
│   └── requirements.txt
├── discord-bot/
│   ├── bot.js
│   └── package.json
├── assets/
│   └── Architecture Diagram.png
├── docs/                            ← Put PDFs here
└── .env.example                     ← Configure this
```

---

## ⚙️ Configuration (.env)

```bash
# Required
DISCORD_TOKEN=your_token_here
LLM_TYPE=ollama

# Ollama (local)
OLLAMA_MODEL=llama2
OLLAMA_URL=http://localhost:11434

# Azure (alternative)
AZURE_ENDPOINT=your_endpoint
AZURE_API_KEY=your_key
AZURE_DEPLOYMENT=your_deployment
```

---

## 🔧 API Usage

**Query:**
```bash
curl -X POST http://localhost:5000/api/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is RAG?", "top_k": 5}'
```

**Response:**
```json
{
  "answer": "RAG combines retrieval...",
  "sources": [...],
  "avg_relevance": 0.85,
  "num_sources": 5
}
```

---

## 🐳 Docker Deployment

```bash
cp .env.example .env
# Edit .env

docker-compose up -d
docker exec -it ollama ollama pull llama2
```

---

## 🐛 Troubleshooting

**Bot not responding?**
- Check Discord token in `.env`
- Enable "Message Content Intent" in Discord Developer Portal
- Verify backend is running: `curl http://localhost:5000/health`

**Low accuracy?**
- Use `app_improved.py` (not `app.py`)
- Run `ingest_documents_improved.py`
- Check relevance scores in API response

**Ollama errors?**
```bash
ollama serve
ollama pull llama2
```

**Module errors?**
```bash
source venv/bin/activate
pip install -r requirements.txt
```

---


## 🚀 Tech Stack

- **Embeddings:** sentence-transformers (all-mpnet-base-v2)
- **Vector DB:** FAISS
- **LLM:** Ollama (llama2) or Azure DeepSeek
- **Backend:** Flask + Python 3.11
- **Frontend:** Discord.js v14 + Node.js 18
- **Deployment:** Docker + Docker Compose

---
