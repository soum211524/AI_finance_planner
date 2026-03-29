# 🔥 ET Money Mentor — FIRE Path Planner



A full-stack AI-powered FIRE (Financial Independence, Retire Early) planning tool for Indian investors. Built with a FastAPI + LangChain RAG backend and a single-file vanilla HTML/CSS/JS frontend.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    FRONTEND (index.html)                 │
│  Vanilla HTML/CSS/JS · No framework · Single file        │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────┐  │
│  │ Input Panel │  │  Results     │  │  AI Chat       │  │
│  │ (profile)   │  │  Dashboard   │  │  (RAG Q&A)     │  │
│  └──────┬──────┘  └──────┬───────┘  └───────┬────────┘  │
└─────────┼────────────────┼──────────────────┼───────────┘
          │  /calculate    │  /full-plan       │  /ask
┌─────────▼────────────────▼──────────────────▼───────────┐
│                  FASTAPI BACKEND (main.py)               │
│  ┌──────────────────┐    ┌───────────────────────────┐   │
│  │ Calculation      │    │ LangChain RAG Chain        │   │
│  │ Engine (pure     │    │  ┌─────────────────────┐  │   │
│  │ math, instant)   │    │  │ HuggingFace Embed.  │  │   │
│  │                  │    │  │ (all-MiniLM-L6-v2)  │  │   │
│  │ • FIRE corpus    │    │  └──────────┬──────────┘  │   │
│  │ • Goal SIPs      │    │             │              │   │
│  │ • Asset alloc    │    │  ┌──────────▼──────────┐  │   │
│  │ • Tax savings    │    │  │ FAISS Vector Store  │  │   │
│  │ • Insurance gaps │    │  │ (fire_knowledge.txt)│  │   │
│  │ • Health score   │    │  └──────────┬──────────┘  │   │
│  │ • Milestones     │    │             │              │   │
│  └──────────────────┘    │  ┌──────────▼──────────┐  │   │
│                          │  │ Groq LLM            │  │   │
│                          │  │ llama-3.3-70b-versatile  │   │
│                          │  │ (fast inference,    │  │   │
│                          │  │  free tier)         │  │   │
│                          │  └─────────────────────┘  │   │
│                          └───────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
          │
┌─────────▼──────────┐
│  Knowledge Base     │
│  fire_knowledge.txt │
│  (FIRE fundamentals,│
│  Indian tax laws,   │
│  MF instruments,    │
│  insurance rules,   │
│  FIRE strategies)   │
└────────────────────┘
```

---

## Features

| Feature | Description |
|---|---|
| **SIP Calculator** | Goal-wise SIP breakdown: FIRE corpus, home, education, emergency fund |
| **Asset Allocation** | Age-appropriate equity/debt/gold split with glide path advice |
| **Insurance Gap Finder** | Detects underinsurance in term life, health, and critical illness |
| **Tax Savings** | Section 80C, 80CCD(1B) NPS, 80D health — with exact rupee savings |
| **FIRE Timeline** | Milestone roadmap from today to FIRE age |
| **Emergency Fund Tracker** | 6-month target with build-up SIP plan |
| **Money Health Score** | 6-dimension wellness score (0–100) |
| **AI Chat Mentor** | RAG-powered Q&A grounded in your profile |
| **AI Plan Narrative** | Personalized FIRE action plan generated on demand |

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Health check + RAG status |
| `POST` | `/calculate` | Pure math calculation (instant, no LLM) |
| `POST` | `/ask` | RAG Q&A with optional user profile context |
| `POST` | `/full-plan` | Calculations + AI narrative in one call |

FastAPI auto-generates interactive docs at **`/docs`** once the server is running.

---

## Quick Start

### 1. Backend Setup

```bash
cd backend

# Install dependencies
pip install -r requirements.txt
```

Open `main.py` and set your Groq API key directly at the top of the file:

```python
# main.py
GROQ_API_KEY = "gsk_your_key_here"
```

Get a free key at [console.groq.com](https://console.groq.com).

```bash
# Start the server
uvicorn main:app --reload --port 8000
```

On startup the server will:
1. Load `fire_knowledge.txt` from the `data/` directory
2. Chunk it into 800-token segments with 150-token overlap
3. Embed chunks using `sentence-transformers/all-MiniLM-L6-v2` (runs locally, no API cost)
4. Build a FAISS vector index in memory
5. Wire retrieval to the Groq LLM via LangChain `RetrievalQA`

### 2. Frontend

No build step needed — just open the file:

```bash
open frontend/index.html

# Or serve it locally
cd frontend && python3 -m http.server 3000
```

The frontend auto-detects whether the backend is running and shows the connection status in the header. It works in **local-only mode** (pure JS calculations) even without the backend.

---

## Project Structure

```
fire_planner/
├── backend/
│   ├── main.py              # FastAPI + LangChain RAG app (API key set here)
│   └── requirements.txt     # Python dependencies
├── data/
│   └── fire_knowledge.txt   # RAG knowledge base (~5000 tokens)
├── frontend/
│   └── index.html           # Single-file vanilla frontend
└── README.md
```

---

## RAG Design

**Chunking** — `RecursiveCharacterTextSplitter` with 800-char chunks and 150-char overlap. Splits on `\n\n`, `\n`, `.` to preserve semantic units like "Section 80C" rules.

**Embeddings** — `sentence-transformers/all-MiniLM-L6-v2`. Lightweight, runs locally (no API cost), 384-dimensional vectors, well-suited to financial English text.

**Retrieval** — FAISS cosine similarity, top-5 chunks returned per query. The user's profile is prepended to every question so retrieval is grounded in their specific numbers.

**LLM** — `llama-3.3-70b-versatile` via `langchain-groq`. Temperature 0.3 for factual, conservative financial advice. A custom system prompt enforces Indian context, specific instruments, and rupee amounts.

**Prompt strategy** — Profile injected as a prefix before every question:
```
Age 28, FIRE at 45, income ₹1.2L/mo... Question: How do I increase my savings rate?
```
This grounds both retrieval and generation in the user's actual situation.

---

## Impact Model

| Metric | Status quo | With ET FIRE Mentor | Impact |
|---|---|---|---|
| Financial advisor access | ₹25,000+/year, HNI only | Free, instant | **Democratizes access** |
| Time to generate plan | 2–3 advisor meetings | < 30 seconds | **99% time reduction** |
| Insurance gap detection | Manual, often missed | Automated every session | **Lower underinsurance risk** |
| Tax savings identified | Manual, often partial | All applicable sections | **₹67,500/year avg at 30% bracket** |
| SIP accuracy | Gut feel / generic rules | Goal-specific to the rupee | **Better corpus outcomes** |
| Target market | 14 crore demat accounts | All ET users | **~10 crore potential users** |

---

## Hackathon Checklist

- [x] Clean repo structure with clear separation of concerns
- [x] FastAPI backend with RAG (LangChain + FAISS + Groq)
- [x] Single-file vanilla frontend — zero framework dependency
- [x] All 6 features: SIP, allocation, insurance, tax, timeline, emergency fund
- [x] Works without backend (local JS fallback for all calculations)
- [x] CORS configured for demo
- [x] `requirements.txt` with pinned versions
- [ ] Containerize with Docker for demo reliability
- [ ] Add Form 16 PDF upload → automated tax wizard
- [ ] Add CAMS statement upload → MF portfolio X-ray
