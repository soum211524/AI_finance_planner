# 🔥 ET Money Mentor — FIRE Path Planner

**ET AI Hackathon 2026 · Problem Statement #9 · AI Money Mentor**

A full-stack AI-powered FIRE (Financial Independence, Retire Early) planning tool for Indian investors. Built with FastAPI + LangChain RAG backend and a single-file vanilla HTML/CSS/JS frontend.

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
│  │ Engine (pure     │    │  ┌─────────────────────┐   │   │
│  │ math, instant)   │    │  │ HuggingFace Embed.  │   │   │
│  │                  │    │  │ (all-MiniLM-L6-v2)  │   │   │
│  │ • FIRE corpus    │    │  └──────────┬──────────┘   │   │
│  │ • Goal SIPs      │    │             │               │   │
│  │ • Asset alloc    │    │  ┌──────────▼──────────┐   │   │
│  │ • Tax savings    │    │  │ FAISS Vector Store  │   │   │
│  │ • Insurance gaps │    │  │ (fire_knowledge.txt)│   │   │
│  │ • Health score   │    │  └──────────┬──────────┘   │   │
│  │ • Milestones     │    │             │               │   │
│  └──────────────────┘    │  ┌──────────▼──────────┐   │   │
│                          │  │ Groq LLM (llama-3.3-70b)      │   │   │
│                          │  │ (ultra-fast, free tier)      │   │   │
│                          │  └─────────────────────┘   │   │
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

## Features

| Feature | Description |
|---|---|
| **SIP Calculator** | Goal-wise SIP breakdown: FIRE corpus, home, education, emergency fund |
| **Asset Allocation** | Age-appropriate equity/debt/gold split with glide path advice |
| **Insurance Gap Finder** | Detects underinsurance in term life, health, critical illness |
| **Tax Savings** | Section 80C, 80CCD(1B) NPS, 80D health — with rupee savings |
| **Month-by-Month Timeline** | Milestone roadmap from today to FIRE age |
| **Emergency Fund Tracker** | 6-month target with build-up SIP plan |
| **Money Health Score** | 6-dimension wellness score (0-100) |
| **AI Chat Mentor** | RAG-powered Q&A using your profile context |
| **AI Plan Narrative** | Personalized 3-paragraph FIRE action plan |

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Health check + RAG status |
| POST | `/calculate` | Pure math calculation (instant, no LLM) |
| POST | `/ask` | RAG Q&A with optional user profile context |
| POST | `/full-plan` | Calculate + AI narrative (full plan) |

---

## Quick Start

### 1. Backend Setup

```bash
cd backend

# Install dependencies
pip install -r requirements.txt

# Set your Anthropic API key
export GROQ_API_KEY=gsk_...

# Copy knowledge base
cp ../data/fire_knowledge.txt ../data/

# Start server
uvicorn main:app --reload --port 8000
```

The server will:
1. Load `fire_knowledge.txt`
2. Chunk it into 800-token segments with 150-token overlap
3. Embed using `sentence-transformers/all-MiniLM-L6-v2` (runs locally, free)
4. Build a FAISS vector index in memory
5. Wire it to Claude Sonnet via LangChain `RetrievalQA`

### 2. Frontend

```bash
# Just open the file — no build step needed
open frontend/index.html

# Or serve it
cd frontend && python3 -m http.server 3000
```

The frontend auto-detects if the backend is running and shows connection status in the header. It works in **local-only mode** (pure JS calculations) even without the backend.

---

## Project Structure

```
fire_planner/
├── backend/
│   ├── main.py              # FastAPI + LangChain RAG app
│   └── requirements.txt     # Python dependencies
├── data/
│   └── fire_knowledge.txt   # RAG knowledge base (~5000 tokens)
├── frontend/
│   └── index.html           # Single-file vanilla frontend
└── README.md
```

---

## RAG Design

**Chunking**: `RecursiveCharacterTextSplitter` with 800-char chunks, 150 overlap. Splits on `\n\n`, `\n`, `.` — preserving semantic units like "Section 80C" rules.

**Embeddings**: `sentence-transformers/all-MiniLM-L6-v2` — lightweight, runs locally (no API cost), 384-dim vectors. Good for financial English text.

**Retrieval**: FAISS cosine similarity, top-5 chunks. Profile context prepended to every question for personalized retrieval.

**LLM**: Groq LLM (llama-3.3-70b) via `langchain-groq`. Temperature 0.3 for factual financial advice. Custom system prompt enforces Indian context, specific instruments, and rupee amounts.

**Prompt strategy**: User profile injected as prefix: `"Age 28, FIRE at 45, income ₹1.2L/mo... Question: How do I..."` — this grounds retrieval in the user's specific numbers.

---

## Impact Model

| Metric | Current (status quo) | With ET FIRE Mentor | Impact |
|---|---|---|---|
| Financial advisor access | ₹25,000+/year, HNI only | Free, available to all | **Democratizes access** |
| Time to generate plan | 2-3 advisor meetings | < 30 seconds | **99% time reduction** |
| Insurance gap detection | Manual, often missed | Automated every session | **Risk reduction** |
| Tax savings identification | Manual, often partial | 100% of applicable sections | **₹67,500/year avg saving at 30% bracket** |
| SIP accuracy | Gut feel / generic rule | Goal-specific to the rupee | **Better corpus outcomes** |
| Target market | 14 crore demat accounts | All ET users | **~10 crore potential users** |

---

## Hackathon Checklist

- [x] GitHub-ready codebase with clear structure
- [x] FastAPI backend with RAG (LangChain + FAISS + Claude)
- [x] Single-file vanilla frontend (no framework dependency)
- [x] All 6 features: SIP, allocation, insurance, tax, timeline, emergency fund
- [x] Works without backend (local JS fallback)
- [x] CORS configured for demo
- [x] Requirements.txt with pinned versions
- [ ] Add `/docs` Swagger UI (auto-generated by FastAPI at `/docs`)
- [ ] Containerize with Docker for demo reliability
- [ ] Add Form 16 PDF upload → tax wizard
- [ ] Add CAMS statement upload → MF portfolio X-ray
