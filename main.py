"""
ET AI Money Mentor — FIRE Path Planner
FastAPI + LangChain RAG Backend  (Groq LLM)

Install:
    pip install fastapi uvicorn langchain langchain-community langchain-groq \
                faiss-cpu sentence-transformers python-multipart

Run:
    uvicorn main:app --reload --port 8000

Environment:
    export GROQ_API_KEY=your_groq_key_here
    (Free key at https://console.groq.com)
"""

import os
import json
import math
from typing import Optional
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# LangChain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.schema import Document

# ─────────────────────────────────────────────
# App setup
# ─────────────────────────────────────────────
app = FastAPI(title="ET FIRE Mentor API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────
# RAG Setup — loads once at startup
# ─────────────────────────────────────────────
rag_chain = None

def build_rag_chain():
    """Build the FAISS vector store and RAG chain from knowledge base."""
    kb_path = Path(__file__).parent / "data" / "fire_knowledge.txt"
    if not kb_path.exists():
        raise FileNotFoundError(f"Knowledge base not found at {kb_path}")

    with open(kb_path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", ".", " "]
    )
    docs = splitter.create_documents([raw_text])

    # Embeddings (free, runs locally)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # FAISS vector store
    vectorstore = FAISS.from_documents(docs, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    # LLM — Groq (ultra-fast inference, free tier available)
    # Models: llama-3.3-70b-versatile, mixtral-8x7b-32768, gemma2-9b-it
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        groq_api_key = "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",
        max_tokens=2000,
        temperature=0.3,
    )

    # Custom prompt
    prompt_template = """You are ET Money Mentor, an expert AI financial advisor specializing in FIRE (Financial Independence, Retire Early) planning for Indian investors. You have deep knowledge of Indian tax laws, mutual funds, insurance, and investment instruments.

Use the following retrieved context to give accurate, personalized advice. Always be specific with numbers when user profile is provided.

Context from knowledge base:
{context}

User question: {question}

Instructions:
- Provide specific, actionable advice
- Use Indian currency (₹) and Indian investment instruments (ELSS, PPF, NPS, SGBs etc.)
- Reference specific tax sections (80C, 80D, etc.) where relevant
- Give concrete numbers and calculations when possible
- Keep response concise but comprehensive
- Format with clear sections if multiple topics are covered
- Always mention both risks and opportunities

Answer: GIVE ANSWERS PROPERLY WITH PROPER INDENTATION AND FORMAT WITH GIVEN SPACE"""

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=False,
    )

    return chain


@app.on_event("startup")
async def startup_event():
    global rag_chain
    try:
        rag_chain = build_rag_chain()
        print("✅ RAG chain initialized successfully")
    except Exception as e:
        print(f"⚠️  RAG chain initialization failed: {e}")
        print("   API will still run — /ask endpoint will return error until fixed")
        print("   Ensure GROQ_API_KEY is set: export GROQ_API_KEY=gsk_...")


# ─────────────────────────────────────────────
# Request / Response Models
# ─────────────────────────────────────────────
class UserProfile(BaseModel):
    age: int
    fire_age: int
    monthly_income: float
    monthly_expenses: float
    existing_investments: float
    current_monthly_sip: float
    epf_ppf_balance: float
    tax_bracket: int  # 20 or 30
    home_goal_amount: Optional[float] = 0
    home_goal_year: Optional[int] = 2030
    edu_goal_amount: Optional[float] = 0
    edu_goal_year: Optional[int] = 2038


class AskRequest(BaseModel):
    question: str
    profile: Optional[UserProfile] = None


class FirePlanResponse(BaseModel):
    profile: UserProfile
    fire_corpus: float
    monthly_sip_needed: float
    sip_gap: float
    savings_rate: float
    years_to_fire: int
    emergency_fund: float
    goal_sips: dict
    asset_allocation: dict
    insurance_gaps: list
    tax_savings: dict
    milestones: list
    health_score: dict


# ─────────────────────────────────────────────
# Calculation Engine
# ─────────────────────────────────────────────
def calculate_fire_plan(p: UserProfile) -> dict:
    """Pure financial calculation — no LLM needed."""
    annual_rate = 0.12
    monthly_rate = annual_rate / 12
    inflation = 0.06
    years = p.fire_age - p.age
    months = years * 12
    current_year = 2026

    # Adjust expenses for inflation at FIRE date
    future_monthly_expenses = p.monthly_expenses * math.pow(1 + inflation, years)
    future_annual_expenses = future_monthly_expenses * 12

    # 25x rule corpus
    fire_corpus = future_annual_expenses * 25

    # Future value of existing investments
    future_existing = (p.existing_investments + p.epf_ppf_balance) * math.pow(1 + annual_rate, years)
    remaining_corpus = max(0, fire_corpus - future_existing)

    # SIP needed for remaining corpus
    if months > 0 and monthly_rate > 0:
        sip_total = remaining_corpus * monthly_rate / (math.pow(1 + monthly_rate, months) - 1)
    else:
        sip_total = remaining_corpus / max(1, months)

    # Goal SIPs
    def goal_sip(target, target_year):
        goal_months = max(6, (target_year - current_year) * 12)
        if monthly_rate > 0:
            return target * monthly_rate / (math.pow(1 + monthly_rate, goal_months) - 1)
        return target / goal_months

    home_sip = goal_sip(p.home_goal_amount or 0, p.home_goal_year or 2030) if p.home_goal_amount else 0
    edu_sip = goal_sip(p.edu_goal_amount or 0, p.edu_goal_year or 2038) if p.edu_goal_amount else 0
    emergency_fund = p.monthly_expenses * 6
    emergency_sip = emergency_fund / 12  # build in 1 year

    fire_sip_only = max(0, sip_total - home_sip - edu_sip)

    # Savings rate
    surplus = p.monthly_income - p.monthly_expenses
    savings_rate = round((surplus / p.monthly_income) * 100, 1) if p.monthly_income > 0 else 0
    sip_gap = max(0, sip_total - p.current_monthly_sip)

    # Asset allocation
    equity_pct = min(85, max(45, 110 - p.age))
    gold_pct = 10
    debt_pct = 100 - equity_pct - gold_pct

    # Insurance gaps
    insurance_gaps = []
    recommended_life_cover = p.monthly_income * 12 * 12  # 12x annual income
    if p.monthly_income * 12 * 12 > 5000000:  # If >50L needed
        insurance_gaps.append({
            "type": "term_life",
            "severity": "high",
            "title": "Term life insurance",
            "message": f"You need ₹{recommended_life_cover/100000:.0f}L term cover (12× annual income). Pure term plans cost ₹10K-20K/year for ₹1 Cr cover.",
            "action": "Buy online term plan immediately — HDFC Click2Protect, ICICI iProtect, or Tata AIA"
        })
    recommended_health = 1000000  # 10L base
    insurance_gaps.append({
        "type": "health",
        "severity": "medium",
        "title": "Health insurance",
        "message": f"Employer health cover is insufficient for FIRE. Get ₹{recommended_health/100000:.0f}L+ family floater personal policy.",
        "action": "Compare Niva Bupa, Care Health, Star Health on Policybazaar"
    })
    if p.age < 45:
        insurance_gaps.append({
            "type": "critical_illness",
            "severity": "low",
            "title": "Critical illness cover",
            "message": "₹25L-50L critical illness cover recommended. Pays lump sum on diagnosis of cancer, heart attack, stroke.",
            "action": "Add as rider to term plan or buy standalone critical illness policy"
        })

    # Tax savings
    tax_rate = p.tax_bracket / 100
    elss_benefit = min(150000, max(0, 150000 - p.epf_ppf_balance * 0.1)) * tax_rate
    nps_benefit = 50000 * tax_rate
    health_ins_benefit = 25000 * tax_rate  # 80D self+family
    hra_note = "Applicable if in rented accommodation. Saves 10-20% of rent amount from tax."

    tax_savings = {
        "section_80c": {
            "max_limit": 150000,
            "instruments": ["ELSS (best - equity returns + tax saving)", "PPF", "EPF", "5-year FD"],
            "annual_saving": round(150000 * tax_rate)
        },
        "section_80ccd_nps": {
            "max_limit": 50000,
            "annual_saving": round(nps_benefit)
        },
        "section_80d_health": {
            "max_limit": 25000,
            "annual_saving": round(health_ins_benefit),
            "note": "Up to ₹50K if parents are senior citizens"
        },
        "total_potential_saving": round((150000 + 50000 + 25000) * tax_rate),
        "regime_recommendation": "old" if (150000 + 50000 + 25000) * tax_rate > 37500 else "new"
    }

    # Milestones
    milestones = []
    milestones.append({"year": current_year, "age": p.age, "event": "FIRE journey starts", "desc": "Build 6-month emergency fund, get term + health insurance, start SIPs, max 80C+80CCD.", "phase": "foundation"})
    milestones.append({"year": current_year + 2, "age": p.age + 2, "event": "First SIP step-up", "desc": f"Increase SIPs by 10-15% with salary hike. Target ₹{(sip_total*1.1)/1000:.0f}K/mo total SIP.", "phase": "foundation"})
    milestones.append({"year": current_year + 5, "age": p.age + 5, "event": "1× annual income milestone", "desc": "Portfolio should equal 1× annual income. Add mid/small-cap and international fund exposure.", "phase": "accumulation"})
    if p.home_goal_year and p.home_goal_amount:
        milestones.append({"year": p.home_goal_year, "age": p.age + (p.home_goal_year - current_year), "event": "Home purchase", "desc": f"Deploy home corpus of ₹{p.home_goal_amount/100000:.0f}L. Reassess remaining FIRE SIPs.", "phase": "goal"})
    if p.edu_goal_year and p.edu_goal_amount:
        milestones.append({"year": p.edu_goal_year, "age": p.age + (p.edu_goal_year - current_year), "event": "Education corpus ready", "desc": f"Move ₹{p.edu_goal_amount/100000:.0f}L education corpus to low-risk debt.", "phase": "goal"})
    milestones.append({"year": current_year + years // 2, "age": p.age + years // 2, "event": "Mid-FIRE check-in", "desc": "Should have ~40% of FIRE corpus. Shift equity 2%. Review insurance adequacy and increase cover.", "phase": "accumulation"})
    milestones.append({"year": p.fire_age - 3, "age": p.fire_age - 3, "event": "Pre-FIRE glide path", "desc": "Start shifting to 50% equity / 40% debt / 10% gold. Build 2-year cash buffer. Plan withdrawal strategy.", "phase": "pre_fire"})
    milestones.append({"year": p.fire_age, "age": p.fire_age, "event": "FIRE achieved 🎯", "desc": f"₹{fire_corpus/10000000:.2f} Cr corpus. Begin 4% SWP (~₹{(fire_corpus*0.04/12)/1000:.0f}K/month). Set up bucket strategy.", "phase": "fire"})

    # Sort milestones
    milestones = sorted(milestones, key=lambda x: x["year"])

    # Money Health Score (0-100)
    scores = {
        "emergency_preparedness": min(100, round((p.existing_investments / emergency_fund) * 50 + 50) if p.existing_investments > emergency_fund else round((p.existing_investments / emergency_fund) * 50)),
        "savings_rate": min(100, round(savings_rate * 2)),
        "investment_diversification": 60,  # Default — improves with more data
        "insurance_coverage": 40,  # Flagged as gap above
        "tax_efficiency": min(100, round((p.epf_ppf_balance / 150000) * 100)) if p.epf_ppf_balance < 150000 else 100,
        "retirement_readiness": min(100, round((future_existing / fire_corpus) * 100))
    }
    overall_score = round(sum(scores.values()) / len(scores))

    return {
        "fire_corpus": round(fire_corpus),
        "monthly_sip_needed": round(sip_total),
        "fire_sip": round(fire_sip_only),
        "home_sip": round(home_sip),
        "edu_sip": round(edu_sip),
        "emergency_sip": round(emergency_sip),
        "sip_gap": round(sip_gap),
        "savings_rate": savings_rate,
        "years_to_fire": years,
        "emergency_fund": round(emergency_fund),
        "future_corpus_from_existing": round(future_existing),
        "goal_sips": {
            "fire_corpus": round(fire_sip_only),
            "home_purchase": round(home_sip),
            "child_education": round(edu_sip),
            "emergency_fund": round(emergency_sip),
        },
        "asset_allocation": {
            "equity": equity_pct,
            "debt": debt_pct,
            "gold": gold_pct,
        },
        "insurance_gaps": insurance_gaps,
        "tax_savings": tax_savings,
        "milestones": milestones,
        "health_score": {
            "overall": overall_score,
            "breakdown": scores
        }
    }


# ─────────────────────────────────────────────
# API Endpoints
# ─────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "rag_ready": rag_chain is not None}


@app.post("/calculate")
def calculate(profile: UserProfile):
    """Pure calculation endpoint — no LLM, instant response."""
    try:
        result = calculate_fire_plan(profile)
        return {"success": True, "data": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ask")
async def ask(req: AskRequest):
    """RAG-powered Q&A endpoint — uses LangChain + Groq (llama-3.3-70b)."""
    if not rag_chain:
        raise HTTPException(status_code=503, detail="RAG chain not initialized. Check GROQ_API_KEY.")

    # Enrich question with user profile if provided
    question = req.question
    if req.profile:
        p = req.profile
        context_prefix = (
            f"User profile: Age {p.age}, targeting FIRE at {p.fire_age}, "
            f"monthly income ₹{p.monthly_income:,.0f}, monthly expenses ₹{p.monthly_expenses:,.0f}, "
            f"existing investments ₹{p.existing_investments:,.0f}, tax bracket {p.tax_bracket}%. "
            f"Question: {req.question}"
        )
        question = context_prefix

    try:
        result = rag_chain.invoke({"query": question})
        answer = result.get("result", result.get("output_text", str(result)))
        return {"success": True, "answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/full-plan")
async def full_plan(profile: UserProfile):
    """Combines calculation + AI narrative for the complete FIRE plan."""
    # Step 1: Calculate
    calc = calculate_fire_plan(profile)

    # Step 2: Generate narrative summary via RAG
    narrative_question = (
        f"Age {profile.age}, FIRE at {profile.fire_age}, income ₹{profile.monthly_income:,.0f}/mo, "
        f"expenses ₹{profile.monthly_expenses:,.0f}/mo, existing investments ₹{profile.existing_investments:,.0f}, "
        f"tax bracket {profile.tax_bracket}%. "
        f"FIRE corpus needed: ₹{calc['fire_corpus']:,.0f}. Monthly SIP needed: ₹{calc['monthly_sip_needed']:,.0f}. "
        f"Current SIP: ₹{profile.current_monthly_sip:,.0f}. SIP gap: ₹{calc['sip_gap']:,.0f}. "
        f"Savings rate: {calc['savings_rate']}%. "
        f"Give me a 3-paragraph personalized FIRE action plan with the 3 most important steps to take immediately, "
        f"the biggest risk to this plan, and how to bridge the SIP gap."
    )

    ai_narrative = "AI narrative unavailable — add GROQ_API_KEY to enable."
    if rag_chain:
        try:
            result = rag_chain.invoke({"query": narrative_question})
            ai_narrative = result.get("result", result.get("output_text", ""))
        except Exception:
            pass

    return {
        "success": True,
        "calculations": calc,
        "ai_narrative": ai_narrative,
        "profile": profile.dict()
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
