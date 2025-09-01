# fastapi_career_advisor_chromadb.py
# FastAPI app with ChromaDB, googlesearch, 128k/session context window trimming,
# Ollama LLM integration (gemma3:4b), and a compact set of APIs.

import os
import json
import uuid
import time
import asyncio
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

import motor.motor_asyncio
import requests
from bs4 import BeautifulSoup
from googlesearch import search

from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

load_dotenv()

# ---------- Configuration ----------
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB_NAME = os.getenv("DB_NAME", "career_advisor")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma3:4b")
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chromadb_persist")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
# Context window max characters per session (128k)
SESSION_CONTEXT_MAX_CHARS = int(os.getenv("SESSION_CONTEXT_MAX_CHARS", 128000))

# ---------- App & DB ----------
app = FastAPI(title="CareerAdvisor RAG API (Chroma)")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])

client = motor.motor_asyncio.AsyncIOMotorClient(MONGO_URI)
db = client[DB_NAME]
users_col = db["users"]
sessions_col = db["sessions"]
docs_meta_col = db["docs_meta"]

# ---------- Embedding + Chroma init ----------
# ---------- Embedding + Chroma init (safe with fallback) ----------
import traceback
import numpy as np   # <--- new import (used by fallback)

embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)

# Chroma client / collection (attempt new-style init, fallback to default, fallback to in-memory)
chroma_client = None
collection = None
collection_lock = asyncio.Lock()

# Simple in-memory fallback store (id -> meta/content/embedding)
_inmem_docs: Dict[str, Dict[str, Any]] = {}
_inmem_ids: List[str] = []
_inmem_embs: List[np.ndarray] = []

def create_chroma_client():
    global chroma_client, collection
    # 1) Try new-style Settings with persist_directory only
    try:
        settings = Settings(persist_directory=CHROMA_PERSIST_DIR)
        chroma_client = chromadb.Client(settings)
        collection = chroma_client.get_or_create_collection(name="career_docs")
        print("Chroma initialized (new-style) with persist_directory:", CHROMA_PERSIST_DIR)
        return
    except Exception as e:
        print("Chroma new-style initialization failed:", e)
        traceback.print_exc()

    # 2) Try default constructor (some chroma versions accept this)
    try:
        chroma_client = chromadb.Client()
        collection = chroma_client.get_or_create_collection(name="career_docs")
        print("Chroma initialized with default constructor.")
        return
    except Exception as e:
        print("Chroma default constructor failed:", e)
        traceback.print_exc()

    # 3) If both attempts fail, leave chroma_client/collection as None and use fallback
    chroma_client = None
    collection = None
    print("Chroma unavailable. Using in-memory fallback vector store.")

# call during import/startup
create_chroma_client()

def _inmem_add(doc_id: str, title: str, content: str, source: Optional[str], emb: List[float]):
    _inmem_docs[doc_id] = {"title": title, "content": content, "source": source, "embedding": np.array(emb, dtype="float32")}
    _inmem_ids.append(doc_id)
    _inmem_embs.append(np.array(emb, dtype="float32"))

def _inmem_query(qemb: np.ndarray, k: int):
    if len(_inmem_embs) == 0:
        return []
    arr = np.vstack(_inmem_embs)  # shape (n, dim)
    # L2 distances:
    dists = np.linalg.norm(arr - qemb.astype("float32"), axis=1)
    idxs = np.argsort(dists)[:k]
    results = []
    for idx in idxs:
        doc_id = _inmem_ids[int(idx)]
        meta = _inmem_docs[doc_id]
        results.append({
            "id": doc_id,
            "score": float(dists[int(idx)]),
            "meta": {"title": meta.get("title"), "source": meta.get("source")},
            "document": meta.get("content")
        })
    return results

def embed_text(text: str) -> List[float]:
    # keep original embed_text API but now it uses numpy internally if needed
    vec = embedder.encode([text], convert_to_numpy=True)[0]
    return vec.astype("float32").tolist()

async def add_doc_to_chroma(title: str, content: str, source: Optional[str] = None) -> str:
    """
    Adds document either to Chroma (if available) or to in-memory fallback.
    Returns doc_id.
    """
    doc_id = str(uuid.uuid4())
    emb = embed_text(content)
    if collection is not None:
        # Use chroma collection
        try:
            async with collection_lock:
                # Chromadb expects list inputs
                collection.add(ids=[doc_id], documents=[content],
                               metadatas=[{"title": title, "source": source}],
                               embeddings=[emb])
        except Exception as e:
            # If Chroma fails at runtime, fallback to in-memory for this item
            print("Chroma add failed at runtime, falling back to in-memory for this doc:", e)
            traceback.print_exc()
            _inmem_add(doc_id, title, content, source, emb)
    else:
        # In-memory fallback
        _inmem_add(doc_id, title, content, source, emb)

    # store metadata in Mongo (unchanged behavior)
    await docs_meta_col.insert_one({"_id": doc_id, "title": title, "source": source, "content": content, "created_at": time.time()})
    return doc_id

async def retrieve_similar(query: str, k: int = 4):
    """
    Query Chroma if available; otherwise use in-memory nearest neighbors (L2).
    Returns list of {id, score, meta, document}.
    """
    qemb = np.array(embed_text(query), dtype="float32")
    # If Chroma is available, use it first
    if collection is not None:
        try:
            res = collection.query(query_embeddings=[qemb.tolist()], n_results=k, include=["metadatas", "documents", "distances"])
            items = []
            # structure: res is dict with keys ids/metadatas/documents/distances
            if not res or "ids" not in res or len(res.get("ids", [])) == 0:
                return []
            for i in range(len(res["ids"])):
                for j, doc_id in enumerate(res["ids"][i]):
                    items.append({
                        "id": doc_id,
                        "score": float(res["distances"][i][j]),
                        "meta": res["metadatas"][i][j],
                        "document": res["documents"][i][j]
                    })
            return items
        except Exception as e:
            print("Chroma query failed at runtime, falling back to in-memory:", e)
            traceback.print_exc()

    # In-memory fallback search
    return _inmem_query(qemb, k)


# ---------- Pydantic models ----------
class UserCreate(BaseModel):
    user_id: str
    name: Optional[str] = None
    age: Optional[int] = None
    gender: Optional[str] = None
    extra: Optional[Dict[str, Any]] = {}

class SessionCreate(BaseModel):
    user_id: str
    title: Optional[str] = "Chat Session"
    meta: Optional[Dict[str, Any]] = {}

class MessageIn(BaseModel):
    user_id: str
    text: str

class DocIn(BaseModel):
    title: str
    content: str
    source: Optional[str] = None

# ---------- Utilities ----------

def embed_text(text: str) -> List[float]:
    vec = embedder.encode([text], convert_to_numpy=True)[0]
    return vec.astype("float32").tolist()

async def add_doc_to_chroma(title: str, content: str, source: Optional[str] = None) -> str:
    doc_id = str(uuid.uuid4())
    emb = embed_text(content)
    async with collection_lock:
        collection.add(ids=[doc_id], documents=[content], metadatas=[{"title": title, "source": source}], embeddings=[emb])
    # store metadata in Mongo
    await docs_meta_col.insert_one({"_id": doc_id, "title": title, "source": source, "content": content, "created_at": time.time()})
    return doc_id

async def retrieve_similar(query: str, k: int = 4):
    if collection.count() == 0:
        return []
    qemb = embed_text(query)
    res = collection.query(query_embeddings=[qemb], n_results=k, include=["metadatas", "documents", "distances"])
    items = []
    for i in range(len(res["ids"])):
        for j, doc_id in enumerate(res["ids"][i]):
            items.append({
                "id": doc_id,
                "score": float(res["distances"][i][j]),
                "meta": res["metadatas"][i][j],
                "document": res["documents"][i][j]
            })
    return items

# Simple web search using googlesearch + fetch snippet
def web_search(query: str, num: int = 4):
    results = []
    try:
        urls = list(search(query, num_results=num))
    except Exception:
        # fallback: return empty
        urls = []
    for url in urls:
        try:
            r = requests.get(url, timeout=6)
            if r.status_code != 200:
                continue
            soup = BeautifulSoup(r.text, "html.parser")
            # try meta description
            desc = None
            if soup.find("meta", attrs={"name": "description"}):
                desc = soup.find("meta", attrs={"name": "description"}).get("content")
            if not desc:
                p = soup.find("p")
                desc = p.get_text().strip()[:800] if p else ""
            results.append({"url": url, "snippet": desc})
        except Exception:
            continue
    return results

# Ollama call
def call_ollama_generate(prompt: str, timeout: int = 60) -> str:
    url = f"{OLLAMA_URL}/api/generate"
    payload = {"model": OLLAMA_MODEL, "prompt": prompt, "stream": False}
    r = requests.post(url, json=payload, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    if isinstance(data, dict) and "response" in data:
        return data["response"]
    return r.text

# Build prompt
def build_prompt(system_instructions: str, user_memory_text: str, retrieved_docs: List[Dict], web_results: List[Dict], session_messages: List[Dict], user_message: str) -> str:
    parts = [f"SYSTEM: {system_instructions}\n"]
    if user_memory_text:
        parts.append(f"USER_MEMORY:\n{user_memory_text}\n")
    if retrieved_docs:
        parts.append("RETRIEVED_DOCS:")
        for i, r in enumerate(retrieved_docs, start=1):
            meta = r.get("meta", {})
            parts.append(f"\n--- Doc {i} ---\nTitle: {meta.get('title')}\nSource: {meta.get('source')}\nContent: {r.get('document')}\n")
    if web_results:
        parts.append("WEB_RESULTS:")
        for i, w in enumerate(web_results, start=1):
            parts.append(f"\n[{i}] {w.get('url')} - {w.get('snippet')[:300]}\n")
    if session_messages:
        parts.append("\nCONVERSATION_HISTORY:")
        for m in session_messages[-8:]:
            parts.append(f"\n{m['role'].upper()}: {m['text']}")
    parts.append(f"\nUSER: {user_message}\n")
    parts.append("\nASSISTANT: (Be concise, provide actionable steps, cite source titles/URLs where used.)\n")
    return "\n".join(parts)

# Trim session messages so total characters <= SESSION_CONTEXT_MAX_CHARS
async def trim_session_messages(session_id: str):
    s = await sessions_col.find_one({"_id": session_id})
    if not s:
        return
    messages = s.get("messages", [])
    total_chars = sum(len(m.get("text", "")) for m in messages)
    if total_chars <= SESSION_CONTEXT_MAX_CHARS:
        return
    # remove oldest until under limit
    while messages and total_chars > SESSION_CONTEXT_MAX_CHARS:
        removed = messages.pop(0)
        total_chars -= len(removed.get("text", ""))
    await sessions_col.update_one({"_id": session_id}, {"$set": {"messages": messages}})

# Memory extraction (same approach as before)
def extract_memory_from_text(text: str) -> Dict[str, Any]:
    prompt = f"""You are a tool that extracts personal facts about a user from a short conversation.
Return a JSON object with keys like: name, age, education_level, interests (list), preferred_streams (list), location, other_facts (dict). Only include keys you can infer directly; do not hallucinate. Input:
{text}
Respond ONLY with valid JSON."""
    raw = call_ollama_generate(prompt, timeout=20)
    try:
        return json.loads(raw)
    except Exception:
        return {}

# ---------- Compact API Endpoints ----------
@app.post("/user", summary="Create or upsert user")
async def create_user(payload: UserCreate):
    doc = payload.dict()
    doc["_id"] = payload.user_id
    await users_col.update_one({"_id": payload.user_id}, {"$set": doc}, upsert=True)
    return {"status": "ok", "user_id": payload.user_id}

@app.get("/user/{user_id}")
async def get_user(user_id: str):
    u = await users_col.find_one({"_id": user_id}, {"_id": 0})
    if not u:
        raise HTTPException(status_code=404, detail="user not found")
    return u

@app.delete("/user/{user_id}")
async def delete_user(user_id: str):
    await users_col.delete_one({"_id": user_id})
    await sessions_col.delete_many({"user_id": user_id})
    return {"status": "deleted"}

@app.post("/session", summary="Create a chat session")
async def create_session(payload: SessionCreate):
    session_id = str(uuid.uuid4())
    doc = {"_id": session_id, "user_id": payload.user_id, "title": payload.title, "meta": payload.meta, "messages": [], "created_at": time.time()}
    await sessions_col.insert_one(doc)
    return {"session_id": session_id}

@app.get("/session/{session_id}")
async def get_session(session_id: str):
    s = await sessions_col.find_one({"_id": session_id}, {"_id": 0})
    if not s:
        raise HTTPException(status_code=404, detail="session not found")
    return s

@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    await sessions_col.delete_one({"_id": session_id})
    return {"status": "deleted"}

@app.post("/session/{session_id}/message", summary="Send a user message (chat + RAG)")
async def post_message(session_id: str, payload: MessageIn):
    session = await sessions_col.find_one({"_id": session_id})
    if not session:
        raise HTTPException(status_code=404, detail="session not found")

    # Append user message
    user_msg = {"role": "user", "user_id": payload.user_id, "text": payload.text, "ts": time.time()}
    await sessions_col.update_one({"_id": session_id}, {"$push": {"messages": user_msg}})

    # Trim session if exceeding context window
    await trim_session_messages(session_id)

    # Retrieve top docs from Chroma
    retrieved = await retrieve_similar(payload.text, k=4)

    # Web search
    web_results = web_search(payload.text, num=3)

    # Load user memory
    user = await users_col.find_one({"_id": payload.user_id}) or {}
    user_memory_text = json.dumps(user.get("memory", {}), indent=2) if user.get("memory") else ""

    # Load recent session messages
    session = await sessions_col.find_one({"_id": session_id})
    session_messages = session.get("messages", []) if session else []

    system_instructions = (
        "You are an assistant that helps students choose subject streams, degree programs, "
        "and nearby government colleges. Provide succinct options, mention application timelines, "
        "required eligibility, and potential career paths. Use retrieved docs and web results to ground answers."
    )
    prompt = build_prompt(system_instructions, user_memory_text, retrieved, web_results, session_messages, payload.text)

    try:
        assistant_text = call_ollama_generate(prompt, timeout=60)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    assistant_msg = {"role": "assistant", "text": assistant_text, "ts": time.time()}
    await sessions_col.update_one({"_id": session_id}, {"$push": {"messages": assistant_msg}})

    # Try to extract memory
    combined_for_mem = payload.text + "\n" + assistant_text
    mem = extract_memory_from_text(combined_for_mem)
    if mem:
        await users_col.update_one({"_id": payload.user_id}, {"$set": {"memory": mem}}, upsert=True)

    return {"assistant": assistant_text, "retrieved": retrieved, "web_results": web_results, "memory_extracted": mem}

@app.post("/docs", summary="Ingest a doc into Chroma DB")
async def ingest_doc(payload: DocIn):
    doc_id = await add_doc_to_chroma(payload.title, payload.content, payload.source)
    return {"status": "ok", "doc_id": doc_id}

@app.get("/health")
async def health():
    return {"status": "ok", "ollama": OLLAMA_URL, "model": OLLAMA_MODEL, "chroma_persist": CHROMA_PERSIST_DIR}