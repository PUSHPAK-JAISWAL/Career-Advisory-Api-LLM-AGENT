# fastapi_career_advisor_chromadb.py
# FastAPI CareerAdvisor (2025) - Chroma fallback, upgraded semantic retrieval & web re-ranking,
# Ollama integration, token-aware session trimming, personalized prompts, structured links output,
# and improved filtering to reduce noisy / off-topic web results.

import os
import json
import uuid
import time
import asyncio
import re
import traceback
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime

from fastapi import FastAPI, HTTPException
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

import numpy as np

load_dotenv()

# ---------------- Config ----------------
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB_NAME = os.getenv("DB_NAME", "career_advisor")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma3:4b")
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chromadb_persist")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# Token window (tokens). Default 128k tokens.
SESSION_CONTEXT_MAX_TOKENS = int(os.getenv("SESSION_CONTEXT_MAX_TOKENS", 128000))

# Ollama timeout ("None" for no timeout)
OLLAMA_TIMEOUT_ENV = os.getenv("OLLAMA_TIMEOUT", "180")
def _ollama_timeout_from_env():
    if str(OLLAMA_TIMEOUT_ENV).lower() in ("none", "null", ""):
        return None
    try:
        return float(OLLAMA_TIMEOUT_ENV)
    except Exception:
        return 60.0
OLLAMA_TIMEOUT = _ollama_timeout_from_env()

API_RATE_LIMIT_PER_MIN = int(os.getenv("API_RATE_LIMIT_PER_MIN", 60))

# Hybrid search tuning
RECENCY_WEIGHT = float(os.getenv("RECENCY_WEIGHT", 1.0))
AUTHORITY_WEIGHT = float(os.getenv("AUTHORITY_WEIGHT", 0.7))
SNIPPET_QUALITY_WEIGHT = float(os.getenv("SNIPPET_QUALITY_WEIGHT", 0.2))
MIN_AUTHORITATIVE_TO_INCLUDE = int(os.getenv("MIN_AUTHORITATIVE_TO_INCLUDE", 1))
DEFAULT_RECENT_DAYS = int(os.getenv("DEFAULT_RECENT_DAYS", 365))
AUTHORITATIVE_SUFFIXES = [".gov", ".gov.in", ".nic.in", ".edu", ".ac", ".ac.in", ".edu.in", ".org"]

# Keywords to prefer for education/admissions queries
EDU_KEYWORDS = ["college", "university", "admission", "admissions", "engineering", "computer", "computer science",
                "b.tech", "btech", "b.e", "be", "cse", "institute", "department", "faculty", "ph.d", "m.tech", "mtech"]

# Domain blacklist heuristics (strongly deprioritize unless authoritative)
DOMAIN_BLACKLIST_PARTS = ["reddit.com", "tripadvisor.com", "facebook.com", "instagram.com", "pinterest.com",
                          "youtube.com", "yelp.com", "wikipedia.org", "twitter.com", "quora.com", "medium.com"]

# ---------------- App & DB ----------------
app = FastAPI(title="CareerAdvisor RAG API (Chroma) - Improved Filtering")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])

client = motor.motor_asyncio.AsyncIOMotorClient(MONGO_URI)
db = client[DB_NAME]
users_col = db["users"]
sessions_col = db["sessions"]
docs_meta_col = db["docs_meta"]
api_keys_col = db["api_keys"]
api_usage_col = db["api_usage"]

# ---------------- Embedding + Chroma init ----------------
embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)

chroma_client = None
collection = None
collection_lock = asyncio.Lock()

_inmem_docs: Dict[str, Dict[str, Any]] = {}
_inmem_ids: List[str] = []
_inmem_embs: List[np.ndarray] = []


def create_chroma_client():
    global chroma_client, collection
    try:
        settings = Settings(persist_directory=CHROMA_PERSIST_DIR)
        chroma_client = chromadb.Client(settings)
        collection = chroma_client.get_or_create_collection(name="career_docs")
        print("Chroma initialized (new-style) with persist_directory:", CHROMA_PERSIST_DIR)
        return
    except Exception as e:
        print("Chroma new-style init failed:", e)
        traceback.print_exc()
    try:
        chroma_client = chromadb.Client()
        collection = chroma_client.get_or_create_collection(name="career_docs")
        print("Chroma initialized with default constructor.")
        return
    except Exception as e:
        print("Chroma default constructor failed:", e)
        traceback.print_exc()
    chroma_client = None
    collection = None
    print("Chroma unavailable. Using in-memory fallback.")


create_chroma_client()


def _inmem_add(doc_id: str, title: str, content: str, source: Optional[str], emb: List[float]):
    _inmem_docs[doc_id] = {"title": title, "content": content, "source": source, "embedding": np.array(emb, dtype="float32")}
    _inmem_ids.append(doc_id)
    _inmem_embs.append(np.array(emb, dtype="float32"))


def _inmem_query(qemb: np.ndarray, k: int):
    if len(_inmem_embs) == 0:
        return []
    arr = np.vstack(_inmem_embs)
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


def embed_text_np(text: str) -> np.ndarray:
    vec = embedder.encode([text], convert_to_numpy=True)[0]
    return vec.astype("float32")


def embed_text(text: str) -> List[float]:
    return embed_text_np(text).tolist()


async def add_doc_to_chroma(title: str, content: str, source: Optional[str] = None) -> str:
    doc_id = str(uuid.uuid4())
    emb = embed_text(content)
    if collection is not None:
        try:
            async with collection_lock:
                collection.add(ids=[doc_id], documents=[content],
                               metadatas=[{"title": title, "source": source}],
                               embeddings=[emb])
        except Exception as e:
            print("Chroma add runtime fail, fallback in-memory:", e)
            traceback.print_exc()
            _inmem_add(doc_id, title, content, source, emb)
    else:
        _inmem_add(doc_id, title, content, source, emb)
    await docs_meta_col.insert_one({"_id": doc_id, "title": title, "source": source, "content": content, "created_at": time.time()})
    return doc_id


# ---------------- Models ----------------
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


class APIKeyCreate(BaseModel):
    owner: str
    label: Optional[str] = "public"
    rate_limit_per_min: Optional[int] = API_RATE_LIMIT_PER_MIN


class PublicMessageIn(BaseModel):
    api_key: Optional[str] = None
    text: str
    session_id: Optional[str] = None
    prefer_recent_days: Optional[int] = None


# ---------------- Token counting (tiktoken optional) ----------------
try:
    import tiktoken  # type: ignore
    _ENC = tiktoken.get_encoding("cl100k_base")
    def count_tokens(text: str) -> int:
        if not text:
            return 0
        return len(_ENC.encode(text))
    TIKTOKEN_AVAILABLE = True
except Exception:
    TIKTOKEN_AVAILABLE = False
    def count_tokens(text: str) -> int:
        if not text:
            return 0
        return max(1, int(len(text) / 4))


# ---------------- Date extraction helpers ----------------
def try_parse_iso_like(s: str) -> Optional[datetime]:
    if not s:
        return None
    s = s.strip()
    try:
        return datetime.fromisoformat(s)
    except Exception:
        pass
    patterns = [r'(\d{4}-\d{2}-\d{2})', r'(\d{4}/\d{2}/\d{2})', r'(\d{1,2} \w{3,9} \d{4})', r'(\w{3,9} \d{1,2},? \d{4})']
    for p in patterns:
        m = re.search(p, s)
        if m:
            try:
                return datetime.fromisoformat(m.group(1))
            except Exception:
                try:
                    return datetime.strptime(m.group(1), "%Y-%m-%d")
                except Exception:
                    try:
                        return datetime.strptime(m.group(1), "%Y/%m/%d")
                    except Exception:
                        try:
                            return datetime.strptime(m.group(1), "%d %B %Y")
                        except Exception:
                            pass
    return None


def extract_publish_date_from_soup(soup: BeautifulSoup) -> Optional[datetime]:
    meta_props = [
        ("meta", {"property": "article:published_time"}),
        ("meta", {"property": "og:article:published_time"}),
        ("meta", {"name": "pubdate"}),
        ("meta", {"name": "publishdate"}),
        ("meta", {"name": "date"}),
        ("meta", {"itemprop": "datePublished"}),
        ("meta", {"property": "article:modified_time"})
    ]
    for tag, attrs in meta_props:
        t = soup.find(tag, attrs=attrs)
        if t:
            content = t.get("content") or t.get("value") or t.get_text()
            dt = try_parse_iso_like(content)
            if dt:
                return dt
    t = soup.find("time")
    if t:
        dt = try_parse_iso_like(t.get("datetime") or t.get_text())
        if dt:
            return dt
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            jd = json.loads(script.string or "{}")
            if isinstance(jd, list):
                for o in jd:
                    if isinstance(o, dict) and "datePublished" in o:
                        dt = try_parse_iso_like(o.get("datePublished"))
                        if dt:
                            return dt
            elif isinstance(jd, dict):
                if "datePublished" in jd:
                    dt = try_parse_iso_like(jd.get("datePublished"))
                    if dt:
                        return dt
                nested = jd.get("mainEntityOfPage") or jd.get("articleBody")
                if isinstance(nested, dict) and "datePublished" in nested:
                    dt = try_parse_iso_like(nested.get("datePublished"))
                    if dt:
                        return dt
        except Exception:
            continue
    return None


# ---------------- Utility: authority & snippet scoring ----------------
def is_authoritative_domain(url: str) -> bool:
    try:
        host = re.sub(r'^https?://', '', url).split('/')[0].lower()
        for suf in AUTHORITATIVE_SUFFIXES:
            if host.endswith(suf):
                return True
    except Exception:
        pass
    return False


def snippet_quality_score(snippet: Optional[str]) -> float:
    if not snippet:
        return 0.0
    l = len(snippet.strip())
    if l < 40:
        return 0.1
    if l > 1200:
        return 0.5
    return min(1.0, max(0.2, (l / 800)))


# ---------------- Semantic helpers: cosine & MMR ----------------
def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    if a is None or b is None:
        return 0.0
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def select_mmr(candidate_embs: List[np.ndarray], candidate_idxs: List[int], query_emb: np.ndarray,
               k: int, lambda_param: float = 0.6) -> List[int]:
    if k <= 0 or len(candidate_embs) == 0:
        return []
    selected = []
    unselected = list(range(len(candidate_embs)))
    sim_to_query = [cosine_sim(e, query_emb) for e in candidate_embs]
    first = int(np.argmax(sim_to_query))
    selected.append(first)
    unselected.remove(first)
    while len(selected) < min(k, len(candidate_embs)):
        mmr_scores = []
        for idx in unselected:
            sim_q = sim_to_query[idx]
            sim_sel = max([cosine_sim(candidate_embs[idx], candidate_embs[s]) for s in selected]) if selected else 0.0
            mmr_score = (lambda_param * sim_q) - ((1 - lambda_param) * sim_sel)
            mmr_scores.append((mmr_score, idx))
        mmr_scores.sort(key=lambda x: x[0], reverse=True)
        pick = mmr_scores[0][1]
        selected.append(pick)
        unselected.remove(pick)
    return selected

# Replace your current retrieve_similar(...) with this version
async def retrieve_similar(query: str, k: int = 4, top_pool_size: int = 32, mmr_lambda: float = 0.6, min_sim_threshold: float = 0.12):
    q_emb = embed_text_np(query)

    candidates = []  # tuples (doc_id, meta, doc_text, emb)
    # 1) Chroma: expanded pool, then re-embed & rerank
    if collection is not None:
        try:
            # NOTE: do NOT include "ids" in the include list -- chroma.validate_include rejects it.
            raw = collection.query(
                query_embeddings=[q_emb.tolist()],
                n_results=min(top_pool_size, 256),
                include=["metadatas", "documents", "distances"]  # <-- removed "ids"
            )
            # raw commonly contains keys: "ids", "metadatas", "documents", "distances"
            ids_list = []
            if isinstance(raw, dict) and "ids" in raw:
                # some chroma returns nested lists e.g. ids: [[id1,id2,...]]
                ids_list = raw.get("ids", [[]])[0] if isinstance(raw.get("ids", []), list) and len(raw.get("ids", []))>0 else raw.get("ids", [])
            # fallback: try to get docs/metas directly
            docs_list = []
            metas_list = []
            if isinstance(raw, dict):
                docs_list = raw.get("documents", [[]])[0] if isinstance(raw.get("documents", []), list) and len(raw.get("documents", []))>0 else raw.get("documents", [])
                metas_list = raw.get("metadatas", [[]])[0] if isinstance(raw.get("metadatas", []), list) and len(raw.get("metadatas", []))>0 else raw.get("metadatas", [])
            # If ids_list empty but docs_list present, generate pseudo-ids
            if not ids_list and docs_list:
                ids_list = [f"chroma_doc_{i}" for i in range(len(docs_list))]
            # zip them safely
            for doc_id, doc_text, meta in zip(ids_list, docs_list, metas_list if metas_list else [None]*len(ids_list)):
                try:
                    emb = embed_text_np(doc_text)
                    # ensure meta is a dict
                    meta = meta if isinstance(meta, dict) else (meta or {})
                    candidates.append((doc_id, meta, doc_text, emb))
                except Exception:
                    continue
        except Exception as e:
            # if chroma query fails, fallback to in-memory path
            print("Chroma expanded query failed, falling back to in-memory path:", e)
            traceback.print_exc()

    # 2) In-memory fallback if no chroma results
    if not candidates and len(_inmem_embs) > 0:
        try:
            qarr = q_emb.astype("float32")
            arr = np.vstack(_inmem_embs)
            dists = np.linalg.norm(arr - qarr, axis=1)
            idxs = np.argsort(dists)[:min(top_pool_size, len(dists))]
            for idx in idxs:
                doc_id = _inmem_ids[int(idx)]
                meta = _inmem_docs[doc_id]
                emb = _inmem_docs[doc_id]["embedding"]
                candidates.append((doc_id, {"title": meta.get("title"), "source": meta.get("source")}, meta.get("content"), emb))
        except Exception as e:
            print("In-memory retrieval failed:", e)
            traceback.print_exc()

    if not candidates:
        return []

    # Rerank & MMR (same logic as before)
    cand_embs = [c[3] for c in candidates]
    sims = [cosine_sim(e, q_emb) for e in cand_embs]

    def authority_boost(meta):
        src = (meta.get("source") or "") if isinstance(meta, dict) else ""
        if not src:
            return 0.0
        return 0.12 if any(src.lower().endswith(suf) for suf in AUTHORITATIVE_SUFFIXES) or any(suf in src.lower() for suf in AUTHORITATIVE_SUFFIXES) else 0.0

    boosted_scores = []
    for i, (doc_id, meta, doc_text, emb) in enumerate(candidates):
        boost = authority_boost(meta)
        boosted_scores.append(sims[i] + boost)

    filtered = []
    for i, score in enumerate(boosted_scores):
        meta = candidates[i][1]
        is_auth = authority_boost(meta) > 0
        if score >= min_sim_threshold or is_auth:
            filtered.append((i, score))

    if not filtered:
        top_idxs = list(np.argsort(sims)[::-1][:k])
        filtered = [(i, sims[i]) for i in top_idxs]

    filtered_idxs = [i for (i, s) in sorted(filtered, key=lambda x: x[1], reverse=True)]
    candidate_pool_embs = [cand_embs[i] for i in filtered_idxs]

    chosen_local_idxs = select_mmr(candidate_pool_embs, filtered_idxs, q_emb, k=min(k, len(candidate_pool_embs)), lambda_param=mmr_lambda)
    final = []
    for local_idx in chosen_local_idxs:
        global_idx = filtered_idxs[local_idx]
        doc_id, meta, doc_text, emb = candidates[global_idx]
        score = float(cosine_sim(emb, q_emb))
        final.append({"id": doc_id, "score": score, "meta": meta, "document": doc_text})
    return final


# ---------------- Upgraded web_search_hybrid (semantic re-rank + MMR + keyword boost + blacklist) ----------------
def web_search_hybrid(query: str, num: int = 6, prefer_recent: bool = True, recent_days: Optional[int] = None,
                      top_pool_size: int = 12, similarity_threshold: float = 0.09, mmr_lambda: float = 0.65):
    server_now = datetime.utcnow()
    recent_days = recent_days if recent_days is not None else DEFAULT_RECENT_DAYS

    try:
        candidate_urls = list(search(query, num_results=max(num * 3, top_pool_size * 3)))
    except Exception:
        candidate_urls = []

    seen = set()
    pool = []
    q_emb = embed_text_np(query)
    ql = query.lower()

    for url in candidate_urls:
        if url in seen:
            continue
        seen.add(url)
        try:
            r = requests.get(url, timeout=6, headers={"User-Agent": "career-advisor-bot/2025"})
            if r.status_code != 200:
                continue
            soup = BeautifulSoup(r.text, "html.parser")
            title = (soup.title.string.strip() if soup.title and soup.title.string else "") or ""
            desc = ""
            if soup.find("meta", attrs={"name": "description"}):
                desc = soup.find("meta", attrs={"name": "description"}).get("content", "") or ""
            if not desc:
                p = soup.find("p")
                desc = (p.get_text().strip()[:1200] if p else "")
            published_dt = extract_publish_date_from_soup(soup)
            days_old = (server_now - published_dt).days if published_dt else None
            auth = is_authoritative_domain(url)

            text_for_emb = (title + "\n" + desc)[:2000]
            emb = embed_text_np(text_for_emb)
            sem_sim = cosine_sim(emb, q_emb)

            # keyword boost: if title/desc contains college/admission keywords
            txt = (title + " " + desc).lower()
            kw_score = 0.0
            for kw in EDU_KEYWORDS:
                if kw in txt:
                    kw_score += 0.12
            kw_score = min(kw_score, 0.8)

            auth_b = 0.16 if auth else 0.0
            if days_old is None:
                recency_score = 0.3
            else:
                cutoff = max(1, recent_days * 3)
                recency_score = max(0.0, 1.0 - (days_old / cutoff))

            # domain blacklist penalization (strong)
            noisy_domain = any(part in url.lower() for part in DOMAIN_BLACKLIST_PARTS)
            penalty = -0.45 if (noisy_domain and not auth) else 0.0

            # final composite score (weights tuned for education search)
            final_score = (sem_sim * 0.6) + (kw_score * 0.25) + (auth_b * 0.2) + (0.15 * recency_score) + penalty

            pool.append({
                "url": url,
                "title": title,
                "snippet": desc,
                "published": published_dt.isoformat() if published_dt else None,
                "days_old": days_old,
                "authoritative": auth,
                "score": float(final_score),
                "emb": emb
            })
        except Exception:
            continue

    if not pool:
        return []

    # keep those above threshold or authoritative; otherwise pick top by score
    filtered_pool = []
    for p in pool:
        if p["score"] >= similarity_threshold or p["authoritative"]:
            filtered_pool.append(p)
    if not filtered_pool:
        filtered_pool = sorted(pool, key=lambda x: x["score"], reverse=True)[:num]

    pool_embs = [p["emb"] for p in filtered_pool]
    selected_local_idxs = select_mmr(pool_embs, list(range(len(pool_embs))), q_emb, k=min(num, len(pool_embs)), lambda_param=mmr_lambda)
    final_selected = [filtered_pool[i] for i in selected_local_idxs]
    final_selected = sorted(final_selected, key=lambda x: (x["authoritative"], x["score"]), reverse=True)

    out = []
    for p in final_selected:
        # reduce chance of returning noisy domains; skip if penalty too large and not authoritative
        if p["score"] < (similarity_threshold * 0.6) and (not p["authoritative"]):
            continue
        out.append({
            "url": p["url"],
            "snippet": p["snippet"],
            "published": p["published"],
            "days_old": p["days_old"],
            "authoritative": p["authoritative"],
            "score": round(float(p["score"]), 4)
        })
    # final fallback: if out is empty, return top num from pool (best-effort)
    if not out:
        pool_sorted = sorted(pool, key=lambda x: x["score"], reverse=True)[:num]
        for p in pool_sorted:
            out.append({
                "url": p["url"],
                "snippet": p["snippet"],
                "published": p["published"],
                "days_old": p["days_old"],
                "authoritative": p["authoritative"],
                "score": round(float(p["score"]), 4)
            })
    return out


# ---------------- Ollama call ----------------
def call_ollama_generate(prompt: str, timeout: Optional[float] = None) -> str:
    if timeout is None:
        timeout = OLLAMA_TIMEOUT
    url = f"{OLLAMA_URL}/api/generate"
    payload = {"model": OLLAMA_MODEL, "prompt": prompt, "stream": False, "temperature": 0.0}
    r = requests.post(url, json=payload, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    if isinstance(data, dict) and "response" in data:
        raw = data["response"]
    elif isinstance(data, dict) and "generated" in data:
        raw = data["generated"]
    else:
        raw = json.dumps(data) if not isinstance(data, str) else data
    return raw


# ---------------- Prompt builder ----------------
def build_prompt(system_instructions: str, user_memory_text: str, retrieved_docs: List[Dict], web_results: List[Dict],
                 session_messages: List[Dict], user_message: str, prefer_recent_days: Optional[int] = None) -> str:
    server_now = datetime.utcnow().date().isoformat()
    parts = [f"SYSTEM: {system_instructions}\n", f"SERVER_DATE_UTC: {server_now}\n"]
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
            parts.append(f"\n[{i}] {w.get('url')} (published: {w.get('published')}) - snippet: {w.get('snippet')[:300]}\n")
    else:
        parts.append("\nNOTE: Web search returned no recent results; be explicit about uncertainty and recommend verification.\n")
    if session_messages:
        parts.append("\nRECENT_CONVERSATION:")
        for m in session_messages[-8:]:
            parts.append(f"\n{m['role'].upper()}: {m['text']}")
    parts.append(f"\nUSER: {user_message}\n")

    parts.append(
        "\nFORMAT (IMPORTANT): Answer in CLEAR Markdown. Use **bold** for highlights, bullet lists for enumerations. "
        "When referencing sources, do NOT paste long raw URLs inline. Use numbered references like [1], [2] in the body. "
        "At the END include a SOURCES section in this exact pattern (one bullet per source):\n\n"
        "SOURCES:\n"
        "- https://example.com/page (Published: YYYY-MM-DD)\n"
        "- https://another.example (Published: None)\n\n"
        "Make the main answer concise (1-2 short paragraphs), then a bullet list of items with 1-line notes each. "
        "If a source is older than 2 years, append '(may be outdated)' next to the citation. If unsure, say so and recommend verification."
    )
    return "\n".join(parts)


# ---------------- Parse sources & clean assistant text ----------------
URL_REGEX = re.compile(r"https?://[^\s\)\]\}\,]+", re.IGNORECASE)

def parse_sources_and_clean_text(text: str) -> Tuple[str, List[Dict[str, Optional[str]]]]:
    links: List[Dict[str, Optional[str]]] = []
    cleaned = text

    parts = re.split(r'\nSOURCES:\s*', text, flags=re.IGNORECASE)
    if len(parts) >= 2:
        body = parts[0]
        sources_block = "\n".join(parts[1:])
        for line in sources_block.splitlines():
            line = line.strip()
            if not line:
                continue
            m = re.search(r'(https?://[^\s\)\]]+)', line)
            if not m:
                continue
            url = m.group(1).rstrip('.,)')
            pub_m = re.search(r'Published[:\s]*([0-9]{4}-[0-9]{2}-[0-9]{2}|None)', line, flags=re.IGNORECASE)
            published = pub_m.group(1) if pub_m else None
            if published and published.lower() == "none":
                published = None
            links.append({"url": url, "published": published})
        cleaned = body.strip()
    else:
        cleaned = text

    found_urls = URL_REGEX.findall(cleaned)
    for url in found_urls:
        url = url.rstrip('.,)')
        if any(l["url"] == url for l in links):
            idx = next(i for i, l in enumerate(links, start=1) if l["url"] == url)
            cleaned = cleaned.replace(url, f"[{idx}]")
            continue
        links.append({"url": url, "published": None})
        idx = len(links)
        cleaned = cleaned.replace(url, f"[{idx}]")

    return cleaned.strip(), links


# ---------------- Token-aware trimming ----------------
async def trim_session_tokens(session_id: str):
    s = await sessions_col.find_one({"_id": session_id})
    if not s:
        return
    messages = s.get("messages", [])
    total_tokens = 0
    for m in messages:
        if "tokens" not in m:
            try:
                m_tokens = count_tokens(m.get("text", ""))
            except Exception:
                m_tokens = count_tokens(m.get("text", ""))
            m["tokens"] = int(m_tokens)
        total_tokens += int(m.get("tokens", 0))

    if total_tokens <= SESSION_CONTEXT_MAX_TOKENS:
        await sessions_col.update_one({"_id": session_id}, {"$set": {"messages": messages}})
        return

    while messages and total_tokens > SESSION_CONTEXT_MAX_TOKENS:
        removed = messages.pop(0)
        total_tokens -= int(removed.get("tokens", 0))
    await sessions_col.update_one({"_id": session_id}, {"$set": {"messages": messages}})
    await sessions_col.update_one({"_id": session_id}, {"$set": {"last_trim_at": time.time(), "context_tokens": total_tokens}})


# ---------------- Memory extraction (best-effort) ----------------
def extract_memory_from_text(text: str) -> Dict[str, Any]:
    prompt = f"""You are a tool that extracts personal facts about a user from a short conversation.
Return a compact JSON with keys like: name, age, education_level, interests (list), preferred_streams (list), location, other_facts (dict).
Only include facts you can infer directly; do NOT hallucinate. Input:
{text}
Respond ONLY with valid JSON."""
    try:
        raw = call_ollama_generate(prompt, timeout=20)
        try:
            return json.loads(raw)
        except Exception:
            return {}
    except Exception:
        return {}


# ---------------- API-key utilities ----------------
async def generate_api_key(owner: str, label: str = "public", rate_limit_per_min: int = API_RATE_LIMIT_PER_MIN) -> Dict[str, Any]:
    key = str(uuid.uuid4())
    doc = {"_id": key, "owner": owner, "label": label, "rate_limit_per_min": int(rate_limit_per_min), "created_at": time.time()}
    await api_keys_col.insert_one(doc)
    return {"api_key": key, "owner": owner, "label": label, "rate_limit_per_min": rate_limit_per_min}


async def check_rate_limit(api_key: str) -> bool:
    key_doc = await api_keys_col.find_one({"_id": api_key})
    if not key_doc:
        return False
    per_min = key_doc.get("rate_limit_per_min", API_RATE_LIMIT_PER_MIN)
    cutoff = time.time() - 60
    cnt = await api_usage_col.count_documents({"api_key": api_key, "ts": {"$gte": cutoff}})
    if cnt >= per_min:
        return False
    await api_usage_col.insert_one({"api_key": api_key, "ts": time.time()})
    return True


# ---------------- Compact Endpoints (with structured links) ----------------
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
    minimal = {"user_id": user_id}
    if u.get("name"):
        minimal["name"] = u.get("name")
    return minimal


@app.delete("/user/{user_id}")
async def delete_user(user_id: str):
    await users_col.delete_one({"_id": user_id})
    await sessions_col.delete_many({"user_id": user_id})
    return {"status": "deleted"}


@app.post("/session", summary="Create a chat session")
async def create_session(payload: SessionCreate):
    session_id = str(uuid.uuid4())
    doc = {"_id": session_id, "user_id": payload.user_id, "title": payload.title, "meta": payload.meta, "messages": [], "created_at": time.time(), "context_tokens": 0}
    await sessions_col.insert_one(doc)
    return {"session_id": session_id}


@app.get("/session/{session_id}")
async def get_session(session_id: str):
    s = await sessions_col.find_one({"_id": session_id}, {"_id": 0})
    if not s:
        raise HTTPException(status_code=404, detail="session not found")
    return {"session_id": session_id}


@app.get("/session/{session_id}/tokens", summary="Get token usage for session (debug only)")
async def get_session_tokens(session_id: str):
    s = await sessions_col.find_one({"_id": session_id})
    if not s:
        raise HTTPException(status_code=404, detail="session not found")
    messages = s.get("messages", [])
    total = 0
    breakdown = []
    for m in messages:
        if "tokens" not in m:
            m_tokens = count_tokens(m.get("text", ""))
            m["tokens"] = int(m_tokens)
        else:
            m_tokens = int(m["tokens"])
        total += m_tokens
        breakdown.append({"role": m.get("role"), "ts": m.get("ts"), "tokens": m_tokens})
    return {"total_tokens": total, "limit": SESSION_CONTEXT_MAX_TOKENS, "messages": breakdown}


@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    await sessions_col.delete_one({"_id": session_id})
    return {"status": "deleted"}


@app.post("/session/{session_id}/message", summary="Send a user message (chat + RAG). Returns assistant markdown + structured links")
async def post_message(session_id: str, payload: MessageIn):
    session = await sessions_col.find_one({"_id": session_id})
    if not session:
        raise HTTPException(status_code=404, detail="session not found")

    if payload.user_id != session.get("user_id"):
        raise HTTPException(status_code=403, detail="user_id does not match session owner")

    user_tokens = count_tokens(payload.text)
    user_msg = {"role": "user", "user_id": payload.user_id, "text": payload.text, "ts": time.time(), "tokens": int(user_tokens)}
    await sessions_col.update_one({"_id": session_id}, {"$push": {"messages": user_msg}})

    await trim_session_tokens(session_id)

    retrieved = await retrieve_similar(payload.text, k=4)
    web_results = web_search_hybrid(payload.text, num=4, prefer_recent=True, recent_days=DEFAULT_RECENT_DAYS)

    user = await users_col.find_one({"_id": payload.user_id}) or {}
    user_memory_text = json.dumps(user.get("memory", {}), indent=2) if user.get("memory") else ""
    session = await sessions_col.find_one({"_id": session_id})
    session_messages = session.get("messages", []) if session else []

    system_instructions = (
        "You are an assistant that helps students choose subject streams, degree programs, "
        "and nearby government colleges. Provide succinct options, mention application timelines, "
        "required eligibility, and potential career paths. Use retrieved docs and web results to ground answers."
    )
    prompt = build_prompt(system_instructions, user_memory_text, retrieved, web_results, session_messages, payload.text)

    try:
        assistant_raw = call_ollama_generate(prompt, timeout=OLLAMA_TIMEOUT)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    assistant_text_clean, links = parse_sources_and_clean_text(assistant_raw)

    assistant_tokens = count_tokens(assistant_text_clean)
    assistant_msg = {"role": "assistant", "text": assistant_text_clean, "ts": time.time(), "tokens": int(assistant_tokens)}
    await sessions_col.update_one({"_id": session_id}, {"$push": {"messages": assistant_msg}})

    await trim_session_tokens(session_id)

    try:
        combined_for_mem = payload.text + "\n" + assistant_text_clean
        mem = extract_memory_from_text(combined_for_mem)
        if mem:
            await users_col.update_one({"_id": payload.user_id}, {"$set": {"memory": mem}}, upsert=True)
    except Exception:
        pass

    return {"assistant": assistant_text_clean, "links": links}


@app.post("/docs", summary="Ingest a doc into Chroma DB")
async def ingest_doc(payload: DocIn):
    doc_id = await add_doc_to_chroma(payload.title, payload.content, payload.source)
    return {"status": "ok", "doc_id": doc_id}


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "server_utc": datetime.utcnow().isoformat(),
        "ollama": OLLAMA_URL,
        "model": OLLAMA_MODEL,
        "chroma_persist": CHROMA_PERSIST_DIR,
        "tiktoken_available": TIKTOKEN_AVAILABLE
    }


# ---------------- API key endpoints ----------------
@app.post("/apikey", summary="Create API key")
async def create_apikey(payload: APIKeyCreate):
    rec = await generate_api_key(payload.owner, payload.label, payload.rate_limit_per_min)
    return {"status": "ok", **rec}


@app.delete("/apikey/{key}", summary="Revoke API key")
async def revoke_apikey(key: str):
    res = await api_keys_col.delete_one({"_id": key})
    if res.deleted_count == 0:
        raise HTTPException(status_code=404, detail="apikey not found")
    await api_usage_col.delete_many({"api_key": key})
    return {"status": "deleted"}


@app.get("/apikey/{key}", summary="Get API key info (minimal)")
async def get_apikey(key: str):
    d = await api_keys_col.find_one({"_id": key}, {"_id": 0})
    if not d:
        raise HTTPException(status_code=404, detail="not found")
    return {"owner": d.get("owner"), "label": d.get("label")}


# ---------------- Public chat ----------------
@app.post("/public/chat", summary="Public chat (API-keyed). Minimal response + links + session_id")
async def public_chat(payload: PublicMessageIn):
    api_key = payload.api_key
    user_id = None
    if api_key:
        key_doc = await api_keys_col.find_one({"_id": api_key})
        if not key_doc:
            raise HTTPException(status_code=401, detail="invalid api_key")
        allowed = await check_rate_limit(api_key)
        if not allowed:
            raise HTTPException(status_code=429, detail="rate limit exceeded")
        user_id = key_doc.get("owner")
    else:
        user_id = f"anon-{uuid.uuid4().hex[:8]}"

    session_id = payload.session_id
    if session_id:
        s = await sessions_col.find_one({"_id": session_id})
        if not s:
            session_id = str(uuid.uuid4())
            doc = {"_id": session_id, "user_id": user_id, "title": "Public Chat Session", "meta": {"public": True, "owner_api_key": api_key}, "messages": [], "created_at": time.time(), "context_tokens": 0}
            await sessions_col.insert_one(doc)
        else:
            owner_ok = (s.get("user_id") == user_id)
            meta = s.get("meta", {}) or {}
            api_ok = (api_key is not None and meta.get("owner_api_key") == api_key)
            if not (owner_ok or api_ok):
                raise HTTPException(status_code=403, detail="session_id belongs to another user")
    else:
        session_id = str(uuid.uuid4())
        doc = {"_id": session_id, "user_id": user_id, "title": "Public Chat Session", "meta": {"public": True, "owner_api_key": api_key}, "messages": [], "created_at": time.time(), "context_tokens": 0}
        await sessions_col.insert_one(doc)

    user_tokens = count_tokens(payload.text)
    user_msg = {"role": "user", "user_id": user_id, "text": payload.text, "ts": time.time(), "tokens": int(user_tokens)}
    await sessions_col.update_one({"_id": session_id}, {"$push": {"messages": user_msg}})

    await trim_session_tokens(session_id)

    recent_days = payload.prefer_recent_days if payload.prefer_recent_days is not None else DEFAULT_RECENT_DAYS
    retrieved = await retrieve_similar(payload.text, k=4)
    web_results = web_search_hybrid(payload.text, num=4, prefer_recent=True, recent_days=recent_days)

    user = await users_col.find_one({"_id": user_id}) or {}
    user_memory_text = ""
    if user and api_key and user.get("memory") and user.get("user_id") == user_id:
        user_memory_text = json.dumps(user.get("memory", {}), indent=2) if user.get("memory") else ""

    session = await sessions_col.find_one({"_id": session_id})
    session_messages = session.get("messages", []) if session else []

    system_instructions = (
        "You are an assistant that helps students choose subject streams, degree programs, "
        "and nearby government colleges. Provide succinct options, mention application timelines, "
        "required eligibility, and potential career paths. Use retrieved docs and web results to ground answers."
    )
    prompt = build_prompt(system_instructions, user_memory_text, retrieved, web_results, session_messages, payload.text, prefer_recent_days=recent_days)

    try:
        assistant_raw = call_ollama_generate(prompt, timeout=OLLAMA_TIMEOUT)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    assistant_text_clean, links = parse_sources_and_clean_text(assistant_raw)

    assistant_tokens = count_tokens(assistant_text_clean)
    assistant_msg = {"role": "assistant", "text": assistant_text_clean, "ts": time.time(), "tokens": int(assistant_tokens)}
    await sessions_col.update_one({"_id": session_id}, {"$push": {"messages": assistant_msg}})

    await trim_session_tokens(session_id)

    combined_for_mem = payload.text + "\n" + assistant_text_clean
    mem = extract_memory_from_text(combined_for_mem)
    if mem:
        await users_col.update_one({"_id": user_id}, {"$set": {"memory": mem}}, upsert=True)

    return {"assistant": assistant_text_clean, "links": links, "session_id": session_id}
