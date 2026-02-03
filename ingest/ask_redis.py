#!/usr/bin/env python3
# file: ask_redis.py  (RRF hybrid retrieval + Top-10 dumps to a single combined text file)

import sys, os, re
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))  # allow imports from project root

import argparse, json, time
from typing import List, Dict, Tuple
import numpy as np
import redis
import requests

# Project imports
from ingest.retrieve_with_cache import retrieve_with_cache
from stores.redis_store import RedisStore  # must provide: model.encode(...), search_by_vector(...), fetch_chunks_by_ids(...)

# ---------- Optional: reranker imports / device ----------
try:
    import torch
except Exception:
    torch = None

try:
    from sentence_transformers import CrossEncoder
except Exception:
    CrossEncoder = None


def _pick_device() -> str:
    if torch is not None and torch.cuda.is_available():
        return "cuda"
    return "cpu"


# -------------------- Base Config --------------------
# Ollama
OLLAMA_URL   = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b-instruct-q4_K_M")

# Memory (Redis)
REDIS_HOST = os.getenv("REDIS_HOST", "127.0.0.1")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB   = int(os.getenv("REDIS_DB", "0"))

# ---------- Reranker config ----------
USE_RERANKER   = os.getenv("USE_RERANKER", "1") not in ("0", "false", "False", "")
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
RERANK_TOPN    = int(os.getenv("RERANK_TOPN", "50"))
RERANK_KEEP    = int(os.getenv("RERANK_KEEP", "24"))
RERANK_BATCH   = int(os.getenv("RERANK_BATCH", "32"))

# ---------- RRF (fusion) config ----------
RRF_K        = int(os.getenv("RRF_K", "60"))
RRF_W_DENSE  = float(os.getenv("RRF_W_DENSE", "1.0"))
RRF_W_BM25   = float(os.getenv("RRF_W_BM25",  "1.0"))
RRF_W_KW     = float(os.getenv("RRF_W_KW",    "0.5"))
RRF_USE_KW   = os.getenv("RRF_USE_KW", "0") in ("1","true","True")

# Paths produced by your indexer
BM25_PATH    = os.getenv("OUT_BM25", "Data/output/bm25.pkl")
DUCK_PATH    = os.getenv("OUT_DUCK", "Data/output/chunks.duckdb")


# -------------------- Utils --------------------
def count_tokens_rough(s: str) -> int:
    return max(1, len(s)//4)


def split_sentences(text: str) -> List[str]:
    parts = []
    start = 0
    for i, ch in enumerate(text):
        if ch in ".!?\n":
            seg = text[start:i+1].strip()
            if seg:
                parts.append(seg)
            start = i+1
    if start < len(text):
        tail = text[start:].strip()
        if tail:
            parts.append(tail)
    return parts


def strict_pack(cands: List[Dict], max_tokens: int, per_source_cap: int = 3) -> List[Dict]:
    items = sorted(cands, key=lambda c: c.get("score", 0.0), reverse=True)
    used, used_tokens, per_src = [], 0, {}
    for c in items:
        src = c["meta"]["source"]
        if per_src.get(src, 0) >= per_source_cap:
            continue
        t = count_tokens_rough(c["text"])
        if used_tokens + t <= max_tokens:
            used.append(c); used_tokens += t; per_src[src] = per_src.get(src, 0) + 1
        else:
            remain = max_tokens - used_tokens
            if remain <= 0:
                break
            sents = split_sentences(c["text"])
            buf, toks = [], 0
            for s in sents:
                st = count_tokens_rough(s)
                if toks + st > remain:
                    break
                buf.append(s); toks += st
            if buf:
                cc = dict(c); cc["text"] = " ".join(buf)
                used.append(cc); used_tokens += toks; per_src[src] = per_src.get(src, 0) + 1
            break
    return used


def build_prompt(ctx: List[Dict], question: str) -> str:
    lines = [
        "You are a strict RAG assistant. Use ONLY the CONTEXT. If unsure, say you don't know.",
        "\nCONTEXT:"
    ]
    for c in ctx:
        m = c["meta"]
        lines.append(f"[Source: {m['source']} p.{m['page']} | id={c['id']}]\n{c['text']}\n")
    lines += [
        f"\nQUESTION: {question}",
        "\nINSTRUCTIONS: Answer concisely and cite using [source p.X id]."
    ]
    return "\n".join(lines)


def call_ollama(prompt: str, num_ctx: int = 8192, temperature: float = 0.2) -> str:
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "options": {"num_ctx": num_ctx, "temperature": temperature},
        "stream": False
    }
    r = requests.post(f"{OLLAMA_URL}/api/generate", json=payload, timeout=600)
    r.raise_for_status()
    return r.json().get("response", "")


# -------------------- RRF fuse --------------------
from collections import defaultdict

def rrf_fuse(
    rankings: List[List[Tuple[str, float]]],
    k: int = 60,
    weights: List[float] | None = None
) -> Tuple[List[Tuple[str, float]], Dict[str, float]]:
    if weights is None:
        weights = [1.0] * len(rankings)
    scores: Dict[str, float] = defaultdict(float)
    for lst, w in zip(rankings, weights):
        for rank, (cid, _score) in enumerate(lst, start=1):
            scores[cid] += w * (1.0 / (k + rank))
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return ranked, scores


# -------------------- BM25 runtime --------------------
class BM25Runtime:
    def __init__(self, path: str = "Data/output/bm25.pkl"):
        import pickle
        from rank_bm25 import BM25Okapi
        with open(path, "rb") as f:
            pack = pickle.load(f)
        self.ids = pack["ids"]
        self.tokens = pack["tokens"]
        self.bm25 = BM25Okapi(self.tokens)

    def search(self, query_tokens: List[str], top_k: int = 50) -> List[Tuple[str, float]]:
        scores = self.bm25.get_scores(query_tokens)
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [(self.ids[i], float(scores[i])) for i in order]


def regex_tokenize(s: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9][A-Za-z0-9_\-./]+", s.lower())


# -------------------- Optional: simple keyword scan via DuckDB --------------------
def keyword_scan(question_text: str, duck_path: str, top_k: int = 30) -> List[Tuple[str, float]]:
    try:
        import duckdb
        q = question_text.replace("'", "''")
        con = duckdb.connect(duck_path, read_only=True)
        rows = con.execute(f"""
            SELECT id
            FROM chunks
            WHERE lower(text) LIKE '%' || lower('{q}') || '%'
            LIMIT {top_k}
        """).fetchall()
        con.close()
        return [(rid, 0.0) for (rid,) in rows]
    except Exception:
        return []


# -------------------- Plain-text Top-10 writers (unchanged) --------------------
def _ensure_parent_dir(path: str):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def _write_section_header(f, title: str):
    f.write("\n" + "=" * 96 + "\n")
    f.write(title + "\n")
    f.write("=" * 96 + "\n\n")

def _write_item_block(f, rank: int, cid: str, score: float, doc: Dict, label: str):
    meta = doc.get("meta", {}) if doc else {}
    src  = meta.get("source", "?")
    pg   = meta.get("page", "?")
    txt  = (doc or {}).get("text", "") or ""
    f.write(f"{label} #{rank}\n")
    f.write(f"ID: {cid}\n")
    f.write(f"Score: {score:.6f}\n")
    f.write(f"Source: {src}  Page: {pg}\n")
    f.write("-" * 96 + "\n")
    f.write(txt.rstrip() + "\n")
    f.write("-" * 96 + "\n\n")

def save_pre_fusion_top10_plain(out_path: str,
                                dense_ranked: List[Tuple[str, float]],
                                bm25_ranked: List[Tuple[str, float]],
                                fetch_fn,
                                limit: int = 10):
    _ensure_parent_dir(out_path)
    with open(out_path, "a", encoding="utf-8") as f:
        _write_section_header(f, "PRE-FUSION TOP-10 (FAISS/Dense)")
        d_top = dense_ranked[:limit]
        d_ids = [cid for cid, _ in d_top]
        d_docs = {c["id"]: c for c in fetch_fn(d_ids)} if d_ids else {}
        for i, (cid, sc) in enumerate(d_top, start=1):
            _write_item_block(f, i, cid, sc, d_docs.get(cid), "FAISS/Dense")

        _write_section_header(f, "PRE-FUSION TOP-10 (BM25)")
        b_top = bm25_ranked[:limit]
        b_ids = [cid for cid, _ in b_top]
        b_docs = {c["id"]: c for c in fetch_fn(b_ids)} if b_ids else {}
        for i, (cid, sc) in enumerate(b_top, start=1):
            _write_item_block(f, i, cid, sc, d_docs.get(cid), "BM25")

def save_post_fusion_top10_plain(out_path: str,
                                 fused_ranked: List[Tuple[str, float]],
                                 fetch_fn,
                                 limit: int = 10):
    _ensure_parent_dir(out_path)
    with open(out_path, "a", encoding="utf-8") as f:
        _write_section_header(f, "POST-FUSION TOP-10 (RRF/Fused)")
        top = fused_ranked[:limit]
        ids = [cid for cid, _ in top]
        docs = {c["id"]: c for c in fetch_fn(ids)} if ids else {}
        for i, (cid, sc) in enumerate(top, start=1):
            _write_item_block(f, i, cid, sc, docs.get(cid), "RRF/Fused")


# -------------------- Reranker --------------------
class SafeReranker:
    def __init__(self, model_name: str, batch_size: int = 32):
        self.enabled = False
        self.model = None
        self.batch = batch_size
        self.device = _pick_device()
        if not USE_RERANKER or CrossEncoder is None:
            return
        try:
            self.model = CrossEncoder(model_name, device=self.device)
            self.enabled = True
            print(f"[Reranker] {model_name} on {self.device}")
        except Exception as e:
            print(f"[Reranker] Disabled ({e})")

    def rerank(self, question: str, cands: List[Dict], keep: int) -> List[Dict]:
        if not self.enabled or not cands:
            return cands
        seeds = sorted(cands, key=lambda x: x.get("score", 0.0), reverse=True)[:max(keep, RERANK_TOPN)]
        pairs = [(question, c["text"]) for c in seeds]
        scores = self.model.predict(pairs, batch_size=self.batch)
        for c, s in zip(seeds, scores):
            c["score"] = float(s)
        seeds.sort(key=lambda x: x["score"], reverse=True)
        return seeds[:keep]


# -------------------- Lightweight Memory --------------------
class RedisMemory:
    def __init__(self, dim: int):
        self.r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, decode_responses=False)
        self.dim = dim

    def put(self, query: str, qvec: np.ndarray, chunk_ids: List[str]):
        key = f"mem:{int(time.time()*1e6)}"
        pipe = self.r.pipeline(False)
        pipe.hset(key, mapping={
            b"query": query.encode("utf-8"),
            b"qvec": qvec.astype("float32").tobytes(),
            b"chunk_ids": json.dumps(chunk_ids).encode("utf-8"),
            b"ts": str(time.time()).encode("utf-8")
        })
        pipe.lpush(b"mem:index", key.encode("utf-8"))
        pipe.ltrim(b"mem:index", 0, 999)
        pipe.execute()

    def put_or_merge(self, query: str, qvec: np.ndarray, chunk_ids: List[str]):
        self.put(query, qvec, chunk_ids)

    def similar(self, qvec: np.ndarray, topk=3):
        keys = self.r.lrange(b"mem:index", 0, 999)
        sims = []
        for k in keys:
            h = self.r.hgetall(k)
            if not h:
                continue
            v = np.frombuffer(h[b"qvec"], dtype="float32")
            if v.shape[0] != self.dim:
                continue
            sim = float(np.dot(v, qvec))
            q = h[b"query"].decode("utf-8")
            ids = json.loads(h[b"chunk_ids"].decode("utf-8"))
            sims.append((q, ids, sim))
        sims.sort(key=lambda x: x[2], reverse=True)
        return sims[:topk]


# -------------------- Combined writer --------------------
def write_combined(save_dir: str, question: str, sections: List[str], filename: str = "final_results.txt"):
    os.makedirs(save_dir, exist_ok=True)
    outp = os.path.join(save_dir, filename)
    with open(outp, "w", encoding="utf-8") as f:
        f.write("=== FINAL RESULTS ===\n\n")
        f.write(f"QUESTION: {question}\n\n")
        f.write("\n".join(sections).rstrip() + "\n")
    print(f"[Saved combined] → {outp}")


# -------------------- Main --------------------
def main():
    ap = argparse.ArgumentParser(description="Cache-aware Redis RAG → (RRF hybrid) → (optional GPU rerank) → strict pack → Ollama")
    ap.add_argument("question", nargs="+")
    ap.add_argument("--topk", type=int, default=24, help="fresh retrieval pool size (post-RRF)")
    ap.add_argument("--topm", type=int, default=16, help="max IDs to keep before packing (and min rerank keep)")
    ap.add_argument("--max_ctx_tokens", type=int, default=1800, help="token budget for context")
    ap.add_argument("--per_source_cap", type=int, default=3, help="limit chunks per source in context")
    ap.add_argument("--sim_exact", type=float, default=0.995, help="exact repeat shortcut threshold")
    ap.add_argument("--sim_min_union", type=float, default=0.90, help="union threshold for cached hits")
    ap.add_argument("--top10_out", type=str, default="Data/output/top10_chunks.txt",
                    help="path to save Top-10 chunks (plain text)")
    ap.add_argument("--top10_stage", type=str, choices=["pre", "post", "both"], default="pre",
                    help="dump Top-10 before fusion, after fusion, or both")
    ap.add_argument("--answer_all", type=int, default=1,
                    help="1: print answers for bm25, dense, fused; 0: single mode via --mode")
    ap.add_argument("--mode", choices=["fused","bm25","dense"], default="fused")
    ap.add_argument("--save_answers_dir", type=str, default="Data/output")

    args = ap.parse_args()
    question = " ".join(args.question).strip()

    # Init store + BM25 runtime
    store = RedisStore()
    bm25rt = BM25Runtime(BM25_PATH)

    # RRF score cache
    last_rrf_scores: Dict[str, float] = {}

    # ---- helpers bound to this store ----
    def embed_query_offline(text: str) -> np.ndarray:
        return store.model.encode([text], normalize_embeddings=True, convert_to_numpy=True).astype("float32")[0]

    def fresh_search_rrf(question_text: str, qvec: np.ndarray, final_k: int) -> List[str]:
        nonlocal last_rrf_scores

        # Dense
        dense_raw = store.search_by_vector(qvec, top_k=max(final_k, 24))
        dense_ranked = []
        for r in dense_raw:
            if isinstance(r, dict):
                dense_ranked.append((r.get("id"), float(r.get("score", 0.0))))
            else:
                dense_ranked.append((r, 0.0))

        # BM25
        qtok = regex_tokenize(question_text)
        bm25_ranked = bm25rt.search(qtok, top_k=max(final_k * 2, 50))

        # PRE-FUSION dump if requested
        if args.top10_stage in ("pre", "both"):
            if args.top10_stage == "pre" and os.path.exists(args.top10_out):
                os.remove(args.top10_out)
            save_pre_fusion_top10_plain(args.top10_out, dense_ranked, bm25_ranked, store.fetch_chunks_by_ids, limit=10)

        # Optional keyword scan
        rankings = [dense_ranked, bm25_ranked]
        weights  = [RRF_W_DENSE, RRF_W_BM25]
        if RRF_USE_KW:
            kw_ranked = keyword_scan(question_text, DUCK_PATH, top_k=30)
            if kw_ranked:
                rankings.append(kw_ranked)
                weights.append(RRF_W_KW)

        # Fuse
        fused_ranked, scores = rrf_fuse(rankings, k=RRF_K, weights=weights)
        last_rrf_scores = dict(scores)

        # POST-FUSION dump if requested
        if args.top10_stage in ("post", "both"):
            if args.top10_stage == "post" and os.path.exists(args.top10_out):
                os.remove(args.top10_out)
            save_post_fusion_top10_plain(args.top10_out, fused_ranked, store.fetch_chunks_by_ids, limit=10)

        return [cid for cid, _ in fused_ranked[:final_k]]

    # ---------- Helpers that close over store/bm25rt ----------
    def bm25_only_ids(question_text: str, final_k: int) -> list[str]:
        qtok = regex_tokenize(question_text)
        bm25_ranked = bm25rt.search(qtok, top_k=max(final_k * 2, 50))
        return [cid for cid, _ in bm25_ranked[:final_k]]

    def dense_only_ids(qvec: np.ndarray, final_k: int) -> list[str]:
        dense_raw = store.search_by_vector(qvec, top_k=max(final_k, 24))
        ids: list[str] = []
        for item in dense_raw:
            if isinstance(item, dict):
                iid = item.get("id")
                if iid:
                    ids.append(iid)
            elif isinstance(item, str):
                ids.append(item)
        return ids[:final_k]

    def answer_section(mode_label: str, ids: list[str],
                       max_ctx_tokens: int, per_source_cap: int, question: str) -> str:
        """Build a single text section for one retrieval mode without writing to disk."""
        docs = store.fetch_chunks_by_ids(ids)
        packed = strict_pack(docs, max_ctx_tokens, per_source_cap=per_source_cap)
        prompt = build_prompt(packed, question)
        ans = call_ollama(prompt, num_ctx=max(max_ctx_tokens, 4096))

        # Console print for visibility
        print(f"\n=== ANSWER ({mode_label}) ===\n{ans}\n")
        print("=== CITATIONS ===")
        for c in packed:
            print(f"- {c['meta']['source']} p.{c['meta']['page']} id={c['id']}")

        # Return a formatted section
        lines = [f"=== ANSWER ({mode_label.upper()}) ===", "", ans, "", "=== CITATIONS ==="]
        for c in packed:
            lines.append(f"- {c['meta']['source']} p.{c['meta']['page']} id={c['id']}")
        lines.append("")  # trailing newline
        return "\n".join(lines)

    # Adapter matching retrieve_with_cache signature
    def run_vector_search(qvec: np.ndarray, k: int) -> List[str]:
        return fresh_search_rrf(question, qvec, final_k=k)

    # Build memory (infer dim from a temp encode)
    qvec_tmp = embed_query_offline(question)
    mem = RedisMemory(dim=qvec_tmp.shape[0])

    # -------- Branch: gather sections; write ONE combined file --------
    sections: List[str] = []
    if args.answer_all:
        # 1) BM25-only
        bm_ids = bm25_only_ids(question, final_k=args.topk)
        sections.append(answer_section("bm25", bm_ids, args.max_ctx_tokens, args.per_source_cap, question))

        # 2) Dense-only
        de_ids = dense_only_ids(qvec_tmp, final_k=args.topk)
        sections.append(answer_section("dense", de_ids, args.max_ctx_tokens, args.per_source_cap, question))

        # 3) Fused (cache-aware → RRF)
        res = retrieve_with_cache(
            query=question,
            embed_query_fn=embed_query_offline,
            fresh_search_fn=run_vector_search,  # calls fresh_search_rrf()
            memory=mem,
            fresh_k=args.topk,
            max_chunks=args.topm,
            sim_exact=args.sim_exact,
            sim_min_union=args.sim_min_union,
        )
        fused_ids = res["chunk_ids"]
        sections.append(answer_section("fused", fused_ids, args.max_ctx_tokens, args.per_source_cap, question))

    else:
        # Single mode → still write a final_results.txt with that one section
        if args.mode == "bm25":
            ids = bm25_only_ids(question, final_k=args.topk)
            sections.append(answer_section("bm25", ids, args.max_ctx_tokens, args.per_source_cap, question))
        elif args.mode == "dense":
            ids = dense_only_ids(qvec_tmp, final_k=args.topk)
            sections.append(answer_section("dense", ids, args.max_ctx_tokens, args.per_source_cap, question))
        else:
            res = retrieve_with_cache(
                query=question,
                embed_query_fn=embed_query_offline,
                fresh_search_fn=run_vector_search,
                memory=mem,
                fresh_k=args.topk,
                max_chunks=args.topm,
                sim_exact=args.sim_exact,
                sim_min_union=args.sim_min_union,
            )
            ids = res["chunk_ids"]
            sections.append(answer_section("fused", ids, args.max_ctx_tokens, args.per_source_cap, question))

    # Write ONE combined file named "final_results.txt"
    write_combined(args.save_answers_dir, question, sections, filename="final_results.txt")


if __name__ == "__main__":
    main()
