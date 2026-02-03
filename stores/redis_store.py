import os
from typing import List, Dict
import numpy as np
import redis
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# ==== NEW: GPU imports & knobs ====
try:
    import torch
except Exception:
    torch = None

load_dotenv()

REDIS_HOST  = os.getenv("REDIS_HOST", "127.0.0.1")
REDIS_PORT  = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB    = int(os.getenv("REDIS_DB", "0"))
INDEX_NAME  = os.getenv("REDIS_INDEX", "idx:chunks")
KEY_PREFIX  = os.getenv("REDIS_PREFIX", "doc:")

EMBED_MODEL_PATH = os.getenv("EMBED_MODEL_PATH") or ""
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")

# ==== NEW: tunables ====
EMBED_BATCH = int(os.getenv("EMBED_BATCH", "64"))           # adjust if you hit VRAM limits
CUDA_VISIBLE = os.getenv("CUDA_VISIBLE_DEVICES", None)      # respect user-set device pinning

def _b2s(x) -> str:
    if isinstance(x, (bytes, bytearray)):
        return x.decode("utf-8", errors="ignore")
    return str(x)

def _pick_device() -> str:
    if torch is not None and torch.cuda.is_available():
        return "cuda"
    # (Optional) Apple: if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available(): return "mps"
    return "cpu"

class RedisStore:
    def __init__(self):
        # decode_responses=False so vectors stay bytes
        self.r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, decode_responses=False)

        # ==== decide device once ====
        self.device = _pick_device()

        # embedder (bound to device)
        if EMBED_MODEL_PATH and os.path.exists(EMBED_MODEL_PATH):
            self.model = SentenceTransformer(EMBED_MODEL_PATH, device=self.device)
        else:
            self.model = SentenceTransformer(EMBED_MODEL_NAME, device=self.device)

        # ✅ silence transformers FutureWarning & lock behavior
        if hasattr(self.model, "tokenizer"):
            try:
                self.model.tokenizer.clean_up_tokenization_spaces = True  # set False if you prefer upcoming default
            except Exception:
                pass

        # (optional) faster matmul on Ampere/Ada
        if torch is not None and self.device == "cuda":
            try:
                torch.set_float32_matmul_precision("high")
            except Exception:
                pass

        self.dim = self.model.get_sentence_embedding_dimension()
        print(f"[Embed] Model: {EMBED_MODEL_PATH or EMBED_MODEL_NAME} | device={self.device} | dim={self.dim}")

    # ---- Index (RediSearch) ----
    def create_index(self):
        try:
            self.r.execute_command("FT.INFO", INDEX_NAME)
            return
        except redis.ResponseError:
            pass

        vec_args = [
            "TYPE", "FLOAT32",
            "DIM", str(self.dim),
            "DISTANCE_METRIC", "COSINE",
            "M", "16",
            "EF_CONSTRUCTION", "200",
        ]
        self.r.execute_command(
            "FT.CREATE", INDEX_NAME,
            "ON", "HASH",
            "PREFIX", "1", KEY_PREFIX,
            "SCHEMA",
            "text", "TEXT",
            "source", "TAG",
            "page", "NUMERIC",
            "vec", "VECTOR", "HNSW", str(len(vec_args)),
            *vec_args
        )

    # ---- Ingest ----
    def embed_batch(self, texts: List[str]) -> np.ndarray:
        X = self.model.encode(
            texts,
            batch_size=EMBED_BATCH,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return np.asarray(X, dtype="float32")

    def upsert_chunks(self, chunks: List[Dict], batch_size: int = 256):
        pipe = self.r.pipeline(transaction=False)
        buf_texts, buf_ids, buf_meta = [], [], []
        count = 0

        def flush():
            nonlocal pipe, buf_texts, buf_ids, buf_meta, count
            if not buf_texts:
                return
            vecs = self.embed_batch(buf_texts)
            for i, cid in enumerate(buf_ids):
                key = f"{KEY_PREFIX}{cid}"
                meta = buf_meta[i]
                pipe.hset(key, mapping={
                    b"text": buf_texts[i].encode("utf-8"),
                    b"source": meta["source"].encode("utf-8"),
                    b"page": str(meta["page"]).encode("utf-8"),
                    b"vec": vecs[i].tobytes()
                })
            pipe.execute()
            count += len(buf_ids)
            buf_texts, buf_ids, buf_meta = [], [], []

        for ch in chunks:
            buf_texts.append(ch["text"])
            buf_ids.append(ch["id"])
            buf_meta.append({"source": ch["meta"]["source"], "page": ch["meta"]["page"]})
            if len(buf_ids) >= batch_size:
                flush()
        flush()
        return count

    # ========== NEW: Two-pass Hybrid (BM25 + Vector) with RRF ==========
    def _tokenize_query(self, q: str) -> str:
        """Return a RediSearch OR-expression like 'dgaqa|directorate|aeronautical' (or '' if empty)."""
        import re
        toks = [t.lower() for t in re.split(r"[^A-Za-z0-9]+", q) if t]
        stop = {"the","and","for","with","this","that","from","into","over","under","shall","were","been","being"}
        toks = [t for t in toks if t not in stop and len(t) >= 2]
        return "|".join(toks) if toks else ""

    def _rrf_fuse(self, vec_ids: List[str], bm_ids: List[str], C: int = 60) -> List[str]:
        rank = {}
        for i, cid in enumerate(vec_ids):
            rank[cid] = rank.get(cid, 0.0) + 1.0/(C+i+1)
        for i, cid in enumerate(bm_ids):
            rank[cid] = rank.get(cid, 0.0) + 1.0/(C+i+1)
        return [cid for cid,_ in sorted(rank.items(), key=lambda kv: kv[1], reverse=True)]
    
   


   
    def _knn_ids(self, qvec: bytes, k: int) -> List[str]:
        q = f'*=>[KNN {k} @vec $vec AS score]'
        res = self.r.execute_command(
            "FT.SEARCH", INDEX_NAME, q,
            "PARAMS", "2", "vec", qvec,
            "RETURN", "0",
            "SORTBY", "score",
            "LIMIT", "0", str(k),
            "DIALECT", "2"
        )
        out: List[str] = []
        if res and res[0] > 0:
            i = 1
            while i + 1 < len(res):
                key = _b2s(res[i]); i += 2  # skip score
                out.append(key.replace(KEY_PREFIX, ""))
        return out

    def _bm25_ids(self, query_terms: str, k: int) -> List[str]:
        if not query_terms:
            return []
        q = f'@text:({query_terms})'
        res = self.r.execute_command(
            "FT.SEARCH", INDEX_NAME, q,
            "WITHSCORES",
            "RETURN", "0",
            "LIMIT", "0", str(k),
        )
        out: List[str] = []
        if res and res[0] > 0:
            i = 1
            while i + 1 < len(res):
                key = _b2s(res[i]); i += 2  # skip score
                out.append(key.replace(KEY_PREFIX, ""))
        return out







    def search(self, question: str, top_k: int = 24, bm25_k: int = None) -> List[Dict]:
        """
        Two-pass hybrid: run KNN and BM25 separately, fuse with RRF, then fetch docs.
        - top_k: final number of fused IDs (and KNN pool size)
        - bm25_k: optional larger pool for BM25 (default: max(50, 2*top_k))
        """
        if bm25_k is None:
            bm25_k = max(50, 2*top_k)

        # 1) embed query (normalized)
        qvec_np = self.model.encode([question], normalize_embeddings=True, convert_to_numpy=True).astype("float32")[0]
        qvec = qvec_np.tobytes()
        terms = self._tokenize_query(question)

        # 2) run both searches
        vec_ids = self._knn_ids(qvec, k=top_k)
        bm_ids  = self._bm25_ids(terms, k=bm25_k)

        # 3) fuse
        fused_ids = self._rrf_fuse(vec_ids, bm_ids, C=60)[:top_k]



        


        # 4) fetch docs
        docs = self.fetch_chunks_by_ids(fused_ids)
        # stamp a simple order score
        score_map = {cid: (len(fused_ids)-i) for i, cid in enumerate(fused_ids)}
        for d in docs:
            d["score"] = float(score_map.get(d["id"], 0))
        return docs
    # ==================================================================

    def search_by_vector(self, qvec: np.ndarray, top_k: int = 20) -> List[Dict]:
        """
        Pure vector KNN search in Redis, return only IDs.
        """
        qvec_bytes = qvec.astype("float32").tobytes()
        qstring = f"*=>[KNN {top_k} @vec $vec AS score]"
        res = self.r.execute_command(
            "FT.SEARCH", INDEX_NAME, qstring,
            "PARAMS", "2", "vec", qvec_bytes,
            "RETURN", "0",              # only return IDs
            "SORTBY", "score",
            "LIMIT", "0", str(top_k),
            "DIALECT", "2"
        )
        out: List[Dict] = []
        if not res or res[0] == 0:
            return out
        i = 1
        while i + 1 < len(res):
            key = _b2s(res[i]); i += 2  # skip score
            out.append({"id": key.replace(KEY_PREFIX, "")})
        return out

    def fetch_chunks_by_ids(self, ids: List[str]) -> List[Dict]:
        """
        Given chunk IDs (without KEY_PREFIX), return full text + meta.
        """
        if not ids:
            return []
        pipe = self.r.pipeline()
        for cid in ids:
            pipe.hgetall(f"{KEY_PREFIX}{cid}")
        results = pipe.execute()

        out: List[Dict] = []
        for cid, h in zip(ids, results):
            if not h:
                continue
            out.append({
                "id": cid,
                "text": h.get(b"text", b"").decode("utf-8"),
                "score": 0.0,  # placeholder; search() overwrites
                "meta": {
                    "source": h.get(b"source", b"").decode("utf-8"),
                    "page": int(h.get(b"page", 0) or 0)
                }
            })
        return out
