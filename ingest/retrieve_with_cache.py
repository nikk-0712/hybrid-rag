# file: ingest/retrieve_with_cache.py
import hashlib
from typing import Callable, List, Tuple, Any, Dict
import numpy as np

def _ensure_f32_norm(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32).reshape(-1)
    n = np.linalg.norm(v)
    return v if n == 0 else (v / n)

def _mmr_diversify(ids_in_order: List[str], lambda_diversity: float = 0.3, max_ids: int = 40) -> List[str]:
    """
    Very lightweight MMR-like de-dup on IDs only (no chunk embeddings available here).
    Keeps order bias to earlier items while penalizing duplicates.
    """
    seen = set()
    out = []
    for cid in ids_in_order:
        if cid in seen:
            continue
        out.append(cid)
        seen.add(cid)
        if len(out) >= max_ids:
            break
    return out

def retrieve_with_cache(
    query: str,
    embed_query_fn: Callable[[str], np.ndarray],
    fresh_search_fn: Callable[[np.ndarray, int], List[str]],  # returns list of chunk_ids
    memory,  # your RedisMemory instance (must have similar(), put() / put_or_merge())
    *,
    fresh_k: int = 20,
    max_chunks: int = 40,
    sim_exact: float = 0.995,     # treat as exact repeat if >= this AND text equals
    sim_min_union: float = 0.90,  # only union cached hits above this
    lambda_diversity: float = 0.3
) -> Dict[str, Any]:
    """
    Unified retrieval:
      1) Embed + normalize query.
      2) Try Option A: if top memory hit is the same query (string) and sim>=sim_exact -> shortcut.
      3) Else run fresh retrieval, then Option B: union strong cached results (sim>=sim_min_union).
      4) MMR-style de-dup, cap to max_chunks.
      5) Save back to memory.

    Returns:
      {
        "chunk_ids": [...],             # final ordered list
        "used_shortcut": bool,          # whether we skipped fresh search (Option A)
        "cache_hits": List[Tuple[str, float]],   # [(cached_query, sim), ...]
        "fresh_count": int
      }
    """
    # 1) embed + normalize
    qvec = _ensure_f32_norm(embed_query_fn(query))

    # 2) check cache
    mem_hits = memory.similar(qvec, topk=3) or []
    cache_pairs = [(h[0], h[2]) for h in mem_hits]  # [(cached_query_text, sim), ...]

    # Option A: exact same query string AND very high sim
    used_shortcut = False
    if mem_hits:
        top_q, top_ids, top_sim = mem_hits[0]
        if top_q == query and top_sim >= sim_exact:
            # Shortcut: reuse cached result
            final_ids = list(top_ids)
            used_shortcut = True
            # (Optional) still save/refresh memory entry to bump recency)
            if hasattr(memory, "put_or_merge"):
                memory.put_or_merge(query, qvec, final_ids)
            else:
                memory.put(query, qvec, final_ids)
            return {
                "chunk_ids": _mmr_diversify(final_ids, lambda_diversity, max_chunks),
                "used_shortcut": True,
                "cache_hits": cache_pairs,
                "fresh_count": 0
            }

    # 3) Fresh retrieval
    fresh_ids = list(fresh_search_fn(qvec, fresh_k))  # user-supplied function
    # 4) Option B: union strong cached results
    union_ids = list(fresh_ids)
    for cq, ids, sim in mem_hits:
        if sim >= sim_min_union:
            union_ids.extend(ids)

    # 5) De-dup/MMR + cap
    final_ids = _mmr_diversify(union_ids, lambda_diversity, max_chunks)

    # 6) Save back to memory
    if hasattr(memory, "put_or_merge"):
        memory.put_or_merge(query, qvec, final_ids)
    else:
        memory.put(query, qvec, final_ids)

    return {
        "chunk_ids": final_ids,
        "used_shortcut": False,
        "cache_hits": cache_pairs,
        "fresh_count": len(fresh_ids)
    }
