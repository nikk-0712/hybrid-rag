#!/usr/bin/env python3
# index_all.py
import os, sys, json, glob, pickle, re
from typing import List, Dict, Tuple
import numpy as np
import duckdb
from sentence_transformers import SentenceTransformer



# ========= Outputs =========
OUT_FAISS = os.getenv("OUT_FAISS", "Data/output/faiss.index")
OUT_IDS   = os.getenv("OUT_IDS",   "Data/output/faiss.ids.pkl")
OUT_BM25  = os.getenv("OUT_BM25",  "Data/output/bm25.pkl")
OUT_DUCK  = os.getenv("OUT_DUCK",  "Data/output/chunks.duckdb")

# ========= Env toggles =========
FAISS_USE_COSINE   = os.getenv("FAISS_USE_COSINE", "1") not in ("0","false","False")
FAISS_ADD_BATCH    = int(os.getenv("FAISS_ADD_BATCH", "100000"))  # FAISS add batch
EMBED_BATCH        = int(os.getenv("EMBED_BATCH", "4096"))        # texts per encode pass
EMBED_INFER_BS     = int(os.getenv("EMBED_INFER_BS", "64"))       # model.encode batch_size
FAISS_USE_ALL_GPUS = os.getenv("FAISS_USE_ALL_GPUS", "0") in ("1","true","True")
FAISS_GPU_ID       = int(os.getenv("FAISS_GPU", "0"))
FORCE_FAISS_GPU    = os.getenv("FORCE_FAISS_GPU", "0") in ("1","true","True")
CHUNKS_GLOB        = os.getenv("CHUNKS_GLOB", "Data/output/*.chunks.jsonl")

# ========= Diagnostics =========
def torch_cuda_diag() -> dict:
    try:
        import torch
        return {
            "torch_version": torch.__version__,
            "torch_cuda_version": getattr(torch.version, "cuda", None),
            "cuda_available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "device_name_0": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }
    except Exception as e:
        return {"error": str(e), "cuda_available": False, "device_count": 0, "device_name_0": None}

def faiss_gpu_diag() -> dict:
    info = {
        "faiss_import_ok": False, "faiss_version": None,
        "has_get_num_gpus": False, "num_gpus": 0,
        "has_gpu_api": False, "gpu_resources_ok": False, "error": None
    }
    try:
        import faiss
        info["faiss_import_ok"] = True
        info["faiss_version"] = getattr(faiss, "__version__", "unknown")
        info["has_get_num_gpus"] = hasattr(faiss, "get_num_gpus")
        if info["has_get_num_gpus"]:
            try:
                info["num_gpus"] = faiss.get_num_gpus()
            except Exception:
                info["num_gpus"] = -1
        info["has_gpu_api"] = hasattr(faiss, "StandardGpuResources")
        if info["has_gpu_api"]:
            try:
                faiss.StandardGpuResources()
                info["gpu_resources_ok"] = True
            except Exception as e:
                info["error"] = f"StandardGpuResources failed: {e}"
    except Exception as e:
        info["error"] = f"faiss import failed: {e}"
    return info

def suggest_install(torch_cuda: str | None) -> str:
    if not torch_cuda:
        return (
            "# Torch CUDA not detected. Install a CUDA build of PyTorch first:\n"
            "pip install --index-url https://download.pytorch.org/whl/cu118 torch torchvision torchaudio\n"
            "# Then install FAISS-GPU (cu118):\n"
            "pip install --no-cache-dir faiss-gpu==1.7.2\n"
        )
    cu = (torch_cuda or "").strip()
    if cu.startswith("11.8"):
        return "pip uninstall -y faiss faiss-cpu faiss-gpu && pip install --no-cache-dir faiss-gpu==1.7.2"
    if cu.startswith("12."):
        return (
            "mamba create -n faissgpu python=3.10 -y && mamba activate faissgpu && "
            "mamba install -c conda-forge faiss-gpu=1.7.4 \"cuda-version>=12,<13\" -y && "
            "pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio"
        )
    return (
        "# Try cu118 wheel first (pip):\n"
        "pip uninstall -y faiss faiss-cpu faiss-gpu && pip install --no-cache-dir faiss-gpu==1.7.2\n"
        "# Or use conda-forge FAISS-GPU for CUDA 12.x.\n"
    )

def print_env_diag():
    print("[Python]", sys.executable)
    t = torch_cuda_diag()
    f = faiss_gpu_diag()
    print(f"[Torch] {t.get('torch_version')}  CUDA:{t.get('torch_cuda_version')}  "
          f"avail={t.get('cuda_available')}  gpus={t.get('device_count')}  name0={t.get('device_name_0')}")
    print(f"[FAISS] import={f['faiss_import_ok']}  ver={f['faiss_version']}  "
          f"get_num_gpus={f['has_get_num_gpus']}  num_gpus={f['num_gpus']}  "
          f"gpu_api={f['has_gpu_api']}  gpu_res_ok={f['gpu_resources_ok']}")
    if f["error"]:
        print(f"[FAISS] error: {f['error']}")
    if not (f["faiss_import_ok"] and f["has_gpu_api"] and f["gpu_resources_ok"] and (f["num_gpus"] or 0) > 0):
        print("\n[INFO] FAISS-GPU not available in this interpreter.")
        print("Install suggestion:", suggest_install(t.get("torch_cuda_version")))
        if FORCE_FAISS_GPU:
            raise RuntimeError("FORCE_FAISS_GPU=1 set, but FAISS-GPU not available.")

# ========= Helpers =========
def load_all_chunks() -> List[Dict]:
    chunks: List[Dict] = []
    paths = sorted(glob.glob(CHUNKS_GLOB))
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                chunks.append(json.loads(line))
    if not chunks:
        raise RuntimeError(f"No chunks found in {CHUNKS_GLOB}. Run chunk_pdfs.py first.")
    return chunks

def regex_tokenize(s: str) -> List[str]:
    # keeps acronyms, numbers, simple punctuation inside tokens
    return re.findall(r"[A-Za-z0-9][A-Za-z0-9_\-./]+", s.lower())

# ========= FAISS =========
def build_faiss(chunks: List[Dict]) -> None:
    # 0) Diagnostics
    print_env_diag()

    # 1) Device
    try:
        import torch
        on_cuda = torch.cuda.is_available()
    except Exception:
        on_cuda = False
    print("[FAISS] encoding on", "CUDA" if on_cuda else "CPU")

    # 2) Model (future-proof tokenizer behavior)
    model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2",
        device="cuda" if on_cuda else "cpu"
    )
    if hasattr(model, "tokenizer"):
        try:
            # quiets the transformers FutureWarning and locks behavior
            model.tokenizer.clean_up_tokenization_spaces = True
        except Exception:
            pass

    # 3) Create base FAISS index (CPU)
    import faiss
    dim = 384  # all-MiniLM-L6-v2
    cpu_index = faiss.IndexFlatIP(dim) if FAISS_USE_COSINE else faiss.IndexFlatL2(dim)
    index = cpu_index
    gpu_ok = False

    # 4) Try GPU attach
    try:
        ng = faiss.get_num_gpus() if hasattr(faiss, "get_num_gpus") else 0
        if ng > 0 and hasattr(faiss, "StandardGpuResources"):
            print(f"[FAISS] {ng} GPU(s) visible")
            if FAISS_USE_ALL_GPUS and hasattr(faiss, "index_cpu_to_all_gpus"):
                index = faiss.index_cpu_to_all_gpus(cpu_index)
                gpu_ok = True
                print("[FAISS] Using ALL GPUs")
            else:
                res = faiss.StandardGpuResources()
                index = faiss.index_cpu_to_gpu(res, FAISS_GPU_ID, cpu_index)
                gpu_ok = True
                print(f"[FAISS] Using GPU {FAISS_GPU_ID}")
        else:
            print("[FAISS] GPU not found or FAISS not compiled with GPU → CPU")
            if FORCE_FAISS_GPU:
                raise RuntimeError("FORCE_FAISS_GPU=1 set, but FAISS-GPU not available.")
    except Exception as e:
        print("[FAISS] GPU path unavailable → CPU. Reason:", e)
        if FORCE_FAISS_GPU:
            raise

    # 5) Encode & add in streaming batches (low RAM)
    n = len(chunks)
    print(f"[FAISS] indexing {n} chunks; embed_batch={EMBED_BATCH}, infer_bs={EMBED_INFER_BS}")
    start_idx = 0
    while start_idx < n:
        end_idx = min(start_idx + EMBED_BATCH, n)
        texts = [c["text"] for c in chunks[start_idx:end_idx]]
        X = model.encode(
            texts,
            batch_size=EMBED_INFER_BS,
            normalize_embeddings=FAISS_USE_COSINE,
            convert_to_numpy=True,
            show_progress_bar=False
        )
        X = np.ascontiguousarray(X.astype("float32"))
        # If using cosine, vectors are already normalized by encode(); do not re-normalize here.
        # If L2, leave as-is.
        # Add to index in smaller FAISS-add batches
        for a in range(0, X.shape[0], FAISS_ADD_BATCH):
            b = min(a + FAISS_ADD_BATCH, X.shape[0])
            index.add(X[a:b])
        start_idx = end_idx
        print(f"[FAISS] added {end_idx}/{n}")

    # 6) Save CPU index + IDs (same order as added)
    if gpu_ok:
        import faiss as _f
        index = _f.index_gpu_to_cpu(index)
    os.makedirs(os.path.dirname(OUT_FAISS), exist_ok=True)
    with open(OUT_IDS, "wb") as f:
        pickle.dump([c["id"] for c in chunks], f)
    import faiss as _f2
    _f2.write_index(index, OUT_FAISS)
    print(f"[FAISS] saved → {OUT_FAISS} (+ ids: {OUT_IDS})")

# ========= BM25 =========
def build_bm25(chunks: List[Dict]) -> None:
    from rank_bm25 import BM25Okapi
    print("[BM25] tokenizing…")
    tokens = [regex_tokenize(c["text"]) for c in chunks]
    # Build once now (ensures data is valid), we don't persist idf but we persist tokens to rebuild at query time
    _ = BM25Okapi(tokens)
    os.makedirs(os.path.dirname(OUT_BM25), exist_ok=True)
    with open(OUT_BM25, "wb") as f:
        pickle.dump({"ids": [c["id"] for c in chunks], "tokens": tokens}, f)
    print(f"[BM25] saved → {OUT_BM25}  (ids+tokens; rebuild BM25Okapi at query time)")

# ========= DuckDB =========
def build_duckdb(chunks: List[Dict]) -> None:
    os.makedirs(os.path.dirname(OUT_DUCK), exist_ok=True)
    con = duckdb.connect(OUT_DUCK)
    con.execute("""
        CREATE TABLE IF NOT EXISTS chunks(
            id TEXT PRIMARY KEY,
            source TEXT,
            page INTEGER,
            text TEXT
        )
    """)
    data: List[Tuple[str, str, int, str]] = []
    for c in chunks:
        meta = c.get("meta", {}) or {}
        src = meta.get("source", "") or ""
        page = meta.get("page")
        try:
            page = int(page) if page is not None else -1
        except Exception:
            page = -1
        data.append((c["id"], src, page, c["text"]))
    con.executemany("INSERT OR REPLACE INTO chunks VALUES (?,?,?,?)", data)
    con.execute("CREATE INDEX IF NOT EXISTS idx_chunks_source ON chunks(source)")
    con.execute("CREATE INDEX IF NOT EXISTS idx_chunks_page ON chunks(page)")
    con.close()
    print(f"[DuckDB] saved → {OUT_DUCK} rows={len(chunks)}")

# ========= Main =========
if __name__ == "__main__":
    # Optional: determinism for debugging
    os.environ["PYTHONHASHSEED"] = "0"
    try:
        import torch, random
        torch.manual_seed(0); random.seed(0); np.random.seed(0)
    except Exception:
        pass

    chunks = load_all_chunks()
    print(f"Loaded chunks: {len(chunks)}")
    build_faiss(chunks)
    build_bm25(chunks)
    build_duckdb(chunks)
    print("✅ indexing done.")
