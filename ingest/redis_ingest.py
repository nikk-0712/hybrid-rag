import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import os, glob, json
from typing import List, Dict
from stores.redis_store import RedisStore

def load_jsonl_chunks(pattern: str) -> List[Dict]:
    chunks = []
    for path in glob.glob(pattern):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                chunks.append(json.loads(line))
    return chunks

def main():
    pattern = os.environ.get("CHUNKS_GLOB", "Data/output/*.chunks.jsonl")
    store = RedisStore()
    print("[Redis] Creating index (if not exists)...")
    store.create_index()
    print(f"[Load] Reading chunks from {pattern} ...")
    chunks = load_jsonl_chunks(pattern)
    print(f"[Load] {len(chunks)} chunks")
    if not chunks:
        print("No chunks found. Run chunk_pdfs.py first.")
        return
    print("[Ingest] Embedding + HSET -> Redis ...")
    n = store.upsert_chunks(chunks, batch_size=256)
    print(f"[Done] Upserted {n} chunks into Redis index.")

if __name__ == "__main__":
    main()
