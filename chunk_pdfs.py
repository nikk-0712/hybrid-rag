import os, io, re, json, argparse
from pathlib import Path
from ingest.cleaning import ( 
    extract_pages_cleaned,
    make_cleaning_report, 
    write_cleaning_report
)
import fitz  # PyMuPDF
from PIL import Image
import pytesseract
import unicodedata
import csv


# ---------------- chunking helpers ----------------
def split_paragraphs(text: str):
    parts = re.split(r"\n\s*\n", text.strip())
    return [p.strip() for p in parts if p.strip()]

def chunk_pages(pages, chunk_size=1200, overlap=150):
    """
    Greedy paragraph packer with tail-overlap carry.
    chunk_size is in approx 'chars'*4 ~= tokens (1 token ≈ 4 chars).
    """
    out = []
    for page in pages:
        src = page["meta"]["source"]
        pg = page["meta"]["page"]
        paras = split_paragraphs(page["text"])
        buf, buf_len_chars, idx = [], 0, 0

        def flush():
            nonlocal buf, buf_len_chars, idx
            if not buf:
                return
            text = "\n\n".join(buf)
            out.append({
                "id": f"{src}-p{pg}-c{idx}",
                "text": text,
                "meta": {"source": src, "page": pg, "chunk": idx},
            })
            # overlap carry (approximate tokens: 1 tok ≈ 4 chars)
            tail = text[-overlap*4:]
            buf, buf_len_chars = ([tail], len(tail))
            idx += 1

        for para in paras:
            if buf_len_chars + len(para) + 2 > chunk_size*4:
                flush()
            buf.append(para)
            buf_len_chars += len(para) + 2
        flush()
    return out

def save_chunks_and_manifest(chunks, out_dir: str, base: str):
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # JSONL (what downstream scripts read)
    jsonl_path = os.path.join(out_dir, f"{base}.chunks.jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as jf:
        for ch in chunks:
            jf.write(json.dumps(ch, ensure_ascii=False) + "\n")

    # Manifest CSV (for visibility/debugging)
    mpath = os.path.join(out_dir, f"{base}.chunks.manifest.csv")
    with open(mpath, "w", encoding="utf-8", newline="") as cf:
        w = csv.writer(cf)
        w.writerow(["chunk_id", "source", "page", "chunk_idx", "n_chars", "n_tokens_est", "preview"])
        for ch in chunks:
            m = ch["meta"]
            preview = ch["text"][:160].replace("\n", " ").strip()
            toks = max(1, len(ch["text"]) // 4)  # rough token estimate
            w.writerow([
                ch["id"], m["source"], m["page"], m.get("chunk", -1),
                len(ch["text"]), toks, preview
            ])
    return jsonl_path, mpath

# ---------------- CLI main ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf-dir", default=r"/home/cemilac/Documents/Hybrid RAG/Data/pdfs")
    ap.add_argument("--out-dir", default=r"/home/cemilac/Documents/Hybrid RAG/Data/output")
    ap.add_argument("--ocr-langs", default="eng")
    ap.add_argument("--chunk-size", type=int, default=1200)
    ap.add_argument("--chunk-overlap", type=int, default=150)
    args = ap.parse_args()

    if not os.path.isdir(args.pdf_dir):
        print(f"[!] PDF directory not found: {args.pdf_dir}")
        return

    pdfs = sorted([f for f in os.listdir(args.pdf_dir) if f.lower().endswith(".pdf")])
    if not pdfs:
        print(f"No PDFs found in {args.pdf_dir}")
        return

    total_chunks = 0
    for fname in pdfs:
        path = os.path.join(args.pdf_dir, fname)
        base = os.path.splitext(fname)[0]
        print(f"→ Processing {fname}")

        # 1) extract cleaned pages + keep raw arrays for report
        pages = []; raw_pages = stripped_pages = None
        for rec, raw_all, clean_all in extract_pages_cleaned(path, ocr_langs=args.ocr_langs):
            pages.append(rec); raw_pages, stripped_pages = raw_all, clean_all

        # 2) write cleaning report (new)
        stats = make_cleaning_report(raw_pages or [], stripped_pages or [])
        write_cleaning_report(args.out_dir, base, stats, raw_pages or [], stripped_pages or [])

        # 3) chunk (your existing chunker)
        chunks = chunk_pages(pages, args.chunk_size, args.chunk_overlap)

        # 4) save chunks + manifest (new)
        jpath, mpath = save_chunks_and_manifest(chunks, args.out_dir, base)
        print(f"   pages={len(pages)} chunks={len(chunks)} saved={jpath}")
        print(f"   cleaning_report={os.path.join(args.out_dir, base+'.cleaning_report.csv')}")
        print(f"   manifest={mpath}")
        total_chunks += len(chunks)

    print(f"\n✅ Done. Total chunks: {total_chunks}  (out: {args.out_dir})")

if __name__ == "__main__":
    main()
    