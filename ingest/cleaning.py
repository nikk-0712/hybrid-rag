# ingest/cleaning.py
import os, re, io, csv, unicodedata
from dataclasses import dataclass
from typing import List
import fitz  # PyMuPDF
from PIL import Image
import pytesseract

COMMON_LIGATURES = {
    "\ufb00": "ff", "\ufb01": "fi", "\ufb02": "fl", "\ufb03": "ffi", "\ufb04": "ffl",
    "ﬁ": "fi", "ﬂ": "fl", "ﬀ": "ff", "ﬃ": "ffi", "ﬄ": "ffl"
}

def fix_ligatures(s: str) -> str:
    if not s: return s
    for k,v in COMMON_LIGATURES.items(): s = s.replace(k, v)
    return unicodedata.normalize("NFKC", s)

def dehyphenate_lines(s: str) -> str:
    return re.sub(r"(\w+)-\n(\w+)", r"\1\2\n", s)

def clean_page_text(raw_text: str) -> str:
    s = (raw_text or "").replace("\x0c", " ")
    s = fix_ligatures(s)
    s = dehyphenate_lines(s)
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

def page_has_text(page) -> bool:
    txt = page.get_text("text") or ""
    return len(txt.strip()) > 40

def ocr_page(page, dpi=300, langs="eng") -> str:
    zoom = dpi / 72.0
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
    img = Image.open(io.BytesIO(pix.tobytes("png")))
    return pytesseract.image_to_string(img, lang=langs)

def strip_page_numbers_headers_footers(pages_text: List[str], freq_thresh: float = 0.6) -> List[str]:
    n = len(pages_text)
    if n == 0: return pages_text
    top_lines, bot_lines = [], []
    for t in pages_text:
        ls = [x.strip() for x in t.splitlines() if x.strip()]
        if ls:
            top_lines.append(ls[0]); 
            if len(ls) > 1: top_lines.append(ls[1])
            bot_lines.append(ls[-1]); 
            if len(ls) > 1: bot_lines.append(ls[-2])
    from collections import Counter
    top_c, bot_c = Counter(top_lines), Counter(bot_lines)
    def is_common(line, c): return c[line] >= int(freq_thresh * n)
    cleaned = []
    for t in pages_text:
        ls = t.splitlines(); out = []
        for i, line in enumerate(ls):
            l = line.strip()
            if (i <= 1 and is_common(l, top_c)) or (i >= len(ls)-2 and is_common(l, bot_c)):
                continue
            if re.fullmatch(r"(?:page\s*)?\d{1,4}(?:/\d{1,4})?", l, flags=re.I):
                continue
            out.append(line)
        cleaned.append("\n".join(out))
    return cleaned

# ---------- reports ----------
@dataclass
class PageCleanStat:
    page: int; raw_chars: int; cleaned_chars: int; removed_chars: int; removed_pct: float

def make_cleaning_report(raw_pages, stripped_pages):
    stats = []
    for i, (raw, clean) in enumerate(zip(raw_pages, stripped_pages), start=1):
        rc, cc = len(raw or ""), len(clean or "")
        rem = max(rc - cc, 0); pct = (rem/rc*100.0) if rc else 0.0
        stats.append(PageCleanStat(i, rc, cc, rem, round(pct,2)))
    return stats

def write_cleaning_report(out_dir, base, stats, raw_pages, stripped_pages):
    os.makedirs(out_dir, exist_ok=True)
    # CSV
    with open(os.path.join(out_dir, f"{base}.cleaning_report.csv"), "w", encoding="utf-8", newline="") as f:
        import csv
        w = csv.writer(f); w.writerow(["page","raw_chars","cleaned_chars","removed_chars","removed_pct"])
        for s in stats: w.writerow([s.page, s.raw_chars, s.cleaned_chars, s.removed_chars, f"{s.removed_pct:.2f}"])
    # Samples
    with open(os.path.join(out_dir, f"{base}.cleaning_samples.txt"), "w", encoding="utf-8") as f:
        f.write("=== BEFORE / AFTER SAMPLES ===\n\n")
        for i in range(min(4, len(raw_pages))):
            f.write(f"[Page {i+1}] BEFORE:\n{(raw_pages[i] or '')[:700]}\n\n")
            f.write(f"[Page {i+1}] AFTER:\n{(stripped_pages[i] or '')[:700]}\n\n")
            f.write("-"*96 + "\n\n")

def extract_pages_cleaned(pdf_path: str, ocr_langs="eng"):
    doc = fitz.open(pdf_path); fname = os.path.basename(pdf_path)
    raw_pages = []
    for page in doc:
        txt = page.get_text("text") or ""
        if not txt.strip(): txt = ocr_page(page, dpi=300, langs=ocr_langs)
        raw_pages.append(txt)
    cleaned = [clean_page_text(t) for t in raw_pages]
    stripped = strip_page_numbers_headers_footers(cleaned, freq_thresh=0.6)
    for i, txt in enumerate(stripped):
        if txt.strip():
            yield {"text": txt, "meta": {"source": fname, "page": i+1}}, raw_pages, stripped
