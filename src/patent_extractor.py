import re
from pathlib import Path
from pypdf import PdfReader

HEADER_RE = re.compile(r'^\s*(?:\(\d+\)\s+)?JP\s+\d{4}-\d+\s+[A-Z]\s+\d{4}\.\d{2}\.\d{2}\s*$')
LINENO_RE = re.compile(r'^\s*\d+\s*$')
FIGONLY_RE = re.compile(r'^\s*【図\S+】\s*$')

def clean_text(raw: str) -> str:
    cleaned = []
    for line in raw.splitlines():
        if HEADER_RE.match(line) or LINENO_RE.match(line) or FIGONLY_RE.match(line):
            continue
        cleaned.append(line)
    return re.sub(r'\n{3,}', '\n\n', '\n'.join(cleaned)).strip()

def extract_text_from_pdf(pdf_path: Path) -> str:
    reader = PdfReader(str(pdf_path))
    raw_text = "\n".join((page.extract_text() or "") for page in reader.pages)
    cleaned = clean_text(raw_text)
    print("    [Done] テキスト抽出")
    return cleaned