"""
PDF extraction and structured field parsing from myscheme.gov.in PDFs.

Extracts scheme name, eligibility, benefits, application process,
and description from 723+ government scheme PDFs.
"""

import re
import json
import random
from pathlib import Path
from typing import Optional

SECTION_PATTERNS = {
    "description": [
        r"(?:Details|About|Overview|Description)[:\s]*\n(.+?)(?=\n[A-Z]|\Z)",
        r"(?:scheme details|what is)[:\s]*(.+?)(?=\n[A-Z]|\Z)",
    ],
    "eligibility": [
        r"(?:Eligibility|Who Can Apply|Eligible)[:\s]*\n(.+?)(?=\n[A-Z]|\Z)",
        r"(?:eligibility criteria)[:\s]*(.+?)(?=\n[A-Z]|\Z)",
    ],
    "benefits": [
        r"(?:Benefits|What You Get|Incentives)[:\s]*\n(.+?)(?=\n[A-Z]|\Z)",
        r"(?:benefit|financial assistance)[:\s]*(.+?)(?=\n[A-Z]|\Z)",
    ],
    "application_process": [
        r"(?:How To Apply|Application Process|Apply)[:\s]*\n(.+?)(?=\n[A-Z]|\Z)",
        r"(?:application procedure)[:\s]*(.+?)(?=\n[A-Z]|\Z)",
    ],
}


def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extract all text from a PDF using PyMuPDF."""
    try:
        import fitz
        doc = fitz.open(str(pdf_path))
        text = "".join(page.get_text() for page in doc)
        doc.close()
        return text.strip()
    except Exception:
        return ""


def parse_scheme(text: str, filename: str) -> dict:
    """Parse structured fields from raw PDF text."""
    result = {"source_file": filename}
    text_lines = text.strip().split('\n')

    for line in text_lines[:5]:
        line = line.strip()
        if len(line) > 5 and not line.startswith("http") and not line.isdigit():
            result["name"] = line
            break

    if "name" not in result:
        result["name"] = Path(filename).stem.replace("-", " ").title()

    for field, patterns in SECTION_PATTERNS.items():
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                value = re.sub(r'\s+', ' ', match.group(1).strip())[:800]
                if len(value) > 30:
                    result[field] = value
                    break

    return result


def extract_all(pdf_dir: str = "./raw_pdfs") -> list[dict]:
    """Extract and parse all PDFs in a directory."""
    from tqdm import tqdm

    pdf_files = list(Path(pdf_dir).rglob("*.pdf"))
    print(f"Found {len(pdf_files)} PDFs")

    schemes = []
    for pdf_path in tqdm(pdf_files, desc="Extracting"):
        text = extract_text_from_pdf(pdf_path)
        if len(text) < 100:
            continue
        parsed = parse_scheme(text, pdf_path.name)
        content_fields = [parsed.get(f) for f in
                          ["description", "eligibility", "benefits", "application_process"]
                          if parsed.get(f)]
        if parsed.get("name") and content_fields:
            schemes.append(parsed)

    print(f"Parsed {len(schemes)} valid schemes")
    return schemes
