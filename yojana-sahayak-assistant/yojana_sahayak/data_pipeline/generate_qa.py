"""
Bilingual (English + Hindi) QA pair generation from parsed scheme data.

Generates instruction-tuning pairs in chat format for fine-tuning LLMs.
Covers: description, eligibility, benefits, application_process, multi_turn.
"""

import json
import re
import random
from pathlib import Path
from typing import Optional

SYSTEM_PROMPT = (
    "You are Yojana Sahayak, a helpful assistant that provides accurate information "
    "about Indian government welfare schemes in simple, clear language. "
    "Answer in the same language the user asks in. Be concise and actionable."
)

EN_TEMPLATES = {
    "description": [
        ("What is {name}?", "{name} is a government scheme. {description}"),
        ("Tell me about the {name} scheme.", "{description}"),
        ("Explain {name} in simple words.", "{name}: {description}"),
    ],
    "eligibility": [
        ("Who is eligible for {name}?", "To be eligible for {name}: {eligibility}"),
        ("What are the eligibility criteria for {name}?", "Eligibility for {name}: {eligibility}"),
        ("Can a farmer apply for {name}? What are the conditions?",
         "Here are the eligibility conditions for {name}: {eligibility}"),
    ],
    "benefits": [
        ("What are the benefits of {name}?", "Under {name}, you will receive: {benefits}"),
        ("How much financial support does {name} provide?", "{benefits}"),
        ("What will I get if I apply for {name}?",
         "If eligible, you will receive under {name}: {benefits}"),
    ],
    "application_process": [
        ("How do I apply for {name}?", "To apply for {name}: {application_process}"),
        ("What is the process to register for {name}?", "{application_process}"),
        ("Where can I apply for {name}?",
         "You can apply for {name} as follows: {application_process}"),
    ],
}

HI_TEMPLATES = {
    "description": [
        ("{name} kya hai?", "{name} ek sarkari yojana hai. {description}"),
        ("{name} ke baare mein batao.", "{name}: {description}"),
    ],
    "eligibility": [
        ("{name} ke liye kaun apply kar sakta hai?",
         "{name} ke liye yeh log apply kar sakte hain: {eligibility}"),
        ("{name} mein kaun eligible hai?", "{name} ki eligibility: {eligibility}"),
        ("Kya main {name} ke liye apply kar sakta hoon?",
         "{name} ke liye eligibility criteria: {eligibility}"),
    ],
    "benefits": [
        ("{name} mein kya milta hai?", "{name} ke antargat aapko milega: {benefits}"),
        ("{name} se kitna paisa milega?", "{benefits}"),
        ("{name} ke kya fayde hain?", "{name} ke fayde: {benefits}"),
    ],
    "application_process": [
        ("{name} ke liye apply kaise karein?",
         "{name} mein aavedan karne ka tarika: {application_process}"),
        ("{name} ka registration kahan hoga?", "{application_process}"),
        ("{name} ka form kahan milega aur kaise bharen?",
         "{name} ke liye: {application_process}"),
    ],
}

# Noise markers from web-scraping artifacts
JUNK_PATTERNS = [
    r"Are you sure you want to sign out\?.*",
    r"CancelSign\s*Out.*",
    r"Eng\s*English/.*",
    r"Sign\s*Out.*",
    r"\s{3,}.*",
]


def clean_text(text: str) -> str:
    """Strip UI navigation junk from scraped text."""
    if not text:
        return text
    for pattern in JUNK_PATTERNS:
        text = re.sub(pattern, "", text, flags=re.DOTALL)
    return text.strip()


def generate_pairs(scheme: dict) -> list[dict]:
    """Generate all QA pairs (EN + HI + multi-turn) for a single scheme."""
    pairs = []
    name = scheme.get("name", "this scheme")

    def fmt(tpl: str) -> str:
        return tpl.format(
            name=name,
            description=scheme.get("description", ""),
            eligibility=scheme.get("eligibility", ""),
            benefits=scheme.get("benefits", ""),
            application_process=scheme.get("application_process",
                                           "Please visit the official portal."),
        )

    def make(user: str, assistant: str, lang: str, field: str) -> dict:
        return {
            "language": lang, "scheme_name": name, "field": field,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user},
                {"role": "assistant", "content": assistant},
            ],
        }

    for templates, lang in [(EN_TEMPLATES, "en"), (HI_TEMPLATES, "hi")]:
        for field, tmpls in templates.items():
            if not scheme.get(field):
                continue
            for q_tpl, a_tpl in tmpls:
                pairs.append(make(fmt(q_tpl), fmt(a_tpl), lang, field))

    # Multi-turn: eligibility → benefits
    if scheme.get("eligibility") and scheme.get("benefits"):
        for lang, q1, a1, q2, a2 in [
            ("en", f"Am I eligible for {name}?",
             f"Eligibility for {name}: {scheme['eligibility']}",
             "What benefits will I get if I qualify?",
             f"If eligible, you will receive: {scheme['benefits']}"),
            ("hi", f"Kya main {name} ke liye eligible hoon?",
             f"{name} ki eligibility: {scheme['eligibility']}",
             "Agar eligible hoon toh kya milega?",
             f"Aapko milega: {scheme['benefits']}"),
        ]:
            pairs.append({
                "language": lang, "scheme_name": name, "field": "multi_turn",
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": q1},
                    {"role": "assistant", "content": a1},
                    {"role": "user", "content": q2},
                    {"role": "assistant", "content": a2},
                ],
            })

    return pairs


def build_dataset(schemes: list[dict], output_dir: str = "./data",
                  seed: int = 42) -> tuple[str, str]:
    """Generate full dataset, split 80/20, save as JSONL."""
    all_pairs = []
    for scheme in schemes:
        all_pairs.extend(generate_pairs(scheme))

    random.seed(seed)
    random.shuffle(all_pairs)
    split = int(len(all_pairs) * 0.8)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    train_path = str(out / "train.jsonl")
    eval_path = str(out / "eval.jsonl")

    for path, data in [(train_path, all_pairs[:split]),
                       (eval_path, all_pairs[split:])]:
        with open(path, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Dataset: {len(all_pairs)} total | {split} train | {len(all_pairs)-split} eval")
    return train_path, eval_path
