"""Centralized configuration for all pipeline components."""

import os
import warnings as _warnings
from pathlib import Path

# Load .env if present (no-op when python-dotenv is absent)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# ── HuggingFace ───────────────────────────────────────────────────────────────
HF_USERNAME = "Subh24ai"
HF_DATASET = f"{HF_USERNAME}/yojana-sahayak-instruct"
HF_MODEL_MERGED = f"{HF_USERNAME}/yojana-sahayak-qwen2.5-1.5b-merged"
HF_MODEL_QLORA = f"{HF_USERNAME}/yojana-sahayak-qwen2.5-1.5b-qlora"
BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

# MLX model for inference — 3-tier priority:
#   1. mlx-yojana/  (local 4-bit, fully offline, fastest)
#   2. HF merged model (first-run download ~1 GB)
#   3. Base model fallback (no fine-tuning — quality degrades noticeably)
_LOCAL_MLX = str(PROJECT_ROOT / "mlx-yojana")

if os.path.isdir(_LOCAL_MLX):
    MLX_MODEL = _LOCAL_MLX
else:
    MLX_MODEL = f"{HF_USERNAME}/yojana-sahayak-qwen2.5-1.5b-merged"
    _warnings.warn(
        f"Local MLX model not found at '{_LOCAL_MLX}'. "
        f"Using HuggingFace model '{MLX_MODEL}' (requires first-run download ~1 GB). "
        f"For fully offline use, run: python scripts/setup_mlx_model.py",
        stacklevel=2,
    )

# ── ASR ───────────────────────────────────────────────────────────────────────
WHISPER_MODEL = "mlx-community/whisper-large-v3-turbo"
SAMPLE_RATE = 16000
RECORD_DURATION_SEC = 6

# ── RAG ───────────────────────────────────────────────────────────────────────
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
RAG_TOP_K = 3
RAG_MIN_SCORE = 0.40        # multilingual MiniLM scores low cross-script (Hindi↔Hinglish)
RAG_MIN_SCORE_NAMED = 0.35  # even lower for exact scheme name matches

TRAIN_CLEAN_PATH = str(DATA_DIR / "train_clean.jsonl")
CORE_SCHEMES_PATH = str(DATA_DIR / "core_schemes.jsonl")

# ── LLM ───────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are Yojana Sahayak, a helpful assistant for Indian government welfare schemes. "
    "Rules:\n"
    "1. Answer ONLY using the Reference provided. Do not add information not in the Reference.\n"
    "2. Be direct and brief — 2 to 4 sentences maximum. Stop immediately after answering.\n"
    "3. Reply in the same language the user used (Hindi, English, or Hinglish).\n"
    "4. If the Reference does not contain the answer, say exactly: "
    "'Mujhe is baare mein specific jankari nahi hai.' Do NOT guess or make up information.\n"
    "5. Never output lists, bullet points, or step numbers. Plain sentences only."
)
MAX_HISTORY_TURNS = 5
LLM_MAX_TOKENS = 120
LLM_TEMPERATURE = 0.0

# ── Scheme Aliases (for query expansion) ──────────────────────────────────────
SCHEME_ALIASES = {
    # Roman / Hinglish
    "pm kisan":     "Pradhan Mantri Kisan Samman Nidhi",
    "pm awas":      "PM Awas Yojana",
    "ayushman":     "Ayushman Bharat PM-JAY",
    "ujjwala":      "PM Ujjwala Yojana",
    "mudra":        "Pradhan Mantri Mudra Yojana",
    "mnrega":       "Mahatma Gandhi National Rural Employment Guarantee",
    "mgnrega":      "Mahatma Gandhi National Rural Employment Guarantee",
    "sukanya":      "Sukanya Samriddhi Yojana",
    "atal pension": "Atal Pension Yojana",
    "pmkvy":        "Pradhan Mantri Kaushal Vikas Yojana",
    # Devanagari — so Hindi queries trigger the named-scheme search path
    "पीएम किसान":   "Pradhan Mantri Kisan Samman Nidhi",
    "प्रधान मंत्री किसान": "Pradhan Mantri Kisan Samman Nidhi",
    "आयुष्मान":     "Ayushman Bharat PM-JAY",
    "उज्ज्वला":     "PM Ujjwala Yojana",
    "मुद्रा":        "Pradhan Mantri Mudra Yojana",
    "मनरेगा":       "Mahatma Gandhi National Rural Employment Guarantee",
    "सुकन्या":      "Sukanya Samriddhi Yojana",
    "अटल पेंशन":   "Atal Pension Yojana",
    "पीएम आवास":   "PM Awas Yojana",
}

# ── LLM output noise markers (web-scraping artifacts the model may hallucinate) ─
NOISE_MARKERS = (
    "sEligibility", "Application ProcessDocuments Required",
    "Frequently Asked Questions", "Sources And References",
    "Something went wrong", "Was this helpful",
    "Are you sure", "sign out", "CancelSign", "Sign OutEng",
    "NowDocuments Required", "\xef\xbb\xbf",
    # Hallucination transition phrases
    "इस प्रकार,", "Is prakar,", "Hamaare eligibility",
    "hamaare eligibility", "Additional eligibility",
    # Fine-tune noise artifacts observed in model output
    "उप-संस्करण", "यहाँ दिखाया गया", "उन्हें देखना और चुनना",
    "विश्वसनीयताओं", "उप-सेवान", "फॉलो करना और उन्हें",
    "कार्यक्रम: और", "असलेले", "इन दो प्राचीन", "प्राचीन व्यक्ति",
    "इन दो", "अनुसार इन",
)

# ── ASR Corrections ───────────────────────────────────────────────────────────
ASR_CORRECTIONS = {
    # Devanagari scheme name errors
    "आइसमान":      "आयुष्मान",
    "आइश्मान":     "आयुष्मान",
    "पीम":          "पीएम",
    "पी-म":         "पीएम",
    "पी-एम":        "पीएम",
    "पी एम":        "पीएम",
    "प्यांकिसान":   "पीएम किसान",
    "प्यांकिसन":    "पीएम किसान",
    # Common word misheard errors
    "इलेजवल":      "एलिजिबल",
    "फिरीक":        "फ्री",
    "ऐस कनेक्शन":  "गैस कनेक्शन",
    "ऐस":           "गैस",
    "निलेज":        "जानकारी",
    "बलाएं":        "बताएं",
    "अपलाई":       "अप्लाई",
    "डाकुमेंट्स":  "डॉक्यूमेंट्स",
    # Scheme name errors
    "उज्जुला":     "उज्ज्वला",
    # Roman script errors
    "pm kisaan":    "PM Kisan",
    "ayushmann":    "Ayushman",
    "ujwala":       "Ujjwala",
    "mudhra":       "Mudra",

    # ── NEW: Common Hindi word errors observed from Whisper ──
    "किसानी":       "किसान",
    "किसन":         "किसान",
    "होजना":        "योजना",
    "योजाना":       "योजना",
    "योजन":         "योजना",
    "यौजना":        "योजना",
    "जोजना":        "योजना",
    "लिजवल":        "एलिजिबल",
    "एलिजबल":       "एलिजिबल",
    "इलिजिबल":      "एलिजिबल",
    "ईलिजिबल":      "एलिजिबल",
    "एलिजबिल":      "एलिजिबल",
    "ओने":           "कौन",
    "काऊन":          "कौन",
    "कोन":           "कौन",

    # ── NEW: Scheme name Devanagari errors ──
    "आवस":           "आवास",
    "आवाज":          "आवास",
    "मुद्रां":        "मुद्रा",
    "सुक्न्या":      "सुकन्या",
    "कौशल":          "कौशल",
    "प्रधान मंत्री": "पीएम",

    # ── NEW: Common Hindi verb/word errors ──
    "बतएं":          "बताएं",
    "बतये":          "बताइए",
    "बतइए":          "बताइए",
    "मिलगा":         "मिलेगा",
    "मिलेगी":        "मिलेगा",
    "केसे":          "कैसे",
    "केस":           "कैसे",
    "किसी":          "किसे",
    "आवदन":          "आवेदन",
    "आबेदन":         "आवेदन",
    "फायदा":         "फायदा",
    "दस्तवेज़":      "डॉक्यूमेंट्स",
    "जरूरी":         "ज़रूरी",
    "पात्रता":       "पात्रता",
    "प्रात्रता":     "पात्रता",

    # ── NEW: Roman/English scheme errors ──
    "pm kissan":     "PM Kisan",
    "pm kisson":     "PM Kisan",
    "ayushman":      "Ayushman",
    "ayusman":       "Ayushman",
    "aayushman":     "Ayushman",
    "ujjawala":      "Ujjwala",
    "ujwalla":       "Ujjwala",
    "mudraa":        "Mudra",
    "sukannya":      "Sukanya",
    "elegible":      "eligible",
    "elligible":     "eligible",
    "yojana":        "Yojana",
    "yojna":         "Yojana",
}
