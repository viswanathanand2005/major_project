"""
Text-to-Speech using macOS system voices (fully offline, no download needed).

Hindi (Devanagari) → Lekha  (hi_IN, Apple Neural voice)
English / Hinglish  → Rishi  (en_IN, Indian-accented English)

Both voices ship with macOS and run instantly on Apple Silicon.
"""

import re
import subprocess
import tempfile
from typing import Optional


# macOS voice names  (install via System Settings → Accessibility → Spoken Content if missing)
_VOICE_HINDI   = "Lekha"   # hi-IN female
_VOICE_ENGLISH = "Rishi"   # en-IN male, handles Hinglish well


def detect_language(text: str) -> str:
    """Return 'hi' if text is primarily Devanagari, else 'en'."""
    devanagari = len(re.findall(r'[ऀ-ॿ]', text))
    total_chars = max(len(re.findall(r'[a-zA-Zऀ-ॿ]', text)), 1)
    return "hi" if devanagari / total_chars > 0.3 else "en"


def synthesize(text: str, lang: Optional[str] = None,
               output_path: Optional[str] = None) -> str:
    """
    Convert text to speech using macOS `say` command (offline).

    Returns path to the generated AIFF/WAV file.
    """
    if lang is None:
        lang = detect_language(text)

    voice = _VOICE_HINDI if lang == "hi" else _VOICE_ENGLISH

    if output_path is None:
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        output_path = tmp.name

    subprocess.run(
        ["say", "-v", voice, "--data-format=LEF32@22050", "-o", output_path, text],
        check=True,
        capture_output=True,
    )
    return output_path


def play(audio_path: str) -> None:
    """Play an audio file through the default speakers."""
    subprocess.run(["afplay", audio_path], check=False)
