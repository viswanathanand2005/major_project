"""
Tests for Yojana Sahayak MCP tools and core components.

Run: pytest tests/ -v
"""

import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from yojana_sahayak.config import SCHEME_ALIASES, ASR_CORRECTIONS


# ── Config Tests ──────────────────────────────────────────────────────────────

class TestConfig:
    def test_scheme_aliases_not_empty(self):
        assert len(SCHEME_ALIASES) > 0

    def test_scheme_aliases_have_full_names(self):
        for alias, full_name in SCHEME_ALIASES.items():
            assert len(full_name) > len(alias)

    def test_asr_corrections_have_pairs(self):
        assert len(ASR_CORRECTIONS) > 0
        for wrong, correct in ASR_CORRECTIONS.items():
            assert wrong != correct


# ── ASR Rewrite Tests ─────────────────────────────────────────────────────────

class TestASRRewrite:
    def test_devanagari_corrections(self):
        from yojana_sahayak.asr.whisper import rewrite_query
        assert "आयुष्मान" in rewrite_query("आइसमान भारत")
        assert "पीएम" in rewrite_query("पी-एम किसान")
        assert "उज्ज्वला" in rewrite_query("उज्जुला योजना")

    def test_english_corrections(self):
        from yojana_sahayak.asr.whisper import rewrite_query
        assert "PM Kisan" in rewrite_query("pm kisaan scheme")
        assert "Ayushman" in rewrite_query("ayushmann bharat")

    def test_no_false_corrections(self):
        from yojana_sahayak.asr.whisper import rewrite_query
        original = "मुद्रा लोन के लिए क्या डॉक्यूमेंट्स चाहिए"
        assert rewrite_query(original) == original


# ── MCP Tool Tests ────────────────────────────────────────────────────────────

class TestMCPTools:
    def test_list_schemes_returns_dict(self):
        """list_schemes should return a dict with count and schemes keys."""
        from yojana_sahayak.mcp.server import list_schemes
        # This will try to build the index — may fail without data
        # We test the interface contract
        try:
            result = list_schemes()
            assert "count" in result
            assert "schemes" in result
            assert isinstance(result["schemes"], list)
        except FileNotFoundError:
            pytest.skip("Data files not available")

    def test_get_scheme_details_missing(self):
        """get_scheme_details returns error for unknown schemes."""
        from yojana_sahayak.mcp.server import get_scheme_details, _load_core_schemes
        # Mock to avoid file dependency
        with patch("yojana_sahayak.mcp.server._load_core_schemes", return_value={}):
            with patch("yojana_sahayak.mcp.server.search_schemes", return_value=[]):
                result = get_scheme_details("Nonexistent Scheme XYZ")
                assert "error" in result

    def test_search_schemes_interface(self):
        """search_schemes should accept query and top_k."""
        from yojana_sahayak.mcp.server import search_schemes
        # Test the interface (may skip if no data)
        try:
            results = search_schemes("PM Kisan eligibility", top_k=1)
            assert isinstance(results, list)
        except FileNotFoundError:
            pytest.skip("Data files not available")


# ── Data Quality Tests ────────────────────────────────────────────────────────

class TestDataQuality:
    @pytest.fixture
    def core_schemes_path(self):
        path = Path(__file__).parent.parent / "data" / "core_schemes.jsonl"
        if not path.exists():
            pytest.skip("core_schemes.jsonl not available")
        return path

    def test_core_schemes_valid_jsonl(self, core_schemes_path):
        """Every line in core_schemes.jsonl should be valid JSON."""
        with open(core_schemes_path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                rec = json.loads(line)  # raises on invalid JSON
                assert "scheme_name" in rec, f"Line {i}: missing scheme_name"
                assert "messages" in rec, f"Line {i}: missing messages"

    def test_core_schemes_have_clean_names(self, core_schemes_path):
        """Scheme names should not contain UI navigation artifacts."""
        noise = ["Are you sure", "CancelSign", "Sign Out"]
        with open(core_schemes_path, encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                for marker in noise:
                    assert marker not in rec["scheme_name"], \
                        f"Junk in scheme name: {rec['scheme_name'][:60]}"

    def test_core_schemes_have_content(self, core_schemes_path):
        """Assistant messages should have meaningful content (>20 chars)."""
        with open(core_schemes_path, encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                for msg in rec["messages"]:
                    if msg["role"] == "assistant":
                        assert len(msg["content"]) > 20, \
                            f"Empty answer for {rec['scheme_name']}"


# ── TTS Tests ─────────────────────────────────────────────────────────────────

class TestTTS:
    def test_language_detection_hindi(self):
        from yojana_sahayak.tts.speaker import detect_language
        assert detect_language("पीएम किसान के लिए कौन एलिजिबल है") == "hi"

    def test_language_detection_english(self):
        from yojana_sahayak.tts.speaker import detect_language
        assert detect_language("Who is eligible for PM Kisan?") == "en"

    def test_language_detection_hinglish(self):
        from yojana_sahayak.tts.speaker import detect_language
        # Hinglish with some Devanagari should be detected as Hindi
        result = detect_language("PM Kisan ke liye kaun eligible hai?")
        # This is Roman Hinglish, should detect as English
        assert result == "en"
