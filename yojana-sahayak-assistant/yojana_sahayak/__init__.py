"""
Yojana Sahayak — Sovereign Voice Agent for Indian Government Schemes
====================================================================

A fully offline, multilingual voice AI system that helps Indian citizens
discover and understand government welfare schemes in Hindi and English.

Architecture:
    Hindi Voice → Whisper ASR → Query Rewrite → RAG (FAISS) → Fine-tuned Qwen2.5 → TTS → Hindi Voice

Designed for air-gapped, on-premise deployment. Zero internet dependency at runtime.
"""

__version__ = "2.0.0"
__author__ = "Subhash Gupta"
