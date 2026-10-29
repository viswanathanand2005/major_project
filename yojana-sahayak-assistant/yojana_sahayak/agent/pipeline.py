"""
End-to-end pipeline orchestrator.

    Voice/Text → ASR → Query Rewrite → RAG → LLM → TTS → Voice/Text

Tracks latency at each stage for benchmarking.
"""

import time
from typing import Optional
from dataclasses import dataclass, field


@dataclass
class PipelineResult:
    """Result from a single pipeline invocation with full tracing."""
    query_raw: str = ""
    query_clean: str = ""
    context: str = ""
    answer: str = ""
    tts_audio: str = ""
    asr_latency: float = 0.0
    rag_latency: float = 0.0
    llm_latency: float = 0.0
    tts_latency: float = 0.0
    total_latency: float = 0.0
    asr_skipped: bool = False


class YojanaPipeline:
    """
    Full voice AI pipeline for Indian government scheme queries.

    Supports text-only mode (skip ASR/TTS) for testing and
    voice mode for production deployment.
    """

    def __init__(self):
        self._retriever = None
        self._history: list[dict] = []

    @property
    def retriever(self):
        if self._retriever is None:
            from yojana_sahayak.rag.retriever import SchemeRetriever
            self._retriever = SchemeRetriever()
        return self._retriever

    def run(self, audio_path: str = None, text_input: str = None,
            speak: bool = True) -> PipelineResult:
        """
        Run the full pipeline.

        Args:
            audio_path: Path to audio file (voice mode).
            text_input: Text query (text mode, skips ASR).
            speak: Whether to synthesize and play TTS output.

        Returns:
            PipelineResult with all outputs and latencies.
        """
        result = PipelineResult()
        total_start = time.time()

        # Step 1: ASR
        if text_input:
            result.query_raw = text_input
            result.asr_skipped = True
        else:
            from yojana_sahayak.asr.whisper import transcribe
            print("\n[1/4] ASR — Whisper transcription...")
            t = time.time()
            asr_out = transcribe(audio_path)
            result.asr_latency = time.time() - t
            result.query_raw = asr_out["text"]
            print(f"  Transcript: {result.query_raw}")

        # Step 2: Query rewrite
        from yojana_sahayak.asr.whisper import rewrite_query
        print("\n[2/4] Query rewrite...")
        result.query_clean = rewrite_query(result.query_raw)
        if result.query_clean != result.query_raw:
            print(f"  Corrected: {result.query_clean}")

        # Step 3: RAG + LLM
        print("\n[3/4] RAG retrieval + LLM generation...")
        t = time.time()
        result.context = self.retriever.retrieve_context(result.query_clean)
        result.rag_latency = time.time() - t

        from yojana_sahayak.llm.generator import generate
        t = time.time()
        result.answer = generate(
            result.query_clean,
            context=result.context,
            history=self._history,
        )
        result.llm_latency = time.time() - t
        print(f"  Answer: {result.answer[:200]}{'...' if len(result.answer) > 200 else ''}")

        # Step 4: TTS
        if speak:
            from yojana_sahayak.tts.speaker import synthesize, play
            print("\n[4/4] TTS — generating speech...")
            t = time.time()
            result.tts_audio = synthesize(result.answer)
            result.tts_latency = time.time() - t
            play(result.tts_audio)

        result.total_latency = round(time.time() - total_start, 3)
        print(f"\n  Total latency: {result.total_latency}s")

        # Update conversation history
        self._history.append({
            "user": result.query_clean,
            "assistant": result.answer,
        })
        if len(self._history) > 5:
            self._history = self._history[-5:]

        return result

    def reset_history(self):
        self._history.clear()
