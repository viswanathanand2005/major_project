"""
Retrieval-Augmented Generation over Indian government scheme data.

Uses FAISS (flat inner product) with multilingual MiniLM embeddings.
Indexes ~591 unique scheme facts from curated + filtered training data.
Runs fully offline once the embedding model is cached.
"""

import json
from pathlib import Path
from typing import Optional

import numpy as np

from yojana_sahayak.config import (
    EMBEDDING_MODEL, RAG_TOP_K, RAG_MIN_SCORE, RAG_MIN_SCORE_NAMED,
    TRAIN_CLEAN_PATH, CORE_SCHEMES_PATH, SCHEME_ALIASES, NOISE_MARKERS,
)

_NOISE_MARKERS = NOISE_MARKERS


class SchemeRetriever:
    """
    FAISS-backed retriever for Indian government scheme Q&A.

    Deduplicates by (scheme_name, field) key and filters out
    web-scraping artifacts before indexing.
    """

    def __init__(self, train_path: str = TRAIN_CLEAN_PATH,
                 core_path: str = CORE_SCHEMES_PATH):
        self._index = None
        self._docs = None
        self._encoder = None
        self._train_path = train_path
        self._core_path = core_path

    @property
    def is_loaded(self) -> bool:
        return self._index is not None

    @property
    def doc_count(self) -> int:
        return len(self._docs) if self._docs else 0

    @property
    def scheme_names(self) -> list[str]:
        """Sorted list of unique scheme names currently in the index."""
        if not self._docs:
            return []
        return sorted(set(d["scheme"] for d in self._docs))

    def build_index(self) -> None:
        """Build FAISS index from curated + filtered training data."""
        if self._index is not None:
            return

        from sentence_transformers import SentenceTransformer
        import faiss

        print("Building RAG index...")
        self._encoder = SentenceTransformer(EMBEDDING_MODEL)

        docs, seen = [], set()

        core = Path(self._core_path)
        if core.exists():
            self._load_jsonl(str(core), docs, seen, require_clean=False)

        train = Path(self._train_path)
        if train.exists():
            self._load_jsonl(str(train), docs, seen, require_clean=True)

        print(f"  Indexing {len(docs)} scheme facts...")
        texts = [f"{d['scheme']} {d['field']}: {d['q']}" for d in docs]
        embeddings = self._encoder.encode(texts, batch_size=256, show_progress_bar=True)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        dim = embeddings.shape[1]
        self._index = faiss.IndexFlatIP(dim)
        self._index.add(embeddings.astype(np.float32))
        self._docs = docs
        print(f"  RAG index ready: {len(docs)} docs, dim={dim}")

    def retrieve(self, query: str, top_k: int = RAG_TOP_K) -> list[dict]:
        """
        Retrieve relevant scheme facts for a query.

        Strategy:
        - Named-scheme query (alias detected): search full index, then
          filter results to docs belonging to that scheme.
        - Generic/Hindi query: search with original + scheme-name anchor
          and accept scores above the general threshold.

        Returns list of dicts with 'scheme', 'field', 'answer', 'score'.
        """
        self.build_index()

        expanded, matched_scheme = self._expand_query(query)
        threshold = RAG_MIN_SCORE_NAMED if matched_scheme else RAG_MIN_SCORE

        # Search with a wider k so filtering has candidates to work with
        search_k = max(top_k * 10, 20)
        q_emb = self._encoder.encode([expanded])
        q_emb = q_emb / np.linalg.norm(q_emb, axis=1, keepdims=True)
        scores, indices = self._index.search(q_emb.astype(np.float32), search_k)

        seen_keys: set = set()
        results: list[dict] = []

        for score, idx in zip(scores[0], indices[0]):
            if score < threshold:
                continue
            d = self._docs[idx]

            # For named-scheme queries, only keep docs from that scheme
            if matched_scheme and matched_scheme.lower() not in d["scheme"].lower():
                continue

            key = (d["scheme"], d["field"])
            if key in seen_keys:
                continue
            seen_keys.add(key)
            results.append({
                "scheme": d["scheme"],
                "field": d["field"],
                "answer": d["a"],
                "score": round(float(score), 4),
            })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def retrieve_context(self, query: str, top_k: int = RAG_TOP_K) -> str:
        """Retrieve and format as a context string for LLM prompting."""
        results = self.retrieve(query, top_k)
        parts = [f"Scheme: {r['scheme']}\n{r['answer']}" for r in results]
        return "\n\n".join(parts)

    def _expand_query(self, query: str) -> tuple[str, str]:
        """Expand query with full scheme name when a known alias is detected.

        Handles Roman (case-insensitive), Devanagari, and Hinglish aliases.
        Returns (expanded_query, matched_scheme_name_or_empty).
        """
        q_lower = query.lower()
        for alias, full_name in SCHEME_ALIASES.items():
            if alias.isascii() and alias in q_lower:
                return f"{full_name} {query}", full_name
            elif not alias.isascii() and alias in query:
                return f"{full_name} {query}", full_name
        return query, ""

    @staticmethod
    def _is_clean(text: str) -> bool:
        return not any(m in text for m in _NOISE_MARKERS)

    def _load_jsonl(self, path: str, docs: list, seen: set,
                    require_clean: bool = True) -> None:
        with open(path, encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                msgs = rec.get("messages", [])
                key = (rec["scheme_name"], rec["field"])
                if key in seen:
                    continue
                seen.add(key)

                assistant_msg = next(
                    (m["content"] for m in msgs if m["role"] == "assistant"), None
                )
                user_msg = next(
                    (m["content"] for m in msgs if m["role"] == "user"), None
                )
                if not assistant_msg or not user_msg:
                    continue
                if require_clean and (
                    not self._is_clean(assistant_msg) or not self._is_clean(user_msg)
                ):
                    continue

                content = assistant_msg.replace(rec["scheme_name"], "").strip()
                if len(content) < 50:
                    continue

                docs.append({
                    "scheme": rec["scheme_name"],
                    "field": rec["field"],
                    "q": user_msg,
                    "a": assistant_msg,
                })
