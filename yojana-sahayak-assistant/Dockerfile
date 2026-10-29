# ============================================================================
# Yojana Sahayak — Sovereign Voice Agent
# Supports: Gradio demo (default), MCP server, CLI
# ============================================================================

FROM python:3.11-slim AS base

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libsndfile1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies — core offline stack (no MLX inside Docker; models served externally or via CPU)
COPY requirements.txt .
RUN pip install --no-cache-dir \
    sentence-transformers \
    faiss-cpu \
    numpy \
    python-dotenv \
    soundfile \
    parler-tts \
    transformers \
    gradio \
    "mcp[cli]"

# Application code
COPY yojana_sahayak/ yojana_sahayak/
COPY data/ data/
COPY pyproject.toml .

RUN pip install --no-cache-dir -e . --no-deps

# Pre-download models during build so the container starts fast and runs offline
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')"
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('Subh24ai/yojana-sahayak-qwen2.5-1.5b-merged')"

EXPOSE 7860

# Default: Gradio web demo
CMD ["python", "-m", "yojana_sahayak.cli", "--gradio"]

# Alternative commands:
# MCP server:   docker run --env-file .env yojana-sahayak python -m yojana_sahayak.mcp.server
# Text query:   docker run --env-file .env yojana-sahayak python -m yojana_sahayak.cli --text "PM Kisan"
