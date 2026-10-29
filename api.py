"""
CultivAI API Server
===================
FastAPI server that exposes the CultivAI specialist agents as HTTP endpoints.
Acts as a lightweight Coordinator — routes requests to the right agent(s),
then returns a unified JSON response.

Endpoints
---------
GET  /health          — health check for all registered agents
POST /disease         — plant disease analysis from an uploaded crop image
POST /schemes         — government scheme discovery from a text query
POST /query           — smart routing: picks agent(s) based on input content

Run
---
    cd major_project
    uvicorn api:app --reload --port 8000

Test (curl examples in README_API.md)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

# ── sys.path: make cultivai/, plant_disease_detection/, and yojana_sahayak/ importable
_ROOT = Path(__file__).parent.resolve()
_YOJANA_ROOT = _ROOT / "yojana-sahayak-assistant"
_DISEASE_ROOT = _ROOT / "plant_disease_detection"
for _p in [str(_ROOT), str(_YOJANA_ROOT), str(_DISEASE_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="CultivAI API",
    description=(
        "Multi-agent farm decision-support system for Indian farmers. "
        "Routes queries to the Plant Disease Agent and/or the Government "
        "Scheme Agent and returns a unified response."
    ),
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Lazy agent singletons ─────────────────────────────────────────────────────
# Agents are loaded on first use so the server starts instantly even if
# a heavy dependency (torch, sentence-transformers) takes a moment to import.

_disease_agent = None
_scheme_agent = None


def _get_disease_agent():
    global _disease_agent
    if _disease_agent is None:
        from plant_disease_detection.cultivai_agent import PlantDiseaseAgent
        _disease_agent = PlantDiseaseAgent()
    return _disease_agent


def _get_scheme_agent():
    global _scheme_agent
    if _scheme_agent is None:
        from yojana_sahayak.agent.cultivai_agent import SchemeAgent
        _scheme_agent = SchemeAgent()
    return _scheme_agent


# ── Keyword routing table (mirrors the planned Coordinator router) ─────────────
_SCHEME_KEYWORDS = {
    "scheme", "yojana", "government", "eligible", "eligibility",
    "benefit", "benefits", "subsidy", "loan", "grant", "assistance",
    "apply", "application", "documents", "kisan", "ayushman", "ujjwala",
    "mudra", "mnrega", "mgnrega", "sukanya", "awas", "insurance", "bima",
    "pension", "pmkvy", "pm kisan",
    # Hindi / Hinglish
    "योजना", "सरकार", "पात्रता", "लाभ", "किसान", "आवेदन", "सब्सिडी",
}

_DISEASE_KEYWORDS = {
    "disease", "blight", "spot", "mold", "rot", "wilt", "rust",
    "mosaic", "virus", "bacterial", "fungal", "leaf", "leaves",
    "yellowing", "brown", "symptoms", "infected", "infection",
    "spray", "treatment", "diagnos",
}


def _route_query(query: str, has_image: bool) -> list[str]:
    """
    Return a list of agent names to invoke based on query and image presence.
    Mirrors the IntentRouter logic from coordinator/router.py.
    """
    q = query.lower()
    agents = []

    if has_image:
        agents.append("plant_disease")

    if any(kw in q for kw in _DISEASE_KEYWORDS) and "plant_disease" not in agents:
        agents.append("plant_disease")

    if any(kw in q for kw in _SCHEME_KEYWORDS):
        agents.append("government_scheme")

    # Fallback: nothing matched → try scheme agent (most conversational)
    if not agents:
        agents = ["government_scheme"]

    return agents


# ── Helpers ───────────────────────────────────────────────────────────────────

def _output_to_response(output, extra: dict = None) -> dict:
    """Convert an AgentOutput to a clean API response dict."""
    d = output.to_dict()
    if extra:
        d.update(extra)
    return d


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health", summary="Health check for all registered agents")
def health():
    """
    Returns the registration and readiness status of every specialist agent.
    Call this first to confirm the server and its dependencies are working.
    """
    report = {}

    for label, loader in [
        ("plant_disease", _get_disease_agent),
        ("government_scheme", _get_scheme_agent),
    ]:
        try:
            agent = loader()
            healthy = agent.health_check()
            report[label] = {"registered": True, "healthy": healthy}
        except ImportError as exc:
            report[label] = {
                "registered": False,
                "healthy": False,
                "error": f"Missing dependency: {exc}",
            }
        except Exception as exc:
            report[label] = {
                "registered": False,
                "healthy": False,
                "error": str(exc),
            }

    all_healthy = all(v.get("healthy") for v in report.values())
    return {"status": "ok" if all_healthy else "degraded", "agents": report}


@app.post(
    "/disease",
    summary="Plant disease analysis from a crop leaf image",
    response_description="AgentOutput from the PlantDiseaseAgent",
)
async def disease_endpoint(
    file: UploadFile = File(..., description="Crop leaf image (JPEG or PNG)"),
    crop: Optional[str] = Form(None, description="Crop type, e.g. 'tomato'"),
    top_k: int = Form(3, description="Number of top predictions to return"),
):
    """
    Upload a crop leaf image and receive a disease classification.

    - **file**: The image to analyse (JPEG or PNG).
    - **crop**: Optional — the farmer's stated crop. Used to detect crop
      mismatches between the stated crop and the model's prediction.
    - **top_k**: How many top predictions to include (default 3).

    Returns an `AgentOutput` JSON with `status`, `summary`, `data.predictions`,
    `sources`, and `limitations`.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=415,
            detail="File must be an image (JPEG, PNG, etc.).",
        )

    image_bytes = await file.read()

    try:
        agent = _get_disease_agent()
    except ImportError as exc:
        raise HTTPException(
            status_code=503,
            detail=(
                f"Plant Disease Agent is not available: {exc}. "
                "Ensure tensorflow and pillow are installed."
            ),
        )

    result = agent.run(
        image_bytes=image_bytes,
        crop_context=crop,
        top_k=top_k,
    )
    return JSONResponse(content=result.to_dict())


@app.post(
    "/schemes",
    summary="Government scheme discovery from a text query",
    response_description="AgentOutput from the SchemeAgent",
)
async def schemes_endpoint(
    query: str = Form(..., description="Your question in Hindi, English, or Hinglish"),
    location: Optional[str] = Form(None, description="State or district, e.g. 'Uttar Pradesh'"),
    crop: Optional[str] = Form(None, description="Crop being grown, e.g. 'wheat'"),
    farmer_context: Optional[str] = Form(
        None,
        description="Additional profile: land size, category, income, etc.",
    ),
    language: str = Form("English", description="Preferred response language (e.g. 'English', 'Hindi', 'Telugu', 'Tamil')"),
    top_k: int = Form(3, description="Number of RAG results to retrieve"),
):
    """
    Ask a question about Indian government welfare schemes.

    - **query**: Your question (Hindi / English / Hinglish).  
      Examples: `'PM Kisan ke liye kaun eligible hai?'` or
      `'What are the benefits of Ayushman Bharat?'`
    - **location**: Helps surface state-specific schemes.
    - **crop**: Helps surface agriculture-specific schemes.
    - **farmer_context**: Any extra profile details for eligibility queries.
    - **language**: Preferred language for the response ('English', 'Hindi', etc.).

    Returns an `AgentOutput` JSON with `status`, `summary`, `data.rag_results`,
    `data.schemes_found`, `sources`, and `limitations`.
    """
    try:
        agent = _get_scheme_agent()
    except ImportError as exc:
        raise HTTPException(
            status_code=503,
            detail=(
                f"Scheme Agent is not available: {exc}. "
                "Ensure sentence-transformers and faiss-cpu are installed."
            ),
        )

    result = agent.run(
        query=query,
        location=location,
        crop=crop,
        farmer_context=farmer_context,
        language=language,
        top_k=top_k,
    )
    return JSONResponse(content=result.to_dict())


@app.post(
    "/query",
    summary="Smart routing — automatically picks the right agent(s)",
    response_description="List of AgentOutput objects, one per invoked agent",
)
async def smart_query(
    query: str = Form("", description="Your question (optional if image provided)"),
    location: Optional[str] = Form(None),
    crop: Optional[str] = Form(None),
    farmer_context: Optional[str] = Form(None),
    language: str = Form("English", description="Preferred response language for schemes (e.g. 'English', 'Hindi', 'Telugu')"),
    file: Optional[UploadFile] = File(None, description="Optional crop leaf image"),
    top_k: int = Form(3),
):
    """
    Single endpoint for any farmer request. The server reads the query text
    and checks whether an image was attached, then automatically decides
    which agent(s) to call.

    Routing rules:
    - Image attached → **always** calls Plant Disease Agent
    - Query contains disease/symptom keywords → calls Plant Disease Agent
    - Query contains scheme/yojana/eligibility keywords → calls Scheme Agent
    - No keyword match → defaults to Scheme Agent

    Returns a JSON object with:
    - `agents_invoked`: list of agent names that were called
    - `agents_skipped`: agents identified as relevant but not yet built
      (weather, crop_suitability)
    - `results`: dict mapping agent name → AgentOutput
    - `combined_answer`: all summaries concatenated for easy display
    """
    has_image = file is not None and file.content_type.startswith("image/")
    image_bytes = (await file.read()) if has_image else None

    intended = _route_query(query, has_image)

    # Separate into what we can handle vs. what is not built yet
    available_loaders = {
        "plant_disease": _get_disease_agent,
        "government_scheme": _get_scheme_agent,
    }
    future_agents = {"weather", "crop_suitability"}

    to_run = [n for n in intended if n in available_loaders]
    skipped = [n for n in intended if n in future_agents]

    if not to_run:
        return JSONResponse(content={
            "agents_invoked": [],
            "agents_skipped": skipped,
            "results": {},
            "combined_answer": (
                "I could not identify which capability to use for your query. "
                "Please rephrase or visit https://myscheme.gov.in for scheme "
                "information."
            ),
        })

    # Call each agent and collect results
    results = {}
    errors = []

    for agent_name in to_run:
        try:
            agent = available_loaders[agent_name]()
        except ImportError as exc:
            errors.append(f"{agent_name}: missing dependency ({exc})")
            continue

        if agent_name == "plant_disease":
            output = agent.run(
                image_bytes=image_bytes,
                crop_context=crop,
                top_k=top_k,
            )
        else:  # government_scheme
            output = agent.run(
                query=query,
                location=location,
                crop=crop,
                farmer_context=farmer_context,
                language=language,
                top_k=top_k,
            )

        results[agent_name] = output.to_dict()

    # Build a combined human-readable answer
    _labels = {
        "plant_disease": "🌿 Disease Analysis",
        "government_scheme": "📋 Government Schemes",
    }
    sections = [
        f"{_labels.get(n, n)}\n{r['summary']}"
        for n, r in results.items()
    ]
    combined = "\n\n".join(sections) if sections else "No results available."

    if skipped:
        combined += (
            "\n\n⚠️ Note: The following capabilities are not yet available in "
            f"this deployment: {', '.join(skipped)}."
        )

    return JSONResponse(content={
        "agents_invoked": list(results.keys()),
        "agents_skipped": skipped,
        "load_errors": errors,
        "results": results,
        "combined_answer": combined,
    })
