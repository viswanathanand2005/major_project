# CultivAI API — How to Run & Test

## 1. Install server dependencies

```bash
cd major_project
pip install fastapi "uvicorn[standard]" python-multipart
```

The agent dependencies (tensorflow, pillow, sentence-transformers, etc.) should already
be installed from the individual submodule setups.

---

## 2. Start the server

```bash
cd major_project
uvicorn api:app --reload --port 8000
```

The interactive API docs are at **http://localhost:8000/docs** once running.

---

## 3. Check everything is healthy

```bash
curl http://localhost:8000/health
```

Expected (both agents loaded):
```json
{
  "status": "ok",
  "agents": {
    "plant_disease":     {"registered": true, "healthy": true},
    "government_scheme": {"registered": true, "healthy": true}
  }
}
```

---

## 4. Test — Disease Detection (`POST /disease`)

Send an image of a crop leaf. The agent returns top-3 disease predictions.

```bash
curl -X POST http://localhost:8000/disease \
     -F "file=@/path/to/leaf.jpg" \
     -F "crop=tomato"
```

**PowerShell equivalent:**
```powershell
$form = @{
    file = Get-Item "C:\path\to\leaf.jpg"
    crop = "tomato"
}
Invoke-RestMethod -Uri "http://localhost:8000/disease" -Method POST -Form $form
```

**Expected response:**
```json
{
  "agent": "plant_disease",
  "status": "success",
  "summary": "Possible disease detected: Tomato Early Blight (87% confidence). Verify with a local agronomist before applying any treatment.",
  "data": {
    "predictions": [
      {"label": "Tomato_Early_blight", "prob": 0.87},
      {"label": "Tomato_Late_blight",  "prob": 0.09},
      {"label": "Tomato_healthy",       "prob": 0.02}
    ],
    "top_label": "Tomato_Early_blight",
    "top_confidence": 0.87,
    "crop_context": "tomato"
  },
  "sources": ["PlantVillage dataset — ...", "..."],
  "limitations": ["Model trained on PlantVillage (controlled lab images)...", "..."],
  "retrieved_at": "2026-10-29T12:00:00+00:00",
  "error": null
}
```

---

## 5. Test — Government Schemes (`POST /schemes`)

Ask anything about Indian government welfare schemes in Hindi, English, or Hinglish.

```bash
# English
curl -X POST http://localhost:8000/schemes \
     -F "query=What are the benefits of PM Kisan?" \
     -F "location=Uttar Pradesh" \
     -F "crop=wheat"

# Hindi
curl -X POST http://localhost:8000/schemes \
     -F "query=PM Kisan ke liye kaun eligible hai?" \
     -F "location=Bihar"

# Eligibility with farmer profile
curl -X POST http://localhost:8000/schemes \
     -F "query=Ayushman Bharat eligibility" \
     -F "location=Maharashtra" \
     -F "farmer_context=small farmer, 2 acres land, BPL category"
```

**Expected response:**
```json
{
  "agent": "government_scheme",
  "status": "success",
  "summary": "PM Kisan provides ₹6,000 per year to eligible small farmers...",
  "data": {
    "rag_results": [
      {"scheme": "Pradhan Mantri Kisan Samman Nidhi", "field": "benefits", "answer": "...", "score": 0.84}
    ],
    "schemes_found": ["Pradhan Mantri Kisan Samman Nidhi"],
    "llm_used": false,
    "query_used": "What are the benefits of PM Kisan? | crop: wheat | location: Uttar Pradesh"
  },
  "sources": ["myscheme.gov.in — ...", "..."],
  "limitations": ["Scheme eligibility depends on state, landholding...", "..."],
  "retrieved_at": "...",
  "error": null
}
```

> `llm_used` will be `false` on Windows (MLX is Apple Silicon only).
> The answer still comes from the RAG retriever — it's just not rephrased by the LLM.

---

## 6. Test — Smart Routing (`POST /query`)

Send any request. The server automatically picks the right agent(s).

```bash
# Text only → goes to Scheme Agent
curl -X POST http://localhost:8000/query \
     -F "query=What government support is available for cotton farmers in Vidarbha?" \
     -F "location=Vidarbha" \
     -F "crop=cotton"

# Image only → goes to Disease Agent
curl -X POST http://localhost:8000/query \
     -F "query=" \
     -F "file=@/path/to/leaf.jpg"

# Combined request → BOTH agents called in parallel
curl -X POST http://localhost:8000/query \
     -F "query=Is there a government scheme for this disease on my tomato crop?" \
     -F "crop=tomato" \
     -F "location=Karnataka" \
     -F "file=@/path/to/leaf.jpg"
```

**Expected response (combined):**
```json
{
  "agents_invoked": ["plant_disease", "government_scheme"],
  "agents_skipped": [],
  "load_errors": [],
  "results": {
    "plant_disease":     { ... AgentOutput ... },
    "government_scheme": { ... AgentOutput ... }
  },
  "combined_answer": "🌿 Disease Analysis\nPossible disease: Tomato Early Blight (87%)...\n\n📋 Government Schemes\nRegarding PM Fasal Bima Yojana (eligibility): ..."
}
```

**Routing logic:**

| Input | Agent(s) called |
|-------|----------------|
| Image only | `plant_disease` |
| Query with "disease", "leaf", "blight" etc. | `plant_disease` |
| Query with "scheme", "yojana", "eligible" etc. | `government_scheme` |
| Image + scheme keywords | Both |
| No keyword match | `government_scheme` (fallback) |

---

## 7. Interactive Docs (Swagger UI)

Open **http://localhost:8000/docs** in your browser.
You can upload images and fill in form fields directly from the UI — no curl needed.

---

## 8. What the future `/query` will also support

Once the Weather and Crop Suitability agents are built, they will
automatically appear in the `agents_skipped` list for relevant queries,
and then move to `agents_invoked` once registered.
