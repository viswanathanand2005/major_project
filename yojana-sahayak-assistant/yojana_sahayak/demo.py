"""
Gradio Web Demo for Yojana Sahayak.

Usage:
    python -m yojana_sahayak.demo
    # or
    python -m yojana_sahayak.cli --gradio
"""

import gradio as gr
from yojana_sahayak.agent.pipeline import YojanaPipeline

_pipeline = None

# ── Content ───────────────────────────────────────────────────────────────────

HEADER_HTML = """
<div class="ys-header">
  <div style="display:flex;align-items:center;gap:0.55rem;flex-shrink:0;">
    <span style="font-size:1.55rem;line-height:1;">🇮🇳</span>
    <div>
      <div style="font-size:1.05rem;font-weight:800;color:#fff;letter-spacing:-0.02em;line-height:1.15;">Yojana Sahayak</div>
      <div style="font-size:0.68rem;color:rgba(255,255,255,0.58);line-height:1.2;">Sovereign Voice Agent · Indian Government Schemes</div>
    </div>
  </div>
  <div style="display:flex;flex-wrap:wrap;gap:0.28rem;align-items:center;">
    <span class="ys-badge">🎙️ Whisper MLX</span>
    <span class="ys-badge">🔍 FAISS RAG</span>
    <span class="ys-badge">🧠 Qwen2.5 QLoRA</span>
    <span class="ys-badge ys-badge-active"><span class="ys-pulse-dot"></span>&nbsp;Active</span>
    <span class="ys-badge ys-badge-stats">24% WER &nbsp;·&nbsp; RTF 0.37 &nbsp;·&nbsp; 585 schemes</span>
  </div>
</div>
<div class="ys-tricolor"></div>
"""

CHIPS_HTML = """
<div class="ys-chips-wrap">
  <div class="ys-chips-label">Try an example:</div>
  <div class="ys-chips-grid">
    <button class="ys-chip" onclick="ysSetQuery('PM Kisan ke liye kaun eligible hai?')">🌾 PM Kisan</button>
    <button class="ys-chip" onclick="ysSetQuery('Ayushman Bharat ke benefits kya hain?')">🏥 Ayushman</button>
    <button class="ys-chip" onclick="ysSetQuery('Ujjwala Yojana mein gas connection kaise milega?')">🔥 Ujjwala</button>
    <button class="ys-chip" onclick="ysSetQuery('PM Awas Yojana ke liye kaun apply kar sakta hai?')">🏠 PM Awas</button>
    <button class="ys-chip" onclick="ysSetQuery('Mudra loan ke liye kya documents chahiye?')">💰 Mudra</button>
    <button class="ys-chip" onclick="ysSetQuery('Sukanya Samriddhi Yojana kya hai?')">👧 Sukanya</button>
  </div>
</div>
<script>
function ysSetQuery(text) {
  var el = document.querySelector('#ys-text-input textarea');
  if (!el) return;
  var setter = Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, 'value').set;
  setter.call(el, text);
  el.dispatchEvent(new Event('input', {bubbles: true}));
}
</script>
"""

LATENCY_TEMPLATE = """\
<div class="ys-latency-row">
  <div class="ys-lat-card {asr_cls}">
    <div class="ys-lat-value">{asr}</div>
    <div class="ys-lat-label">🎙️ ASR</div>
  </div>
  <div class="ys-lat-card {rag_cls}">
    <div class="ys-lat-value">{rag}</div>
    <div class="ys-lat-label">🔍 RAG</div>
  </div>
  <div class="ys-lat-card {llm_cls}">
    <div class="ys-lat-value">{llm}</div>
    <div class="ys-lat-label">🧠 LLM</div>
  </div>
  <div class="ys-lat-card ys-lat-total {total_cls}">
    <div class="ys-lat-value">{total}</div>
    <div class="ys-lat-label">⚡ Total</div>
  </div>
</div>"""

LATENCY_EMPTY = """\
<div class="ys-latency-row">
  <div class="ys-lat-card"><div class="ys-lat-value">—</div><div class="ys-lat-label">🎙️ ASR</div></div>
  <div class="ys-lat-card"><div class="ys-lat-value">—</div><div class="ys-lat-label">🔍 RAG</div></div>
  <div class="ys-lat-card"><div class="ys-lat-value">—</div><div class="ys-lat-label">🧠 LLM</div></div>
  <div class="ys-lat-card ys-lat-total"><div class="ys-lat-value">—</div><div class="ys-lat-label">⚡ Total</div></div>
</div>"""

# Pipeline progress for voice (ASR is first active stage)
RUNNING_HTML = """\
<div class="ys-pipeline-progress">
  <div class="ys-pp-stage ys-pp-s1">🎙️ ASR</div>
  <div class="ys-pp-arrow">→</div>
  <div class="ys-pp-stage ys-pp-s2">🔄 Rewrite</div>
  <div class="ys-pp-arrow">→</div>
  <div class="ys-pp-stage ys-pp-s3">🔍 RAG</div>
  <div class="ys-pp-arrow">→</div>
  <div class="ys-pp-stage ys-pp-s4">🧠 LLM</div>
  <div class="ys-pp-arrow">→</div>
  <div class="ys-pp-stage ys-pp-s5">🔊 TTS</div>
</div>"""

# Pipeline progress for text (ASR skipped, Rewrite is first active stage)
RUNNING_HTML_TEXT = """\
<div class="ys-pipeline-progress">
  <div class="ys-pp-skip">⟨text⟩</div>
  <div class="ys-pp-arrow">→</div>
  <div class="ys-pp-stage ys-pp-s1">🔄 Rewrite</div>
  <div class="ys-pp-arrow">→</div>
  <div class="ys-pp-stage ys-pp-s2">🔍 RAG</div>
  <div class="ys-pp-arrow">→</div>
  <div class="ys-pp-stage ys-pp-s3">🧠 LLM</div>
  <div class="ys-pp-arrow">→</div>
  <div class="ys-pp-stage ys-pp-s4">🔊 TTS</div>
</div>"""

ARCH_HTML = """\
<div class="ys-arch-flow">
  <div class="ys-arch-box" style="border-left:4px solid #0369a1;">
    <span class="ys-arch-icon">🎙️</span>
    <div>
      <div class="ys-arch-title">ASR — Whisper large-v3-turbo</div>
      <div class="ys-arch-sub">MLX · 24% WER · RTF 0.37 (2.7× real-time)</div>
    </div>
  </div>
  <div class="ys-arch-connector">↓</div>
  <div class="ys-arch-box" style="border-left:4px solid #0891b2;">
    <span class="ys-arch-icon">🔄</span>
    <div>
      <div class="ys-arch-title">Query Rewrite</div>
      <div class="ys-arch-sub">ASR correction dict · Hinglish normalization</div>
    </div>
  </div>
  <div class="ys-arch-connector">↓</div>
  <div class="ys-arch-box" style="border-left:4px solid #15803d;">
    <span class="ys-arch-icon">🔍</span>
    <div>
      <div class="ys-arch-title">RAG — FAISS IndexFlatIP</div>
      <div class="ys-arch-sub">paraphrase-multilingual-MiniLM-L12-v2 · top-k=3 · cosine ≥ 0.40</div>
    </div>
  </div>
  <div class="ys-arch-connector">↓</div>
  <div class="ys-arch-box" style="border-left:4px solid #7e22ce;">
    <span class="ys-arch-icon">🧠</span>
    <div>
      <div class="ys-arch-title">LLM — Qwen2.5-1.5B QLoRA</div>
      <div class="ys-arch-sub">4-bit MLX · perplexity 1.15 · 39,957 instruct pairs</div>
    </div>
  </div>
  <div class="ys-arch-connector">↓</div>
  <div class="ys-arch-box" style="border-left:4px solid #c2410c;">
    <span class="ys-arch-icon">🔊</span>
    <div>
      <div class="ys-arch-title">TTS</div>
      <div class="ys-arch-sub">macOS AVSpeechSynthesizer · Hindi + English</div>
    </div>
  </div>
</div>
<div class="ys-arch-stats-row">
  <div class="ys-arch-stat"><span class="ys-arch-stat-val">585</span><span class="ys-arch-stat-lbl">Scheme Facts</span></div>
  <div class="ys-arch-stat"><span class="ys-arch-stat-val">2,872</span><span class="ys-arch-stat-lbl">Schemes</span></div>
  <div class="ys-arch-stat"><span class="ys-arch-stat-val">39.9K</span><span class="ys-arch-stat-lbl">Instruct Pairs</span></div>
  <div class="ys-arch-stat"><span class="ys-arch-stat-val">dim 384</span><span class="ys-arch-stat-lbl">FAISS Index</span></div>
</div>"""

CHATBOT_PLACEHOLDER = """\
<div class="ys-welcome">
  <div style="font-size:3rem;">🇮🇳</div>
  <h3 style="margin:0.75rem 0 0.25rem;color:#1e293b;font-size:1.1rem;">Yojana Sahayak</h3>
  <p style="color:#64748b;font-size:0.88rem;max-width:340px;margin:0 auto;">
    Ask about any Indian government welfare scheme in Hindi, English, or Hinglish
  </p>
  <div style="margin-top:1.5rem;display:flex;flex-wrap:wrap;gap:0.5rem;justify-content:center;">
    <span class="ys-welcome-chip">🌾 "PM Kisan eligibility"</span>
    <span class="ys-welcome-chip">🏥 "Ayushman Bharat benefits"</span>
    <span class="ys-welcome-chip">🔥 "Ujjwala gas connection"</span>
  </div>
  <p style="color:#94a3b8;font-size:0.75rem;margin-top:1.5rem;">
    🎙️ Use voice tab or ⌨️ type your question
  </p>
</div>"""

# JS injected once — finds the latest response's word spans in the chatbot bubble
# and highlights the current word as audio plays.
HIGHLIGHT_SCRIPT = """
<script>
(function () {
    var bound = null;

    function latestTranscript() {
        var els = document.querySelectorAll('.ys-transcript');
        var best = null, bestV = -1;
        els.forEach(function (el) {
            var v = parseInt(el.getAttribute('data-ysv') || '0', 10);
            if (v > bestV) { bestV = v; best = el; }
        });
        return best;
    }

    function onTimeUpdate() {
        var transcript = latestTranscript();
        if (!transcript || !bound || !bound.duration) return;
        var words = transcript.querySelectorAll('.ys-word');
        if (!words.length) return;
        var idx = Math.floor((bound.currentTime / bound.duration) * words.length);
        idx = Math.min(idx, words.length - 1);
        words.forEach(function (w, i) {
            w.classList.toggle('ys-highlight', i === idx);
        });
    }

    function onEnded() {
        document.querySelectorAll('.ys-word.ys-highlight').forEach(function (w) {
            w.classList.remove('ys-highlight');
        });
    }

    function tryBind() {
        var el = document.querySelector('#ys-audio-out audio');
        if (!el || el === bound) return;
        if (bound) {
            bound.removeEventListener('timeupdate', onTimeUpdate);
            bound.removeEventListener('ended', onEnded);
        }
        bound = el;
        bound.addEventListener('timeupdate', onTimeUpdate);
        bound.addEventListener('ended', onEnded);
    }

    var obs = new MutationObserver(tryBind);
    function init() {
        obs.observe(document.body, { childList: true, subtree: true });
        tryBind();
    }
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})();
</script>
"""

CSS = """
/* ── Variables ── */
:root {
  --ys-primary: #1e40af;
  --ys-primary-light: #3b82f6;
  --ys-accent-saffron: #ff9933;
  --ys-accent-green: #138808;
  --ys-bg-subtle: #f8fafc;
  --ys-border: #e2e8f0;
  --ys-text: #1e293b;
  --ys-text-muted: #64748b;
  --ys-radius: 10px;
}

footer { display: none !important; }
body { overflow: hidden !important; }

.gradio-container {
  max-width: 100% !important;
  padding: 0.45rem 0.65rem 0 !important;
}

/* ── Header ── */
.ys-header {
  background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 50%, #0f3460 100%);
  border-radius: var(--ys-radius) var(--ys-radius) 0 0;
  padding: 0.6rem 1.1rem;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 0.75rem;
  flex-wrap: wrap;
  font-family: system-ui, -apple-system, sans-serif;
}

/* Indian tricolor accent bar */
.ys-tricolor {
  height: 3px;
  background: linear-gradient(to right, #ff9933 33.3%, #ffffff 33.3% 66.6%, #138808 66.6%);
  border-radius: 0 0 4px 4px;
  margin-bottom: 0.4rem;
}

/* ── Badges ── */
.ys-badge {
  background: rgba(255,255,255,0.11);
  border: 1px solid rgba(255,255,255,0.2);
  border-radius: 20px;
  padding: 0.15rem 0.55rem;
  font-size: 0.68rem;
  color: #e2e8f0;
  display: inline-flex;
  align-items: center;
  gap: 0.25rem;
}
.ys-badge-active {
  background: rgba(19,136,8,0.25) !important;
  border-color: rgba(19,136,8,0.5) !important;
  color: #86efac !important;
}
.ys-badge-stats {
  color: rgba(255,255,255,0.65) !important;
  font-size: 0.66rem !important;
}

/* Pulsing green dot */
.ys-pulse-dot {
  display: inline-block;
  width: 6px;
  height: 6px;
  border-radius: 50%;
  background: #4ade80;
  animation: ys-pulse 2s ease-in-out infinite;
}
@keyframes ys-pulse {
  0%, 100% { opacity: 1; transform: scale(1); }
  50% { opacity: 0.45; transform: scale(0.7); }
}

/* ── Tab nav ── */
.tab-nav { border-bottom: 2px solid var(--ys-border) !important; }
.tab-nav button {
  font-size: 0.85rem !important;
  font-weight: 500 !important;
  padding: 0.4rem 0.8rem !important;
}
.tab-nav button.selected {
  color: var(--ys-primary) !important;
  border-bottom-color: var(--ys-primary) !important;
  font-weight: 600 !important;
}

/* ── Primary buttons ── */
button.primary {
  background: linear-gradient(135deg, #1e40af, #1d4ed8) !important;
  border: none !important;
  font-weight: 600 !important;
  letter-spacing: 0.01em !important;
  box-shadow: 0 2px 6px rgba(30,64,175,0.3) !important;
  transition: opacity 0.15s !important;
}
button.primary:hover { opacity: 0.9 !important; }

/* ── Left panel ── */
.ys-left-col {
  overflow-y: auto !important;
  overflow-x: hidden !important;
  height: 100% !important;
  max-height: calc(100vh - 110px) !important;
  padding-right: 2px !important;
  display: flex !important;
  flex-direction: column !important;
  gap: 0.4rem !important;
}

/* ── Section divider ── */
.ys-divider {
  border: none;
  border-top: 1px solid var(--ys-border);
  margin: 0.1rem 0;
}

/* ── Section label ── */
.ys-section-label {
  font-size: 0.7rem;
  font-weight: 700;
  color: var(--ys-text-muted);
  text-transform: uppercase;
  letter-spacing: 0.06em;
  margin-bottom: 0.15rem;
}

/* ── Voice tab hint ── */
.ys-voice-hint {
  font-size: 0.75rem;
  color: var(--ys-text-muted);
  text-align: center;
  padding: 0.3rem 0.5rem 0;
  line-height: 1.5;
}

/* ── Example chips ── */
.ys-chips-wrap { padding: 0.1rem 0 0.15rem; }
.ys-chips-label {
  font-size: 0.7rem;
  color: var(--ys-text-muted);
  font-weight: 700;
  margin-bottom: 0.35rem;
  text-transform: uppercase;
  letter-spacing: 0.06em;
}
.ys-chips-grid {
  display: flex;
  flex-wrap: wrap;
  gap: 0.32rem;
}
.ys-chip {
  background: var(--ys-bg-subtle);
  border: 1px solid var(--ys-border);
  border-radius: 20px;
  padding: 0.22rem 0.6rem;
  font-size: 0.76rem;
  color: var(--ys-text);
  cursor: pointer;
  transition: all 0.15s;
  font-family: system-ui, -apple-system, sans-serif;
  white-space: nowrap;
  line-height: 1.4;
}
.ys-chip:hover {
  background: #dbeafe;
  border-color: #93c5fd;
  color: var(--ys-primary);
  transform: translateY(-1px);
  box-shadow: 0 2px 6px rgba(30,64,175,0.12);
}

/* ── Pipeline progress animation ── */
.ys-pipeline-progress {
  display: flex;
  align-items: center;
  gap: 0.18rem;
  background: #eff6ff;
  border: 1px solid #bfdbfe;
  border-radius: var(--ys-radius);
  padding: 0.5rem 0.65rem;
  flex-wrap: wrap;
  font-family: system-ui, sans-serif;
}
.ys-pp-stage {
  background: #e2e8f0;
  color: #64748b;
  border-radius: 20px;
  padding: 0.18rem 0.5rem;
  font-size: 0.7rem;
  font-weight: 500;
  white-space: nowrap;
}
.ys-pp-arrow { color: #94a3b8; font-size: 0.72rem; }
.ys-pp-skip {
  background: #f1f5f9;
  border-radius: 20px;
  padding: 0.18rem 0.5rem;
  font-size: 0.67rem;
  color: #94a3b8;
  font-style: italic;
  white-space: nowrap;
}

/* Staggered sweep: each stage pulses in sequence */
@keyframes ys-stage-sweep {
  0%, 100% { background: #dbeafe; color: #1e40af; }
  50% { background: #1d4ed8; color: #ffffff; }
}
.ys-pp-s1 { animation: ys-stage-sweep 1.6s ease-in-out 0.00s infinite; }
.ys-pp-s2 { animation: ys-stage-sweep 1.6s ease-in-out 0.32s infinite; }
.ys-pp-s3 { animation: ys-stage-sweep 1.6s ease-in-out 0.64s infinite; }
.ys-pp-s4 { animation: ys-stage-sweep 1.6s ease-in-out 0.96s infinite; }
.ys-pp-s5 { animation: ys-stage-sweep 1.6s ease-in-out 1.28s infinite; }

/* ── Latency cards ── */
.ys-latency-row {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 0.38rem;
}
.ys-lat-card {
  background: #f8fafc;
  border: 1px solid var(--ys-border);
  border-radius: var(--ys-radius);
  padding: 0.48rem 0.3rem;
  text-align: center;
  transition: border-color 0.2s;
}
.ys-lat-value {
  font-size: 0.92rem;
  font-weight: 700;
  color: var(--ys-text);
  white-space: nowrap;
}
.ys-lat-label {
  font-size: 0.6rem;
  color: var(--ys-text-muted);
  margin-top: 0.1rem;
  white-space: nowrap;
}
.ys-lat-total { background: #fff7ed !important; border-color: #fed7aa !important; }
.ys-lat-total .ys-lat-value { color: #c2410c !important; }

/* Color coding: green < 2s, yellow 2-5s, red > 5s */
.ys-lat-green { background: #f0fdf4 !important; border-color: #bbf7d0 !important; }
.ys-lat-green .ys-lat-value { color: #15803d !important; }
.ys-lat-yellow { background: #fefce8 !important; border-color: #fde68a !important; }
.ys-lat-yellow .ys-lat-value { color: #a16207 !important; }
.ys-lat-red { background: #fef2f2 !important; border-color: #fecaca !important; }
.ys-lat-red .ys-lat-value { color: #dc2626 !important; }

/* ── Word highlighting ── */
.ys-word {
  display: inline;
  border-radius: 3px;
  padding: 0 2px;
  transition: background 0.08s, color 0.08s;
}
.ys-highlight {
  background: #fef08a;
  color: #1a1a1a;
  font-weight: 600;
}

/* ── Welcome / placeholder ── */
.ys-welcome {
  text-align: center;
  padding: 3rem 1rem;
  font-family: system-ui, -apple-system, sans-serif;
}
.ys-welcome-chip {
  background: #f1f5f9;
  border: 1px solid var(--ys-border);
  border-radius: 20px;
  padding: 0.3rem 0.7rem;
  font-size: 0.78rem;
  color: var(--ys-text-muted);
  display: inline-block;
}

/* ── Architecture diagram ── */
.ys-arch-flow {
  display: flex;
  flex-direction: column;
  gap: 0;
  margin-top: 0.2rem;
}
.ys-arch-box {
  display: flex;
  align-items: center;
  gap: 0.6rem;
  background: var(--ys-bg-subtle);
  border: 1px solid var(--ys-border);
  border-radius: var(--ys-radius);
  padding: 0.55rem 0.7rem;
}
.ys-arch-icon { font-size: 1.2rem; flex-shrink: 0; }
.ys-arch-title { font-size: 0.8rem; font-weight: 600; color: var(--ys-text); }
.ys-arch-sub { font-size: 0.67rem; color: var(--ys-text-muted); margin-top: 0.08rem; }
.ys-arch-connector {
  text-align: center;
  color: #94a3b8;
  font-size: 0.85rem;
  padding: 0.08rem 0;
}
.ys-arch-stats-row {
  display: flex;
  gap: 0.45rem;
  margin-top: 0.65rem;
  flex-wrap: wrap;
}
.ys-arch-stat {
  flex: 1;
  min-width: 60px;
  background: #eff6ff;
  border: 1px solid #bfdbfe;
  border-radius: 8px;
  padding: 0.45rem 0.3rem;
  text-align: center;
}
.ys-arch-stat-val {
  display: block;
  font-size: 0.85rem;
  font-weight: 700;
  color: var(--ys-primary);
}
.ys-arch-stat-lbl {
  display: block;
  font-size: 0.62rem;
  color: var(--ys-text-muted);
  margin-top: 0.08rem;
}

/* ── Mobile responsive ── */
@media (max-width: 768px) {
  body { overflow: auto !important; }
  .gradio-container { padding: 0.25rem !important; }
  .ys-left-col {
    max-height: none !important;
    overflow-y: visible !important;
  }
  .ys-latency-row { grid-template-columns: repeat(2, 1fr) !important; }
  .ys-badge-stats { display: none !important; }
  .ys-pipeline-progress { gap: 0.12rem !important; }
  .ys-pp-stage, .ys-pp-skip { font-size: 0.62rem !important; padding: 0.14rem 0.38rem !important; }
  .ys-chips-grid { gap: 0.25rem !important; }
}

@keyframes spin { to { transform: rotate(360deg); } }
"""

# ── Helpers ───────────────────────────────────────────────────────────────────

_ys_version = 0


def _make_transcript_html(text: str) -> str:
    """Wrap each word in a span for in-bubble highlighting during audio playback."""
    global _ys_version
    _ys_version += 1
    spans = " ".join(
        f'<span class="ys-word" data-idx="{i}">{w}</span>'
        for i, w in enumerate(text.split())
    )
    return (
        f'<div class="ys-transcript" data-ysv="{_ys_version}" '
        f'style="line-height:1.9;">'
        + spans
        + "</div>"
    )


def _lat_cls(s: str) -> str:
    """Return CSS class for color-coded latency card."""
    if s == "—":
        return ""
    try:
        v = float(s.rstrip("s"))
        if v < 2:
            return "ys-lat-green"
        if v < 5:
            return "ys-lat-yellow"
        return "ys-lat-red"
    except Exception:
        return ""


# ── Pipeline ──────────────────────────────────────────────────────────────────

def _get_pipeline() -> YojanaPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = YojanaPipeline()
    return _pipeline


def _append(history: list, user: str, assistant: str) -> list:
    history = list(history or [])
    history.append({"role": "user",      "content": user})
    history.append({"role": "assistant", "content": assistant})
    return history


def _latency_html(result) -> str:
    asr_s = f"{result.asr_latency:.2f}s" if not result.asr_skipped else "—"
    rag_s = f"{result.rag_latency:.2f}s"
    llm_s = f"{result.llm_latency:.2f}s"
    tot_s = f"{result.total_latency:.2f}s"
    return LATENCY_TEMPLATE.format(
        asr=asr_s,   asr_cls=_lat_cls(asr_s),
        rag=rag_s,   rag_cls=_lat_cls(rag_s),
        llm=llm_s,   llm_cls=_lat_cls(llm_s),
        total=tot_s, total_cls=_lat_cls(tot_s),
    )


def _show_running_text(text, history):
    """Step 1 for text: immediately echo the user message + pipeline animation."""
    if not text or not text.strip():
        return history, text, RUNNING_HTML_TEXT
    history = _append(list(history or []), text, "⏳ …")
    return history, text, RUNNING_HTML_TEXT


def _run_text(text, history):
    """Step 2 for text: run pipeline and replace the placeholder."""
    if not text or not text.strip():
        return history, LATENCY_EMPTY, None
    pipeline = _get_pipeline()
    result = pipeline.run(text_input=text, speak=False)
    history = list(history or [])
    html = _make_transcript_html(result.answer)
    if history and history[-1]["role"] == "assistant":
        history[-1]["content"] = html
    from yojana_sahayak.tts.speaker import synthesize
    audio_path = synthesize(result.answer)
    return history, _latency_html(result), audio_path


def _show_running_audio(audio, history):
    """Step 1 for voice: immediately show pipeline animation."""
    if audio is None:
        return history, RUNNING_HTML
    history = _append(list(history or []), "🎤 transcribing…", "⏳ …")
    return history, RUNNING_HTML


def _run_audio(audio, history):
    """Step 2 for voice: run pipeline and replace the placeholders."""
    if audio is None:
        return history, LATENCY_EMPTY, None
    pipeline = _get_pipeline()
    result = pipeline.run(audio_path=audio, speak=False)

    # Style ASR correction: show original struck-through if corrected
    if result.query_clean != result.query_raw:
        label = (
            '<span style="font-size:0.78rem;color:#94a3b8;">🎤 Heard → </span>'
            f'<span style="text-decoration:line-through;color:#94a3b8;">{result.query_raw}</span>'
            f' <strong>{result.query_clean}</strong>'
        )
    else:
        label = f"🎤 {result.query_raw}"

    history = list(history or [])
    if len(history) >= 2:
        history[-2]["content"] = label
        history[-1]["content"] = result.answer
    else:
        history = _append(history, f"🎤 {result.query_raw}", result.answer)
    from yojana_sahayak.tts.speaker import synthesize
    audio_path = synthesize(result.answer)
    html = _make_transcript_html(result.answer)
    if len(history) >= 2:
        history[-1]["content"] = html
    return history, _latency_html(result), audio_path


def clear_all():
    _get_pipeline().reset_history()
    return [], None, "", LATENCY_EMPTY, None


# ── UI ────────────────────────────────────────────────────────────────────────

def launch_gradio(share: bool = True):
    with gr.Blocks() as demo:

        gr.HTML(HIGHLIGHT_SCRIPT)
        gr.HTML(HEADER_HTML)

        history_state = gr.State([])

        with gr.Row(height="calc(100vh - 115px)"):

            # ── Left panel ────────────────────────────────────────────────────
            with gr.Column(scale=1, min_width=290, elem_classes=["ys-left-col"]):

                with gr.Tabs():
                    # ── Voice tab ─────────────────────────────────────────────
                    with gr.Tab("🎤 Voice"):
                        audio_input = gr.Audio(
                            sources=["microphone"],
                            type="filepath",
                            label="Record your question",
                        )
                        gr.HTML(
                            '<div class="ys-voice-hint">'
                            "Speak in Hindi, English, or Hinglish<br>"
                            "<span style='font-size:0.7rem;color:#94a3b8;'>"
                            "Click the mic · speak · click Stop · press Ask</span>"
                            "</div>"
                        )
                        voice_btn = gr.Button("🎙️ Ask", variant="primary")

                    # ── Text tab ──────────────────────────────────────────────
                    with gr.Tab("⌨️ Text"):
                        gr.HTML(CHIPS_HTML)
                        text_input = gr.Textbox(
                            placeholder="PM Kisan ke liye kaun eligible hai?",
                            lines=3,
                            show_label=False,
                            submit_btn=True,
                            elem_id="ys-text-input",
                        )
                        text_btn = gr.Button("⌨️ Ask", variant="primary")

                    # ── Info tab ──────────────────────────────────────────────
                    with gr.Tab("🔧 Info"):
                        gr.HTML(ARCH_HTML)

                # Always-visible audio player
                gr.HTML('<hr class="ys-divider"><div class="ys-section-label">🔊 Voice Response</div>')
                audio_out = gr.Audio(
                    label=None,
                    show_label=False,
                    type="filepath",
                    autoplay=True,
                    interactive=False,
                    elem_id="ys-audio-out",
                )

                # Always-visible latency cards
                gr.HTML('<hr class="ys-divider"><div class="ys-section-label">⏱ Pipeline Latency</div>')
                latency_out = gr.HTML(LATENCY_EMPTY)

                gr.HTML('<hr class="ys-divider">')
                clear_btn = gr.Button("🗑 Clear conversation", variant="secondary", size="sm")

            # ── Right panel: chatbot fills all height ─────────────────────────
            with gr.Column(scale=2, min_width=460):
                chatbot = gr.Chatbot(
                    label="Conversation",
                    height="calc(100vh - 155px)",
                    layout="bubble",
                    buttons=["copy"],
                    sanitize_html=False,
                    placeholder=CHATBOT_PLACEHOLDER,
                )

        # ── Events ────────────────────────────────────────────────────────────
        voice_btn.click(
            _show_running_audio,
            inputs=[audio_input, history_state],
            outputs=[history_state, latency_out],
        ).then(
            lambda h: h, inputs=history_state, outputs=chatbot
        ).then(
            _run_audio,
            inputs=[audio_input, history_state],
            outputs=[history_state, latency_out, audio_out],
        ).then(lambda h: h, inputs=history_state, outputs=chatbot)

        def _wire_text(trigger):
            trigger.click(
                _show_running_text,
                inputs=[text_input, history_state],
                outputs=[history_state, text_input, latency_out],
            ).then(
                lambda h: h, inputs=history_state, outputs=chatbot
            ).then(
                _run_text,
                inputs=[text_input, history_state],
                outputs=[history_state, latency_out, audio_out],
            ).then(lambda h: h, inputs=history_state, outputs=chatbot)

        _wire_text(text_btn)

        text_input.submit(
            _show_running_text,
            inputs=[text_input, history_state],
            outputs=[history_state, text_input, latency_out],
        ).then(
            lambda h: h, inputs=history_state, outputs=chatbot
        ).then(
            _run_text,
            inputs=[text_input, history_state],
            outputs=[history_state, latency_out, audio_out],
        ).then(lambda h: h, inputs=history_state, outputs=chatbot)

        clear_btn.click(
            clear_all,
            outputs=[history_state, audio_input, text_input, latency_out, audio_out],
        ).then(lambda: [], outputs=chatbot)

    print("\nLaunching Yojana Sahayak Demo...")
    print("   Open in browser to test and record demo videos.\n")
    demo.launch(share=share, css=CSS, theme=gr.themes.Soft())


if __name__ == "__main__":
    launch_gradio()
