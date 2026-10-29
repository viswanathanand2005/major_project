"""
CultivAI — Government Scheme Specialist Agent
==============================================
Wraps the Yojana Sahayak RAG + LLM pipeline as a CultivAI-compatible
specialist agent. Produces a standard AgentOutput the Coordinator can consume
directly — without needing to know about FAISS, Whisper, MLX, or MCP.

The existing pipeline.py, cli.py, demo.py, and MCP server are completely
unchanged. This file is a coordination layer on top of the RAG retriever
and LLM generator that are already in the yojana_sahayak package.

Cross-platform design:
  - RAG retrieval (FAISS + sentence-transformers) is fully cross-platform
    and works on Windows, Linux, and macOS.
  - LLM generation (mlx_lm / Qwen2.5) requires Apple Silicon. On other
    platforms it is silently skipped and the agent falls back to a
    formatted RAG answer — no crash, no configuration change needed.

Usage (standalone):
    import sys, pathlib
    # Add major_project/ to path so cultivai/ is importable
    sys.path.insert(0, str(pathlib.Path(__file__).parents[3]))

    from yojana_sahayak.agent.cultivai_agent import SchemeAgent

    agent = SchemeAgent()
    print(agent.health_check())   # True if core_schemes.jsonl is present

    result = agent.run(
        query="PM Kisan ke liye kaun eligible hai?",
        location="Uttar Pradesh",
        crop="wheat",
    )
    print(result.status)          # AgentStatus.SUCCESS
    print(result.summary)         # LLM-generated or RAG-formatted answer

    import json
    print(json.dumps(result.to_dict(), indent=2))
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

# ── Path setup ────────────────────────────────────────────────────────────────
# Resolve major_project/ root so cultivai/ is importable from anywhere.
_AGENT_FILE = Path(__file__).resolve()                      # …/agent/cultivai_agent.py
_PROJECT_ROOT = _AGENT_FILE.parents[3]                      # major_project/

if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from cultivai.contracts import AgentOutput, AgentStatus
from cultivai.base_agent import BaseAgent

# ── Constants ─────────────────────────────────────────────────────────────────

_SOURCES = [
    "myscheme.gov.in — India's National Scheme Portal",
    "Yojana Sahayak knowledge base (2,872+ schemes from 723 official PDFs)",
    "Dataset: https://huggingface.co/datasets/Subh24ai/yojana-sahayak-instruct",
]

_LIMITATIONS = [
    "Scheme eligibility and benefits depend on state, landholding, category, "
    "income, registration status, and deadlines — verify with official sources.",
    "Scheme data sourced from myscheme.gov.in. Information may not reflect "
    "the most recent amendments or state-specific modifications.",
    "The system reports potential relevance, not confirmed eligibility or approval.",
    "For applications, always refer to the official government portal or visit "
    "your nearest Common Service Centre (CSC).",
    "On non-Apple-Silicon systems without an LLM key, answers are "
    "derived directly from the RAG knowledge base in the chosen language.",
]

# Verified English versions for core scheme records
_CORE_SCHEMES_EN = {
    ("Pradhan Mantri Kisan Samman Nidhi", "eligibility"): (
        "PM Kisan Eligibility: Small and marginal farmer families with up to 2 hectares (~5 acres) of cultivable landholding. "
        "Exclusions: Institutional landholders, income tax payers, retired government employees receiving Rs. 10,000+ monthly pension, "
        "and professionals (doctors, engineers, lawyers, chartered accountants)."
    ),
    ("Pradhan Mantri Kisan Samman Nidhi", "benefits"): (
        "PM Kisan Benefits: Eligible farmers receive Rs. 6,000 per year directly into their bank accounts via Direct Benefit Transfer (DBT) "
        "in three equal installments of Rs. 2,000 every 4 months."
    ),
    ("Pradhan Mantri Kisan Samman Nidhi", "application_process"): (
        "How to apply for PM Kisan: 1) Visit pmkisan.gov.in or your nearest Common Service Centre (CSC). "
        "2) Complete the Farmer Registration form. Required documents: Aadhaar card, land ownership records (khatauni), and bank passbook."
    ),
    ("Ayushman Bharat PM-JAY", "eligibility"): (
        "Ayushman Bharat Eligibility: Families identified under the SECC 2011 (Socio-Economic Caste Census) database as poor and vulnerable. "
        "There are no restrictions on family size, age, or gender. Verify eligibility on pmjay.gov.in or call helpline 14555."
    ),
    ("Ayushman Bharat PM-JAY", "benefits"): (
        "Ayushman Bharat Benefits: Provides health coverage of up to Rs. 5 lakh per family per year for secondary and tertiary care hospitalization "
        "across empanelled public and private hospitals nationwide, covering over 1,500 medical procedures. Pre-existing conditions are covered from day one."
    ),
    ("PM Ujjwala Yojana", "eligibility"): (
        "PM Ujjwala Yojana Eligibility: Adult women (18+) belonging to BPL (Below Poverty Line) households with no existing LPG connection in the home. "
        "Must be listed in the SECC-2011 database or hold a valid BPL ration card."
    ),
    ("PM Ujjwala Yojana", "benefits"): (
        "PM Ujjwala Yojana Benefits: Provides deposit-free LPG gas connections to eligible women, financial assistance for the security deposit and connection charges, "
        "a subsidy on the first refill, and up to 12 subsidized refill cylinders annually."
    ),
    ("PM Ujjwala Yojana", "application_process"): (
        "How to apply for PM Ujjwala Yojana: 1) Visit your nearest authorized LPG distributor (HP Gas, Bharat Gas, or Indane). "
        "2) Submit the KYC application form along with your BPL ration card / SECC proof, Aadhaar card, and bank account details. "
        "Online applications can be submitted at pmuy.gov.in."
    ),
    ("Mahatma Gandhi National Rural Employment Guarantee", "eligibility"): (
        "MGNREGA Eligibility: Any rural adult citizen (18+) willing to do unskilled manual work. "
        "Apply for a Job Card at your local Gram Panchayat. There is no income or landholding ceiling."
    ),
    ("Mahatma Gandhi National Rural Employment Guarantee", "benefits"): (
        "MGNREGA Benefits: Guarantees at least 100 days of wage employment per financial year to every rural household whose adult members volunteer for unskilled manual work, "
        "with statutory minimum wages deposited directly into bank/post office accounts within 15 days."
    ),
    ("Pradhan Mantri Awas Yojana - Gramin", "eligibility"): (
        "PMAY-Gramin Eligibility: Rural families identified under SECC 2011 as houseless or living in kutcha/dilapidated houses. "
        "Priority is given to SC/ST, minorities, widows, disabled persons, and disaster-affected families. Households with an existing pucca house are not eligible."
    ),
    ("Pradhan Mantri Awas Yojana - Gramin", "benefits"): (
        "PMAY-Gramin Benefits: Financial grant of Rs. 1.20 lakh in plain areas and Rs. 1.30 lakh in hilly/difficult areas to construct a permanent pucca house, "
        "supplemented by 90-95 days of unskilled labour under MGNREGA and toilet construction assistance under Swachh Bharat Mission."
    ),
    ("Pradhan Mantri Mudra Yojana", "eligibility"): (
        "PM Mudra Yojana Eligibility: Non-corporate, non-farm small and micro enterprises, artisans, shopkeepers, and small business owners. "
        "Loans are offered across three tiers: Shishu (up to Rs. 50,000), Kishor (Rs. 50,001 to Rs. 5 lakh), and Tarun (Rs. 5 lakh to Rs. 10 lakh) through commercial banks, RRBs, and NBFCs."
    ),
    ("Kisan Credit Card", "eligibility"): (
        "Kisan Credit Card (KCC) Eligibility: Individual farmers, joint borrowers, tenant farmers, sharecroppers, oral lessees, and Self Help Groups (SHGs). "
        "Also extended to allied agricultural sectors such as dairy, animal husbandry, and fisheries for working capital and short-term credit needs."
    ),
    ("Sukanya Samriddhi Yojana", "eligibility"): (
        "Sukanya Samriddhi Yojana Eligibility: Parents or legal guardians of a girl child below 10 years of age (maximum 2 accounts per family). "
        "Accounts can be opened at any post office or authorized commercial bank with a minimum annual deposit of Rs. 250 and maximum Rs. 1.5 lakh."
    ),
    ("Atal Pension Yojana", "eligibility"): (
        "Atal Pension Yojana Eligibility: Indian citizens aged 18 to 40 years holding a savings bank account who are not income tax payers. "
        "Provides a guaranteed monthly pension of Rs. 1,000 to Rs. 5,000 after attaining 60 years of age, based on chosen contribution."
    ),
}

def _generate_with_gemini(query: str, context: str, language: str = "English", timeout_sec: int = 6) -> Optional[str]:
    """Dynamically generate a multilingual response using LangChain and Gemini if configured."""
    import os
    from dotenv import load_dotenv
    load_dotenv()
    key = os.getenv("GOOGLE_GEMINI_KEY") or os.getenv("GEMINI_API_KEY")
    if not key:
        return None
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        from langchain_core.prompts import ChatPromptTemplate
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=key,
            temperature=0.2,
            request_timeout=timeout_sec,
        )
        prompt = ChatPromptTemplate.from_template(
            "You are CultivAI's expert assistant for Indian government welfare schemes.\n"
            "Answer the user's question accurately using ONLY the official scheme context provided below.\n\n"
            "CRITICAL: Write your entire response in: {language}.\n\n"
            "Official Scheme Context:\n{context}\n\n"
            "Farmer Question:\n{question}"
        )
        chain = prompt | llm
        resp = chain.invoke({"context": context, "question": query, "language": language})
        content = resp.content
        if isinstance(content, str) and content.strip():
            return content.strip()
    except Exception:
        pass
    return None


class SchemeAgent(BaseAgent):
    """
    Specialist agent for Indian government welfare scheme discovery and
    eligibility information. Supports Hindi, English, and Hinglish queries.

    Wraps the Yojana Sahayak RAG retriever + LLM generator with:
      - Query enrichment using location, crop, and farmer profile context
        so the retriever surfaces the most relevant scheme facts.
      - Graceful LLM fallback: attempts Qwen2.5 generation via mlx_lm
        (Apple Silicon); falls back to formatted RAG results on other
        platforms without raising an error.
      - Structured `data` payload exposing raw RAG results and scheme names
        so the Coordinator can pass scheme context to other agents.

    Input kwargs (for run()):
        query          (str)   Farmer's question in Hindi, English, or
                               Hinglish. Required.
        location       (str)   State or district (e.g. "Uttar Pradesh",
                               "Vidarbha"). Enriches the RAG query.
        crop           (str)   Crop being grown (e.g. "wheat", "cotton").
                               Enriches the RAG query.
        farmer_context (str)   Additional profile information: land size,
                               category, income, etc. Enriches the query.
        use_llm        (bool)  Attempt LLM generation. Default True.
                               Set False to get raw RAG results only.
        top_k          (int)   Number of RAG results to retrieve. Default 3.

    Output (AgentOutput.data keys):
        rag_results   list[dict]  Raw retriever output:
                                  [{"scheme", "field", "answer", "score"}]
        query_used    str         Enriched query sent to the retriever.
        llm_used      bool        Whether LLM generation succeeded.
        schemes_found list[str]   Unique scheme names from RAG results.
                                  The Coordinator can pass these to other
                                  agents for cross-referencing.
        context       dict        Echoed location/crop/farmer_context.
    """

    def __init__(self) -> None:
        self._retriever = None

    @property
    def name(self) -> str:
        return "government_scheme"

    # ── Health check ──────────────────────────────────────────────────────────

    def health_check(self) -> bool:
        """Return True when the core scheme data file is present on disk."""
        try:
            from yojana_sahayak.config import CORE_SCHEMES_PATH
            return Path(CORE_SCHEMES_PATH).exists()
        except Exception:
            return False

    # ── Lazy retriever ────────────────────────────────────────────────────────

    @property
    def retriever(self):
        """Lazy-initialize the FAISS RAG retriever on first access."""
        if self._retriever is None:
            from yojana_sahayak.rag.retriever import SchemeRetriever
            self._retriever = SchemeRetriever()
        return self._retriever

    # ── Query enrichment ──────────────────────────────────────────────────────

    @staticmethod
    def _enrich_query(
        query: str,
        location: Optional[str],
        crop: Optional[str],
        farmer_context: Optional[str],
    ) -> str:
        """
        Append available farmer context to the base query.

        This helps the FAISS retriever surface scheme facts that are
        relevant to the farmer's specific situation rather than returning
        generic top results.

        Example:
            query="PM Kisan eligibility"
            location="Uttar Pradesh", crop="wheat"
            → "PM Kisan eligibility | crop: wheat | location: Uttar Pradesh"
        """
        parts = [query.strip()]
        if crop:
            parts.append(f"crop: {crop.strip()}")
        if location:
            parts.append(f"location: {location.strip()}")
        if farmer_context:
            parts.append(farmer_context.strip())
        return " | ".join(parts)

    # ── Core run ──────────────────────────────────────────────────────────────

    def run(
        self,
        query: str = "",
        location: Optional[str] = None,
        crop: Optional[str] = None,
        farmer_context: Optional[str] = None,
        use_llm: bool = True,
        top_k: int = 3,
        language: str = "English",
        **kwargs,
    ) -> AgentOutput:
        """
        Retrieve and explain relevant government schemes for a farmer's query.

        Steps:
            1. Validate that a query was provided.
            2. Enrich the query with location / crop / farmer context.
            3. Run FAISS RAG retrieval (cross-platform, always runs).
            4. Attempt LLM generation for a conversational answer in the chosen language
               (via LangChain + Gemini, or Apple Silicon mlx_lm).
            5. Return standard AgentOutput formatted in the selected language.

        Returns:
            AgentOutput with status SUCCESS, INCONCLUSIVE, MISSING_INPUT,
            or ERROR.
        """

        # ── 1. Validate input ─────────────────────────────────────────────────
        if not query or not query.strip():
            return self._missing_input_output(
                "Please provide a question about government schemes — "
                "for example: 'PM Kisan eligibility' or "
                "'Ayushman Bharat ke kya fayde hain?'"
            )

        # ── 2. Enrich query ───────────────────────────────────────────────────
        enriched_query = self._enrich_query(query, location, crop, farmer_context)

        # ── 3. RAG retrieval ──────────────────────────────────────────────────
        try:
            rag_results = self.retriever.retrieve(enriched_query, top_k=top_k)
        except Exception as exc:
            return self._error_output(
                "Failed to search the scheme knowledge base. "
                "Please check that the data files are present and try again.",
                exc,
            )

        if not rag_results:
            return AgentOutput(
                agent=self.name,
                status=AgentStatus.INCONCLUSIVE,
                summary=(
                    "No closely matching scheme information was found for your "
                    "query. Try rephrasing, or browse all schemes at "
                    "https://myscheme.gov.in."
                ),
                data={
                    "rag_results": [],
                    "query_used": enriched_query,
                    "llm_used": False,
                    "schemes_found": [],
                    "context": _context_dict(location, crop, farmer_context),
                },
                sources=_SOURCES,
                limitations=_LIMITATIONS,
            )

        # ── 4. LLM generation in user's selected language ────────────────────
        llm_used = False
        answer = ""

        if use_llm:
            rag_context = self.retriever.retrieve_context(enriched_query, top_k=top_k)

            # Try LangChain with Google Gemini API first
            gemini_ans = _generate_with_gemini(query, rag_context, language=language)
            if gemini_ans:
                answer = gemini_ans
                llm_used = True
            else:
                # Optional fallback to mlx_lm on Apple Silicon
                try:
                    from yojana_sahayak.llm.generator import generate
                    history = []
                    context_parts = []
                    if crop:
                        context_parts.append(f"grows {crop}")
                    if location:
                        context_parts.append(f"is in {location}")
                    if farmer_context:
                        context_parts.append(farmer_context)

                    if context_parts:
                        context_note = "Farmer context: " + "; ".join(context_parts) + "."
                        history = [{
                            "user": context_note,
                            "assistant": (
                                "Understood. I will keep this farming context "
                                "in mind when answering."
                            ),
                        }]

                    mlx_ans = generate(query, context=rag_context, history=history)
                    if mlx_ans and mlx_ans.strip():
                        answer = mlx_ans
                        llm_used = True
                except (ImportError, Exception):
                    pass

        # ── 5. Build summary in user's selected language ──────────────────────
        if llm_used and answer:
            summary = answer.strip()
        else:
            top = rag_results[0]
            scheme_key = (top.get("scheme"), top.get("field"))

            if language.strip().lower().startswith("en") and scheme_key in _CORE_SCHEMES_EN:
                summary = _CORE_SCHEMES_EN[scheme_key]
            else:
                summary = (
                    f"Regarding {top['scheme']} ({top['field']}): {top['answer']}"
                )

            if len(rag_results) > 1:
                other_schemes = ", ".join(
                    r["scheme"] for r in rag_results[1:]
                    if r["scheme"] != top["scheme"]
                )
                if other_schemes:
                    if language.strip().lower().startswith("en"):
                        summary += f" Related schemes: {other_schemes}."
                    else:
                        summary += f" Anya sambandhit yojanaayein: {other_schemes}."

        # Build source list enriched with the specific schemes found
        scheme_names = sorted({r["scheme"] for r in rag_results})
        sources = list(_SOURCES) + [f"Scheme record: {n}" for n in scheme_names]

        return AgentOutput(
            agent=self.name,
            status=AgentStatus.SUCCESS,
            summary=summary,
            data={
                "rag_results": rag_results,
                "query_used": enriched_query,
                "llm_used": llm_used,
                "language": language,
                "schemes_found": scheme_names,
                "context": _context_dict(location, crop, farmer_context),
            },
            sources=sources,
            limitations=_LIMITATIONS,
        )


# ── Module helpers ─────────────────────────────────────────────────────────────

def _context_dict(
    location: Optional[str],
    crop: Optional[str],
    farmer_context: Optional[str],
) -> dict:
    """Return only the non-None context values as a clean dict."""
    return {
        k: v for k, v in {
            "location": location,
            "crop": crop,
            "farmer_context": farmer_context,
        }.items() if v
    }
