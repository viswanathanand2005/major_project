"""
Yojana Sahayak MCP Server
==========================

Exposes Indian government scheme data as MCP (Model Context Protocol) tools.
Designed for integration with agentic AI systems — including air-gapped,
on-premise deployments where LLMs invoke tools via stdio transport.

Tools:
    - search_schemes      — Semantic search over 585+ scheme facts
    - get_scheme_details  — Retrieve specific scheme info by name and field
    - check_eligibility   — Fetch eligibility criteria for a named scheme
    - list_schemes        — List all indexed scheme names

Transport: stdio (default) — zero network dependency, ideal for air-gapped use.

Usage:
    python -m yojana_sahayak.mcp.server
    # or: yojana-mcp
"""

import json
import sys
from pathlib import Path
from typing import Optional

# Lazy imports to keep startup fast
_retriever = None


def _get_retriever():
    """Lazy-initialize the scheme retriever."""
    global _retriever
    if _retriever is None:
        from yojana_sahayak.rag.retriever import SchemeRetriever
        _retriever = SchemeRetriever()
        _retriever.build_index()
    return _retriever


def _load_core_schemes() -> dict:
    """Load core scheme metadata for direct lookup."""
    core_path = Path(__file__).parent.parent.parent / "data" / "core_schemes.jsonl"
    schemes = {}
    if core_path.exists():
        with open(core_path, encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                name = rec.get("scheme_name", "")
                if name not in schemes:
                    schemes[name] = {}
                field = rec.get("field", "")
                msgs = rec.get("messages", [])
                answer = next(
                    (m["content"] for m in msgs if m["role"] == "assistant"), ""
                )
                if field and answer:
                    schemes[name][field] = answer
    return schemes


# ── MCP Tool Implementations ─────────────────────────────────────────────────

def search_schemes(query: str, top_k: int = 3) -> list[dict]:
    """
    Semantic search over Indian government scheme facts.

    Args:
        query: Natural language query in Hindi or English.
               Examples: 'PM Kisan ke liye kaun eligible hai?',
                         'What are the benefits of Ayushman Bharat?'
        top_k: Number of results to return (default 3).

    Returns:
        List of matching scheme facts with scheme name, field, answer, and relevance score.
    """
    retriever = _get_retriever()
    return retriever.retrieve(query, top_k=top_k)


def get_scheme_details(scheme_name: str,
                       field: Optional[str] = None) -> dict:
    """
    Get details about a specific government scheme.

    Args:
        scheme_name: Name of the scheme (e.g., 'PM Kisan', 'Ayushman Bharat').
        field: Specific field to retrieve. One of:
               'eligibility', 'benefits', 'description', 'application_process'.
               If None, returns all available fields.

    Returns:
        Dict with scheme info. Keys are field names, values are descriptions.
    """
    schemes = _load_core_schemes()

    # Fuzzy match on scheme name
    name_lower = scheme_name.lower()
    matched = None
    for name in schemes:
        if name_lower in name.lower() or name.lower() in name_lower:
            matched = name
            break

    if not matched:
        # Fall back to semantic search
        results = search_schemes(scheme_name, top_k=1)
        if results:
            return {
                "scheme": results[0]["scheme"],
                "field": results[0]["field"],
                "info": results[0]["answer"],
                "source": "semantic_search",
            }
        return {"error": f"No information found for '{scheme_name}'"}

    info = schemes[matched]
    if field and field in info:
        return {"scheme": matched, "field": field, "info": info[field]}
    return {"scheme": matched, "fields": info}


def check_eligibility(scheme_name: str, user_context: str = "") -> dict:
    """
    Retrieve eligibility criteria for a government scheme.

    Args:
        scheme_name: Name of the scheme (e.g., 'PM Kisan', 'Ayushman Bharat').
        user_context: Optional user profile (e.g., 'farmer with 2 acres land in UP').
                      When provided, it is included in the response alongside the
                      eligibility text so the calling agent can assess fit.

    Returns:
        Dict with 'eligibility_criteria' text. If user_context was given, also
        includes 'user_context' and a 'note' prompting the agent to compare.
    """
    query = f"{scheme_name} eligibility {user_context}".strip()
    results = search_schemes(query, top_k=2)

    eligibility_results = [r for r in results if r["field"] == "eligibility"]
    if not eligibility_results:
        eligibility_results = results

    if not eligibility_results:
        return {"error": f"No eligibility information found for '{scheme_name}'"}

    out = {
        "scheme": eligibility_results[0]["scheme"],
        "eligibility_criteria": eligibility_results[0]["answer"],
        "confidence": eligibility_results[0]["score"],
    }
    if user_context:
        out["user_context"] = user_context
        out["note"] = (
            "Compare 'user_context' against 'eligibility_criteria' "
            "to determine whether this person qualifies."
        )
    return out


def list_schemes() -> dict:
    """
    List all government schemes in the knowledge base.

    Returns:
        Dict with 'count' and 'schemes' (sorted list of unique scheme names).
    """
    retriever = _get_retriever()
    names = retriever.scheme_names          # public property, no private access
    return {"count": len(names), "schemes": names}


# ── MCP Server (FastMCP) ─────────────────────────────────────────────────────

def create_mcp_server():
    """Create and configure the MCP server with all tools."""
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError:
        print("Install MCP SDK: pip install 'mcp[cli]'", file=sys.stderr)
        sys.exit(1)

    mcp = FastMCP(
        "yojana-sahayak",
        instructions=(
            "Indian Government Scheme Assistant — search, retrieve, and check "
            "eligibility for 2,872+ central and state government welfare schemes. "
            "Supports Hindi and English queries."
        ),
    )

    @mcp.tool(name="search_schemes")
    def _search_schemes(query: str, top_k: int = 3) -> str:
        """
        Semantic search over 585+ Indian government scheme facts.
        Query in Hindi, English, or Hinglish.
        Returns ranked list of matching scheme facts with scheme name, field, answer, and score.
        """
        try:
            results = search_schemes(query, top_k)
            return json.dumps(results, ensure_ascii=False, indent=2)
        except Exception as exc:
            return json.dumps({"error": str(exc)}, ensure_ascii=False)

    @mcp.tool(name="get_scheme_details")
    def _get_scheme_details(scheme_name: str, field: str = "") -> str:
        """
        Get details about a specific Indian government scheme.
        scheme_name: e.g. 'PM Kisan', 'Ayushman Bharat', 'Ujjwala Yojana'.
        field (optional): one of 'eligibility', 'benefits', 'description', 'application_process'.
        Omit field to get all available information for the scheme.
        """
        try:
            result = get_scheme_details(scheme_name, field or None)
            return json.dumps(result, ensure_ascii=False, indent=2)
        except Exception as exc:
            return json.dumps({"error": str(exc)}, ensure_ascii=False)

    @mcp.tool(name="check_eligibility")
    def _check_eligibility(scheme_name: str, user_context: str = "") -> str:
        """
        Fetch eligibility criteria for an Indian government scheme.
        Optionally pass user_context (e.g. 'farmer with 2 acres in UP') —
        it will be returned alongside the criteria so you can compare and assess fit.
        """
        try:
            result = check_eligibility(scheme_name, user_context)
            return json.dumps(result, ensure_ascii=False, indent=2)
        except Exception as exc:
            return json.dumps({"error": str(exc)}, ensure_ascii=False)

    @mcp.tool(name="list_schemes")
    def _list_schemes() -> str:
        """
        List all government schemes currently in the knowledge base.
        Returns total count and a sorted list of scheme names.
        """
        try:
            result = list_schemes()
            return json.dumps(result, ensure_ascii=False, indent=2)
        except Exception as exc:
            return json.dumps({"error": str(exc)}, ensure_ascii=False)

    return mcp


def main():
    """Run the MCP server (stdio transport by default)."""
    server = create_mcp_server()
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
