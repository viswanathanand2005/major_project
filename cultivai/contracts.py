"""
CultivAI Agent Contracts
========================
Standard output schema shared by all CultivAI specialist agents.

Design goals:
  - Zero external dependencies (stdlib only) so any agent can import this.
  - Every agent returns an AgentOutput, giving the Coordinator a uniform
    surface to aggregate, log, and present results.
  - Provenance fields (sources, retrieved_at, limitations) are first-class so
    the Response layer can always cite and caveat its claims.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional


class AgentStatus(str, Enum):
    """
    Lifecycle status of an agent's response.

    SUCCESS       — Task completed; data field is populated.
    INCONCLUSIVE  — Task ran but the result is uncertain or ambiguous
                    (e.g. low model confidence, no matching scheme found).
    MISSING_INPUT — A required input was absent; the agent did not attempt
                    the task. The summary explains what is needed.
    ERROR         — An unexpected failure occurred; the error field has details.
    """
    SUCCESS = "success"
    INCONCLUSIVE = "inconclusive"
    MISSING_INPUT = "missing_input"
    ERROR = "error"


@dataclass
class AgentOutput:
    """
    Standard output produced by every CultivAI specialist agent.

    Fields
    ------
    agent : str
        Machine-readable agent identifier (e.g. "plant_disease",
        "government_scheme", "weather", "crop_suitability").

    status : AgentStatus
        Whether the task succeeded, was inconclusive, lacked required
        inputs, or failed with an error.

    summary : str
        A concise, human-readable answer suitable for direct inclusion
        in the farmer-facing response. The Coordinator uses this as the
        default text for each agent's contribution.

    data : dict[str, Any]
        Agent-specific structured output for downstream use by the
        Coordinator or other agents. Schema is agent-defined; see each
        agent's docstring for the exact keys.

    sources : list[str]
        Attribution for the data used (dataset names, API providers,
        official URLs). The Response layer must include these when
        presenting results to the farmer.

    retrieved_at : str
        ISO 8601 UTC timestamp of when this output was generated.
        Helps identify stale information (e.g. old scheme deadlines).

    limitations : list[str]
        Known constraints, caveats, or cautions that the farmer should
        understand before acting on this output.

    error : str | None
        Error message when status == ERROR; None otherwise.
    """

    agent: str
    status: AgentStatus
    summary: str
    data: dict[str, Any] = field(default_factory=dict)
    sources: list[str] = field(default_factory=list)
    retrieved_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    limitations: list[str] = field(default_factory=list)
    error: Optional[str] = None

    # ── Helpers ────────────────────────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict (JSON-safe, no enum objects)."""
        return {
            "agent": self.agent,
            "status": self.status.value,
            "summary": self.summary,
            "data": self.data,
            "sources": self.sources,
            "retrieved_at": self.retrieved_at,
            "limitations": self.limitations,
            "error": self.error,
        }

    @property
    def ok(self) -> bool:
        """True when status is SUCCESS or INCONCLUSIVE (task ran to completion)."""
        return self.status in (AgentStatus.SUCCESS, AgentStatus.INCONCLUSIVE)

    def __repr__(self) -> str:
        preview = self.summary[:60]
        ellipsis = "..." if len(self.summary) > 60 else ""
        return (
            f"AgentOutput(agent={self.agent!r}, "
            f"status={self.status.value!r}, "
            f"summary={preview + ellipsis!r})"
        )
