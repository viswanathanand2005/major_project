from __future__ import annotations

from abc import ABC, abstractmethod

from cultivai.contracts import AgentOutput, AgentStatus


class BaseAgent(ABC):
    """
    Abstract base class for all CultivAI specialist agents.

    The Coordinator calls `run(**kwargs)` on any agent and always receives
    an `AgentOutput` in return — regardless of which specialist is invoked.
    This uniform interface allows the Coordinator to:
      - dispatch tasks without knowing implementation details,
      - aggregate results from multiple agents in parallel,
      - handle errors and inconclusives consistently.
    """

    # ── Identity ───────────────────────────────────────────────────────────────

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Unique lowercase identifier for this agent.
        Used as the `agent` field in AgentOutput and as the routing key
        in the Coordinator's dispatch table.

        Examples: "plant_disease", "government_scheme", "weather",
                  "crop_suitability"
        """
        ...

    # ── Core interface ─────────────────────────────────────────────────────────

    @abstractmethod
    def run(self, **kwargs) -> AgentOutput:
        """
        Execute the agent's primary task and return a standard AgentOutput.

        Each concrete agent defines its own keyword arguments; the Coordinator
        is responsible for supplying the correct inputs based on the routing
        decision. Agents must:
          - validate required inputs first and return MISSING_INPUT early,
          - wrap all external calls in try/except and return ERROR on failure,
          - never raise exceptions to the Coordinator.

        Returns:
            AgentOutput with status SUCCESS, INCONCLUSIVE, MISSING_INPUT,
            or ERROR.
        """
        ...

    # ── Optional hooks ─────────────────────────────────────────────────────────

    def health_check(self) -> bool:
        """
        Verify that the agent's dependencies (models, API keys, data files)
        are present and accessible.

        Called by the Coordinator at startup or on demand. Returns True if
        the agent is ready to serve requests, False otherwise.

        Override this method for agents with non-trivial dependencies.
        """
        return True

    # ── Convenience helpers for subclasses ────────────────────────────────────

    def _error_output(self, message: str, exc: Exception = None) -> AgentOutput:
        """
        Build an ERROR AgentOutput. Use inside run() to avoid repetition.

        Args:
            message: Human-readable description of what went wrong.
            exc:     The original exception (optional; adds type + message).
        """
        return AgentOutput(
            agent=self.name,
            status=AgentStatus.ERROR,
            summary=message,
            error=f"{type(exc).__name__}: {exc}" if exc else None,
        )

    def _missing_input_output(self, message: str) -> AgentOutput:
        """Build a MISSING_INPUT AgentOutput. Use inside run() for early exits."""
        return AgentOutput(
            agent=self.name,
            status=AgentStatus.MISSING_INPUT,
            summary=message,
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"
