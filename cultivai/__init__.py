"""
cultivai — Shared contracts and base classes for the CultivAI multi-agent system.

This package defines the standard input/output interface that all specialist
agents must implement, enabling the Coordinator to dispatch tasks and combine
results without knowing each agent's internal implementation.

Usage:
    from cultivai.contracts import AgentOutput, AgentStatus
    from cultivai.base_agent import BaseAgent
"""

from cultivai.contracts import AgentOutput, AgentStatus
from cultivai.base_agent import BaseAgent

__all__ = ["AgentOutput", "AgentStatus", "BaseAgent"]
