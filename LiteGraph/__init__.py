"""
LiteGraph - A minimal, production-ready AI workflow framework.

Provides graph-based orchestration, ReAct agents, and multi-agent coordination.
"""

__version__ = "0.1.0"

# Core framework exports
from .core.graph import Graph
from .core.node import node
from .core.state import State

# Agent exports
from .agent.react_agent import ReActAgent
from .agent.tool_registry import tool, ToolRegistry
from .agent.parser import ReActParser

# LLM exports
from .llms.openai_client import OpenAIClient

__all__ = [
    # Core
    "Graph",
    "node", 
    "State",
    # Agents
    "ReActAgent",
    "tool",
    "ToolRegistry",
    "ReActParser",
    # LLMs
    "OpenAIClient",
] 