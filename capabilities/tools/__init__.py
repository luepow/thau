"""
THAU Tools Module

Sistema de herramientas, MCP integration, tool factory, y system tools
"""

from .mcp_integration import (
    MCPServer,
    create_default_mcp_tools,
    MCPToolCall,
    MCPToolResult,
)

from .tool_factory import (
    ToolFactory,
    ToolSpec,
)

from .api_toolkit import (
    APIToolkit,
)

from .tool_registry import (
    ToolRegistry,
)

from .system_tools import (
    SystemTools,
)

# Web Search Tools (opcional - funciona sin dependencias)
try:
    from .web_search import (
        WebSearchTool,
        WebFetcher,
        ResearchAgent,
        SearchResult,
        WebPage,
        ResearchResult,
        web_search,
        fetch_url,
        research_topic,
    )
    WEB_SEARCH_AVAILABLE = True
except ImportError:
    WEB_SEARCH_AVAILABLE = False

__all__ = [
    # MCP Integration
    "MCPServer",
    "create_default_mcp_tools",
    "MCPToolCall",
    "MCPToolResult",

    # Tool Factory
    "ToolFactory",
    "ToolSpec",

    # API Toolkit
    "APIToolkit",

    # Tool Registry
    "ToolRegistry",

    # System Tools (Code Agent)
    "SystemTools",

    # Web Search (si está disponible)
    "WebSearchTool",
    "WebFetcher",
    "ResearchAgent",
    "SearchResult",
    "WebPage",
    "ResearchResult",
    "web_search",
    "fetch_url",
    "research_topic",
    "WEB_SEARCH_AVAILABLE",
]
