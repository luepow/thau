#!/usr/bin/env python3
"""
THAU Code Server - Backend API for THAU Code Desktop Application

Exposes:
- WebSocket for real-time chat with THAU agents
- REST API for MCP tool invocation
- Agent orchestration endpoints
- Planner endpoints
- Tool Factory endpoints
"""

from fastapi import FastAPI, WebSocket, HTTPException, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Any, Optional
import json
import asyncio
from datetime import datetime
import uuid

# THAU Imports
from capabilities.agents import (
    get_agent_orchestrator,
    AgentRole,
    Task,
    ThauPlanner,
    TaskPriority,
    TaskComplexity
)
from capabilities.tools.mcp_integration import (
    MCPServer,
    create_default_mcp_tools,
    MCPToolCall,
    MCPToolResult
)
from capabilities.tools.tool_factory import ToolFactory, ToolSpec
from capabilities.tools.api_toolkit import APIToolkit

# Initialize FastAPI
app = FastAPI(
    title="THAU Code Server",
    description="Backend for THAU Code Desktop Application",
    version="1.0.0"
)

# CORS Configuration - Allow Electron app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize THAU Components
orchestrator = get_agent_orchestrator()
planner = ThauPlanner()
tool_factory = ToolFactory()
api_toolkit = APIToolkit()
mcp_registry = create_default_mcp_tools()
mcp_server = MCPServer(mcp_registry)

# Active WebSocket connections
active_connections: Dict[str, WebSocket] = {}

# Session management
sessions: Dict[str, Dict[str, Any]] = {}


# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class ChatMessage(BaseModel):
    """Chat message from client"""
    session_id: str
    message: str
    agent_role: Optional[str] = "general"


class ChatResponse(BaseModel):
    """Chat response to client"""
    message: str
    agent_role: str
    timestamp: str
    metadata: Optional[Dict[str, Any]] = None


class TaskRequest(BaseModel):
    """Request to create a task"""
    description: str
    role: str
    priority: Optional[str] = "medium"


class PlanRequest(BaseModel):
    """Request to create a plan"""
    task_description: str
    priority: str = "medium"


class ToolCreationRequest(BaseModel):
    """Request to create a tool"""
    description: str
    template_name: Optional[str] = None


class MCPInvokeRequest(BaseModel):
    """Request to invoke MCP tool"""
    session_id: str
    tool_name: str
    arguments: Dict[str, Any]


# ============================================================================
# WEBSOCKET ENDPOINTS
# ============================================================================

@app.websocket("/ws/chat/{session_id}")
async def websocket_chat(websocket: WebSocket, session_id: str):
    """
    WebSocket endpoint for real-time chat with THAU agents

    Client sends:
    {
        "type": "message",
        "content": "user message",
        "agent_role": "general" | "code_writer" | etc.
    }

    Server responds:
    {
        "type": "message",
        "content": "agent response",
        "agent_role": "general",
        "timestamp": "ISO 8601",
        "thinking": "agent reasoning (optional)"
    }
    """
    await websocket.accept()
    active_connections[session_id] = websocket

    # Create MCP session
    if session_id not in sessions:
        sessions[session_id] = mcp_server.create_session(session_id)

    print(f"🔌 WebSocket connected: {session_id}")

    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            message_data = json.loads(data)

            print(f"📨 Received: {message_data}")

            message_type = message_data.get("type", "message")

            if message_type == "message":
                # Process chat message
                content = message_data.get("content", "")
                agent_role_str = message_data.get("agent_role", "general")

                try:
                    agent_role = AgentRole(agent_role_str)
                except ValueError:
                    agent_role = AgentRole.GENERAL

                # Create task for agent
                task = orchestrator.assign_task(
                    task_description=content,
                    role=agent_role
                )

                # Simulate agent processing (in production, this would call actual THAU model)
                response_content = f"[{agent_role.value.upper()}] Processing: {content}"

                # Send response
                response = {
                    "type": "message",
                    "content": response_content,
                    "agent_role": agent_role.value,
                    "timestamp": datetime.now().isoformat(),
                    "task_id": task.id,
                    "thinking": f"Analyzed task complexity and assigned to {agent_role.value}"
                }

                await websocket.send_text(json.dumps(response))

            elif message_type == "tool_call":
                # Handle MCP tool invocation
                tool_name = message_data.get("tool_name")
                arguments = message_data.get("arguments", {})

                result = mcp_server.handle_tool_call(
                    session_id=session_id,
                    tool_name=tool_name,
                    arguments=arguments
                )

                response = {
                    "type": "tool_result",
                    "tool_name": tool_name,
                    "success": result.success,
                    "result": result.result,
                    "error": result.error,
                    "timestamp": datetime.now().isoformat()
                }

                await websocket.send_text(json.dumps(response))

            elif message_type == "ping":
                # Heartbeat
                await websocket.send_text(json.dumps({"type": "pong"}))

    except WebSocketDisconnect:
        print(f"🔌 WebSocket disconnected: {session_id}")
        if session_id in active_connections:
            del active_connections[session_id]
    except Exception as e:
        print(f"❌ WebSocket error: {e}")
        if session_id in active_connections:
            del active_connections[session_id]


# ============================================================================
# REST API ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    """Health check"""
    return {
        "name": "THAU Code Server",
        "version": "1.0.0",
        "status": "running",
        "timestamp": datetime.now().isoformat()
    }


@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "components": {
            "orchestrator": {
                "active_agents": len(orchestrator.agents),
                "pending_tasks": len([t for t in orchestrator.tasks.values() if t.status == "pending"])
            },
            "mcp_server": {
                "registered_tools": len(mcp_registry.tools),
                "active_sessions": len(mcp_server.active_sessions)
            },
            "websocket": {
                "active_connections": len(active_connections)
            }
        },
        "timestamp": datetime.now().isoformat()
    }


# ============================================================================
# MCP ENDPOINTS
# ============================================================================

@app.get("/api/mcp/tools")
async def list_mcp_tools():
    """List all available MCP tools"""
    return {
        "tools": mcp_registry.list_tools(),
        "total": len(mcp_registry.tools)
    }


@app.post("/api/mcp/invoke")
async def invoke_mcp_tool(request: MCPInvokeRequest):
    """Invoke an MCP tool"""
    result = mcp_server.handle_tool_call(
        session_id=request.session_id,
        tool_name=request.tool_name,
        arguments=request.arguments
    )

    return {
        "call_id": result.call_id,
        "success": result.success,
        "result": result.result,
        "error": result.error,
        "execution_time_ms": result.execution_time_ms
    }


@app.post("/api/mcp/session")
async def create_mcp_session(session_id: Optional[str] = None):
    """Create a new MCP session"""
    if not session_id:
        session_id = str(uuid.uuid4())

    session = mcp_server.create_session(session_id)
    return session


# ============================================================================
# AGENT ENDPOINTS
# ============================================================================

@app.get("/api/agents")
async def list_agents():
    """List all agents"""
    agents_list = []
    for agent in orchestrator.agents.values():
        agents_list.append({
            "id": agent.id,
            "role": agent.config.role.value,
            "name": agent.config.name,
            "description": agent.config.description,
            "status": "active" if agent.current_task else "idle"
        })

    return {"agents": agents_list, "total": len(agents_list)}


@app.post("/api/agents/task")
async def create_agent_task(request: TaskRequest):
    """Create a task for an agent"""
    try:
        role = AgentRole(request.role)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid agent role: {request.role}")

    task = orchestrator.assign_task(
        task_description=request.description,
        role=role
    )

    return {
        "task_id": task.id,
        "description": task.description,
        "agent_role": task.agent_role.value,
        "status": task.status
    }


@app.get("/api/agents/tasks")
async def list_tasks():
    """List all tasks"""
    tasks_list = []
    for task in orchestrator.tasks.values():
        tasks_list.append({
            "id": task.id,
            "description": task.description,
            "agent_role": task.agent_role.value,
            "status": task.status,
            "created_at": task.created_at.isoformat() if task.created_at else None,
            "completed_at": task.completed_at.isoformat() if task.completed_at else None
        })

    return {"tasks": tasks_list, "total": len(tasks_list)}


@app.get("/api/agents/tasks/{task_id}")
async def get_task(task_id: str):
    """Get task details"""
    task = orchestrator.tasks.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    return {
        "id": task.id,
        "description": task.description,
        "agent_role": task.agent_role.value,
        "status": task.status,
        "result": task.result,
        "created_at": task.created_at.isoformat() if task.created_at else None,
        "completed_at": task.completed_at.isoformat() if task.completed_at else None
    }


# ============================================================================
# PLANNER ENDPOINTS
# ============================================================================

@app.post("/api/planner/create")
async def create_plan(request: PlanRequest):
    """Create a plan for a task"""
    try:
        priority = TaskPriority(request.priority)
    except ValueError:
        priority = TaskPriority.MEDIUM

    plan = planner.create_plan(
        task_description=request.task_description,
        priority=priority
    )

    return {
        "task_description": plan.task_description,
        "complexity": plan.complexity.value,
        "priority": plan.priority.value,
        "estimated_hours": plan.estimated_hours,
        "steps": [
            {
                "step_number": step.step_number,
                "description": step.description,
                "action_type": step.action_type,
                "estimated_effort": step.estimated_effort,
                "dependencies": step.dependencies,
                "tools_needed": step.tools_needed
            }
            for step in plan.steps
        ],
        "risks": plan.risks,
        "assumptions": plan.assumptions,
        "success_criteria": plan.success_criteria
    }


@app.post("/api/planner/analyze")
async def analyze_task(task_description: str):
    """Analyze task complexity"""
    analysis = planner.analyze_task(task_description)
    return analysis


# ============================================================================
# TOOL FACTORY ENDPOINTS
# ============================================================================

@app.post("/api/tools/create")
async def create_tool(request: ToolCreationRequest):
    """Create a tool from description"""
    tool = tool_factory.create_from_description(request.description)

    return {
        "name": tool.name,
        "description": tool.spec.description,
        "category": tool.spec.category,
        "parameters": [
            {
                "name": p.name,
                "type": p.type,
                "description": p.description,
                "required": p.required
            }
            for p in tool.spec.parameters
        ],
        "file_path": tool.file_path,
        "created_at": tool.created_at.isoformat()
    }


@app.get("/api/tools/list")
async def list_generated_tools():
    """List all generated tools"""
    return {
        "tools": [
            {
                "name": tool.name,
                "description": tool.spec.description,
                "category": tool.spec.category,
                "created_at": tool.created_at.isoformat()
            }
            for tool in tool_factory.generated_tools
        ],
        "total": len(tool_factory.generated_tools)
    }


@app.get("/api/tools/templates")
async def list_tool_templates():
    """List available tool templates"""
    return {
        "templates": list(tool_factory.templates.keys())
    }


# ============================================================================
# API TOOLKIT ENDPOINTS
# ============================================================================

@app.get("/api/toolkit/webhooks")
async def list_webhooks():
    """List registered webhooks"""
    webhooks = []
    for name, config in api_toolkit.webhook_manager.webhooks.items():
        webhooks.append({
            "name": name,
            "url": config["url"],
            "events": config["events"],
            "active": config["active"]
        })

    return {"webhooks": webhooks, "total": len(webhooks)}


@app.get("/api/toolkit/calendar/events")
async def list_calendar_events():
    """List calendar events"""
    return {
        "events": api_toolkit.calendar.events,
        "total": len(api_toolkit.calendar.events)
    }


# ============================================================================
# STARTUP / SHUTDOWN
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Server startup"""
    print("="*70)
    print("🚀 THAU Code Server Starting...")
    print("="*70)
    print(f"✅ Agent Orchestrator: {len(orchestrator.agents)} agents")
    print(f"✅ MCP Registry: {len(mcp_registry.tools)} tools")
    print(f"✅ Tool Factory: {len(tool_factory.templates)} templates")
    print(f"✅ API Toolkit: Ready")
    print("="*70)
    print("📡 Server ready to accept connections")
    print("="*70)


@app.on_event("shutdown")
async def shutdown_event():
    """Server shutdown"""
    print("\n" + "="*70)
    print("🛑 THAU Code Server Shutting Down...")
    print("="*70)

    # Close all WebSocket connections
    for session_id, ws in list(active_connections.items()):
        await ws.close()
        print(f"   Closed connection: {session_id}")

    print("✅ Shutdown complete")
    print("="*70)


if __name__ == "__main__":
    import uvicorn

    print("="*70)
    print("🧠 THAU Code Server")
    print("="*70)
    print("Starting server on http://localhost:8001")
    print("WebSocket endpoint: ws://localhost:8001/ws/chat/{session_id}")
    print("="*70)

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8001,
        log_level="info"
    )
