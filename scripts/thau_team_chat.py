#!/usr/bin/env python3
"""
THAU Team Chat - Interfaz de chat moderna con equipo de agentes.

Una interfaz estilo ChatGPT/Apple con múltiples agentes especializados
para desarrollo de software.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import json
import asyncio
from typing import Optional
from datetime import datetime
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from capabilities.agent.team_agents import TeamOrchestrator, AgentRole


# Initialize FastAPI app
app = FastAPI(title="THAU Team Chat", version="1.0.0")

# Initialize orchestrator
orchestrator = TeamOrchestrator()


# ============ API Models ============

class ChatMessage(BaseModel):
    content: str
    agent: Optional[str] = None


class ProjectStart(BaseModel):
    description: str


# ============ API Endpoints ============

@app.get("/api/agents")
async def get_agents():
    """Get list of all available agents."""
    return orchestrator.get_all_agents()


@app.post("/api/chat")
async def chat(message: ChatMessage):
    """Send a chat message and get response."""
    target = AgentRole(message.agent) if message.agent else None

    responses = []
    for response in orchestrator.send_message(message.content, target, stream=False):
        responses.append(response)

    return responses[-1] if responses else {"error": "No response"}


@app.post("/api/project/start")
async def start_project(project: ProjectStart):
    """Start a new project."""
    responses = []
    for response in orchestrator.start_project(project.description):
        responses.append(response)

    return responses[-1] if responses else {"error": "No response"}


@app.post("/api/clear")
async def clear_chat():
    """Clear all conversation history."""
    orchestrator.clear_all()
    return {"status": "cleared"}


@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    """WebSocket endpoint for streaming chat."""
    await websocket.accept()

    try:
        while True:
            data = await websocket.receive_json()
            message = data.get("content", "")
            agent = data.get("agent")

            target = AgentRole(agent) if agent else None

            for response in orchestrator.send_message(message, target, stream=True):
                await websocket.send_json(response)
                await asyncio.sleep(0.01)  # Small delay for streaming effect

    except WebSocketDisconnect:
        pass


# ============ HTML Frontend ============

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>THAU Team Chat</title>
    <link href="https://fonts.googleapis.com/css2?family=SF+Pro+Display:wght@400;500;600;700&family=SF+Mono:wght@400;500&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/github-dark.min.css">
    <script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/marked/11.1.1/marked.min.js"></script>
    <style>
        :root {
            --bg-primary: #1a1a1a;
            --bg-secondary: #2d2d2d;
            --bg-tertiary: #3d3d3d;
            --text-primary: #ffffff;
            --text-secondary: #a0a0a0;
            --text-muted: #666666;
            --accent: #6366f1;
            --accent-hover: #818cf8;
            --border: #404040;
            --success: #10b981;
            --warning: #f59e0b;
            --error: #ef4444;
            --shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
            --radius: 16px;
            --radius-sm: 8px;
            --transition: all 0.2s ease;
        }

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Display', 'Segoe UI', Roboto, sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            height: 100vh;
            overflow: hidden;
        }

        /* Layout */
        .app-container {
            display: flex;
            height: 100vh;
        }

        /* Sidebar */
        .sidebar {
            width: 280px;
            background: var(--bg-secondary);
            border-right: 1px solid var(--border);
            display: flex;
            flex-direction: column;
            flex-shrink: 0;
        }

        .sidebar-header {
            padding: 20px;
            border-bottom: 1px solid var(--border);
        }

        .logo {
            display: flex;
            align-items: center;
            gap: 12px;
            font-size: 24px;
            font-weight: 700;
        }

        .logo-icon {
            width: 40px;
            height: 40px;
            background: linear-gradient(135deg, var(--accent), #a855f7);
            border-radius: 12px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 20px;
        }

        /* Agents List */
        .agents-section {
            padding: 16px;
            flex: 1;
            overflow-y: auto;
        }

        .section-title {
            font-size: 12px;
            font-weight: 600;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 12px;
        }

        .agent-list {
            display: flex;
            flex-direction: column;
            gap: 4px;
        }

        .agent-item {
            display: flex;
            align-items: center;
            gap: 12px;
            padding: 12px;
            border-radius: var(--radius-sm);
            cursor: pointer;
            transition: var(--transition);
        }

        .agent-item:hover {
            background: var(--bg-tertiary);
        }

        .agent-item.active {
            background: var(--accent);
        }

        .agent-avatar {
            width: 36px;
            height: 36px;
            border-radius: 10px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 18px;
        }

        .agent-info {
            flex: 1;
        }

        .agent-name {
            font-size: 14px;
            font-weight: 500;
        }

        .agent-role {
            font-size: 12px;
            color: var(--text-secondary);
        }

        .agent-status {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: var(--success);
        }

        /* New Chat Button */
        .new-chat-btn {
            margin: 16px;
            padding: 14px;
            background: var(--accent);
            border: none;
            border-radius: var(--radius-sm);
            color: white;
            font-size: 14px;
            font-weight: 500;
            cursor: pointer;
            transition: var(--transition);
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 8px;
        }

        .new-chat-btn:hover {
            background: var(--accent-hover);
        }

        /* Main Chat Area */
        .chat-container {
            flex: 1;
            display: flex;
            flex-direction: column;
            min-width: 0;
        }

        /* Chat Header */
        .chat-header {
            padding: 16px 24px;
            border-bottom: 1px solid var(--border);
            display: flex;
            align-items: center;
            justify-content: space-between;
            background: var(--bg-secondary);
        }

        .chat-title {
            display: flex;
            align-items: center;
            gap: 12px;
        }

        .chat-title h2 {
            font-size: 16px;
            font-weight: 600;
        }

        .chat-subtitle {
            font-size: 12px;
            color: var(--text-secondary);
        }

        /* Messages Area */
        .messages-container {
            flex: 1;
            overflow-y: auto;
            padding: 24px;
            display: flex;
            flex-direction: column;
            gap: 24px;
        }

        .message {
            display: flex;
            gap: 16px;
            max-width: 900px;
            animation: fadeIn 0.3s ease;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .message.user {
            flex-direction: row-reverse;
            margin-left: auto;
        }

        .message-avatar {
            width: 40px;
            height: 40px;
            border-radius: 12px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 18px;
            flex-shrink: 0;
        }

        .message.user .message-avatar {
            background: var(--accent);
        }

        .message-content {
            flex: 1;
            min-width: 0;
        }

        .message-header {
            display: flex;
            align-items: center;
            gap: 8px;
            margin-bottom: 8px;
        }

        .message-author {
            font-size: 14px;
            font-weight: 600;
        }

        .message-time {
            font-size: 12px;
            color: var(--text-muted);
        }

        .message-body {
            background: var(--bg-secondary);
            padding: 16px;
            border-radius: var(--radius);
            line-height: 1.6;
            font-size: 15px;
        }

        .message.user .message-body {
            background: var(--accent);
        }

        .message-body p {
            margin-bottom: 12px;
        }

        .message-body p:last-child {
            margin-bottom: 0;
        }

        .message-body pre {
            background: var(--bg-primary);
            border-radius: var(--radius-sm);
            padding: 16px;
            overflow-x: auto;
            margin: 12px 0;
        }

        .message-body code {
            font-family: 'SF Mono', 'Fira Code', monospace;
            font-size: 13px;
        }

        .message-body ul, .message-body ol {
            margin: 12px 0;
            padding-left: 24px;
        }

        .message-body li {
            margin-bottom: 6px;
        }

        /* Typing Indicator */
        .typing-indicator {
            display: flex;
            gap: 4px;
            padding: 12px 16px;
        }

        .typing-dot {
            width: 8px;
            height: 8px;
            background: var(--text-secondary);
            border-radius: 50%;
            animation: typing 1.4s infinite;
        }

        .typing-dot:nth-child(2) { animation-delay: 0.2s; }
        .typing-dot:nth-child(3) { animation-delay: 0.4s; }

        @keyframes typing {
            0%, 60%, 100% { transform: translateY(0); }
            30% { transform: translateY(-8px); }
        }

        /* Input Area */
        .input-container {
            padding: 16px 24px 24px;
            background: var(--bg-primary);
        }

        .input-wrapper {
            max-width: 900px;
            margin: 0 auto;
            position: relative;
        }

        .input-box {
            display: flex;
            align-items: flex-end;
            background: var(--bg-secondary);
            border: 1px solid var(--border);
            border-radius: var(--radius);
            padding: 12px;
            transition: var(--transition);
        }

        .input-box:focus-within {
            border-color: var(--accent);
            box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.2);
        }

        .input-textarea {
            flex: 1;
            background: transparent;
            border: none;
            outline: none;
            color: var(--text-primary);
            font-size: 15px;
            font-family: inherit;
            resize: none;
            max-height: 200px;
            line-height: 1.5;
            padding: 4px 8px;
        }

        .input-textarea::placeholder {
            color: var(--text-muted);
        }

        .send-btn {
            width: 40px;
            height: 40px;
            background: var(--accent);
            border: none;
            border-radius: 10px;
            color: white;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: var(--transition);
            flex-shrink: 0;
        }

        .send-btn:hover:not(:disabled) {
            background: var(--accent-hover);
        }

        .send-btn:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }

        .send-btn svg {
            width: 20px;
            height: 20px;
        }

        /* Mention Suggestions */
        .mention-suggestions {
            position: absolute;
            bottom: 100%;
            left: 0;
            right: 0;
            background: var(--bg-secondary);
            border: 1px solid var(--border);
            border-radius: var(--radius-sm);
            margin-bottom: 8px;
            display: none;
            box-shadow: var(--shadow);
        }

        .mention-suggestions.active {
            display: block;
        }

        .mention-item {
            display: flex;
            align-items: center;
            gap: 12px;
            padding: 12px;
            cursor: pointer;
            transition: var(--transition);
        }

        .mention-item:hover {
            background: var(--bg-tertiary);
        }

        /* Welcome Screen */
        .welcome-screen {
            flex: 1;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            text-align: center;
            padding: 40px;
        }

        .welcome-icon {
            width: 80px;
            height: 80px;
            background: linear-gradient(135deg, var(--accent), #a855f7);
            border-radius: 24px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 40px;
            margin-bottom: 24px;
        }

        .welcome-title {
            font-size: 28px;
            font-weight: 700;
            margin-bottom: 12px;
        }

        .welcome-subtitle {
            font-size: 16px;
            color: var(--text-secondary);
            margin-bottom: 32px;
            max-width: 500px;
        }

        .quick-actions {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 12px;
            max-width: 600px;
        }

        .quick-action {
            padding: 16px;
            background: var(--bg-secondary);
            border: 1px solid var(--border);
            border-radius: var(--radius-sm);
            text-align: left;
            cursor: pointer;
            transition: var(--transition);
        }

        .quick-action:hover {
            border-color: var(--accent);
            background: var(--bg-tertiary);
        }

        .quick-action-title {
            font-weight: 500;
            margin-bottom: 4px;
        }

        .quick-action-desc {
            font-size: 13px;
            color: var(--text-secondary);
        }

        /* Scrollbar */
        ::-webkit-scrollbar {
            width: 8px;
        }

        ::-webkit-scrollbar-track {
            background: transparent;
        }

        ::-webkit-scrollbar-thumb {
            background: var(--border);
            border-radius: 4px;
        }

        ::-webkit-scrollbar-thumb:hover {
            background: var(--text-muted);
        }

        /* Mobile Responsive */
        @media (max-width: 768px) {
            .sidebar {
                position: fixed;
                left: -280px;
                top: 0;
                bottom: 0;
                z-index: 100;
                transition: var(--transition);
            }

            .sidebar.open {
                left: 0;
            }

            .quick-actions {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <div class="app-container">
        <!-- Sidebar -->
        <aside class="sidebar">
            <div class="sidebar-header">
                <div class="logo">
                    <div class="logo-icon">🧠</div>
                    <span>THAU Team</span>
                </div>
            </div>

            <div class="agents-section">
                <div class="section-title">Equipo de Desarrollo</div>
                <div class="agent-list" id="agentList">
                    <!-- Agents loaded dynamically -->
                </div>
            </div>

            <button class="new-chat-btn" onclick="newChat()">
                <svg width="16" height="16" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 4v16m8-8H4"/>
                </svg>
                Nueva Conversación
            </button>
        </aside>

        <!-- Main Chat Area -->
        <main class="chat-container">
            <header class="chat-header">
                <div class="chat-title">
                    <span id="currentAgentEmoji">📊</span>
                    <div>
                        <h2 id="currentAgentName">THAU Team</h2>
                        <div class="chat-subtitle" id="currentAgentRole">Equipo de desarrollo autónomo</div>
                    </div>
                </div>
            </header>

            <div class="messages-container" id="messagesContainer">
                <!-- Welcome Screen -->
                <div class="welcome-screen" id="welcomeScreen">
                    <div class="welcome-icon">🚀</div>
                    <h1 class="welcome-title">THAU Team Chat</h1>
                    <p class="welcome-subtitle">
                        Un equipo completo de agentes IA especializados para desarrollar tu proyecto.
                        Describe tu idea y el equipo se organizará para construirla.
                    </p>
                    <div class="quick-actions">
                        <div class="quick-action" onclick="sendQuickMessage('Quiero crear una aplicación web de e-commerce con carrito de compras')">
                            <div class="quick-action-title">🛒 E-commerce</div>
                            <div class="quick-action-desc">Tienda online con carrito</div>
                        </div>
                        <div class="quick-action" onclick="sendQuickMessage('Necesito una API REST para gestión de usuarios con autenticación JWT')">
                            <div class="quick-action-title">🔐 API REST</div>
                            <div class="quick-action-desc">Backend con autenticación</div>
                        </div>
                        <div class="quick-action" onclick="sendQuickMessage('Diseña un dashboard de analytics con gráficos interactivos')">
                            <div class="quick-action-title">📊 Dashboard</div>
                            <div class="quick-action-desc">Analytics con gráficos</div>
                        </div>
                        <div class="quick-action" onclick="sendQuickMessage('Crea una aplicación móvil para gestión de tareas con sincronización')">
                            <div class="quick-action-title">📱 App Móvil</div>
                            <div class="quick-action-desc">Gestión de tareas</div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Input Area -->
            <div class="input-container">
                <div class="input-wrapper">
                    <div class="mention-suggestions" id="mentionSuggestions">
                        <!-- Populated dynamically -->
                    </div>
                    <div class="input-box">
                        <textarea
                            class="input-textarea"
                            id="messageInput"
                            placeholder="Describe tu proyecto o usa @agente para hablar con uno específico..."
                            rows="1"
                            onkeydown="handleKeyDown(event)"
                            oninput="handleInput(event)"
                        ></textarea>
                        <button class="send-btn" id="sendBtn" onclick="sendMessage()">
                            <svg fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8"/>
                            </svg>
                        </button>
                    </div>
                </div>
            </div>
        </main>
    </div>

    <script>
        // State
        let agents = [];
        let currentAgent = null;
        let isLoading = false;
        let ws = null;

        // Agent colors
        const agentColors = {
            pmo: '#6366f1',
            architect: '#8b5cf6',
            backend: '#10b981',
            frontend: '#f59e0b',
            ux: '#ec4899',
            qa: '#14b8a6',
            devops: '#f97316'
        };

        // Initialize
        async function init() {
            await loadAgents();
            setupWebSocket();
            setupTextareaAutoResize();
        }

        // Load agents
        async function loadAgents() {
            try {
                const response = await fetch('/api/agents');
                agents = await response.json();
                renderAgentList();
                renderMentionSuggestions();
            } catch (error) {
                console.error('Error loading agents:', error);
            }
        }

        // Render agent list in sidebar
        function renderAgentList() {
            const container = document.getElementById('agentList');
            container.innerHTML = agents.map(agent => `
                <div class="agent-item" onclick="selectAgent('${agent.role}')" data-role="${agent.role}">
                    <div class="agent-avatar" style="background: ${agent.color}">${agent.emoji}</div>
                    <div class="agent-info">
                        <div class="agent-name">${agent.name}</div>
                        <div class="agent-role">${getRoleDescription(agent.role)}</div>
                    </div>
                    <div class="agent-status"></div>
                </div>
            `).join('');
        }

        // Render mention suggestions
        function renderMentionSuggestions() {
            const container = document.getElementById('mentionSuggestions');
            container.innerHTML = agents.map(agent => `
                <div class="mention-item" onclick="insertMention('${agent.role}')">
                    <div class="agent-avatar" style="background: ${agent.color}; width: 28px; height: 28px; font-size: 14px;">${agent.emoji}</div>
                    <span>@${agent.role}</span>
                    <span style="color: var(--text-muted); font-size: 12px;">${agent.name}</span>
                </div>
            `).join('');
        }

        // Get role description
        function getRoleDescription(role) {
            const descriptions = {
                pmo: 'Project Manager',
                architect: 'Arquitecto de Software',
                backend: 'Desarrollador Backend',
                frontend: 'Desarrollador Frontend',
                ux: 'Diseñador UX/UI',
                qa: 'QA Engineer',
                devops: 'DevOps Engineer'
            };
            return descriptions[role] || role;
        }

        // Select agent
        function selectAgent(role) {
            currentAgent = agents.find(a => a.role === role);

            // Update UI
            document.querySelectorAll('.agent-item').forEach(el => {
                el.classList.toggle('active', el.dataset.role === role);
            });

            document.getElementById('currentAgentEmoji').textContent = currentAgent.emoji;
            document.getElementById('currentAgentName').textContent = currentAgent.name;
            document.getElementById('currentAgentRole').textContent = getRoleDescription(role);
        }

        // Setup WebSocket
        function setupWebSocket() {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            ws = new WebSocket(`${protocol}//${window.location.host}/ws/chat`);

            ws.onmessage = (event) => {
                const data = JSON.parse(event.data);
                handleStreamResponse(data);
            };

            ws.onclose = () => {
                setTimeout(setupWebSocket, 3000);
            };
        }

        // Handle streaming response
        let currentMessageElement = null;

        function handleStreamResponse(data) {
            if (!currentMessageElement) {
                // Create new message
                hideWelcomeScreen();
                currentMessageElement = createAgentMessage(data.agent);
            }

            // Update content
            const bodyElement = currentMessageElement.querySelector('.message-body');
            bodyElement.innerHTML = marked.parse(data.content);
            highlightCode(bodyElement);

            // Scroll to bottom
            const container = document.getElementById('messagesContainer');
            container.scrollTop = container.scrollHeight;

            if (data.done) {
                currentMessageElement = null;
                isLoading = false;
                document.getElementById('sendBtn').disabled = false;
            }
        }

        // Create agent message element
        function createAgentMessage(agent) {
            const container = document.getElementById('messagesContainer');

            const messageHtml = `
                <div class="message agent">
                    <div class="message-avatar" style="background: ${agent.color}">${agent.emoji}</div>
                    <div class="message-content">
                        <div class="message-header">
                            <span class="message-author">${agent.name}</span>
                            <span class="message-time">${formatTime(new Date())}</span>
                        </div>
                        <div class="message-body">
                            <div class="typing-indicator">
                                <div class="typing-dot"></div>
                                <div class="typing-dot"></div>
                                <div class="typing-dot"></div>
                            </div>
                        </div>
                    </div>
                </div>
            `;

            container.insertAdjacentHTML('beforeend', messageHtml);
            return container.lastElementChild;
        }

        // Create user message
        function createUserMessage(content) {
            const container = document.getElementById('messagesContainer');

            const messageHtml = `
                <div class="message user">
                    <div class="message-avatar">👤</div>
                    <div class="message-content">
                        <div class="message-header">
                            <span class="message-author">Tú</span>
                            <span class="message-time">${formatTime(new Date())}</span>
                        </div>
                        <div class="message-body">${escapeHtml(content)}</div>
                    </div>
                </div>
            `;

            container.insertAdjacentHTML('beforeend', messageHtml);
        }

        // Send message
        function sendMessage() {
            const input = document.getElementById('messageInput');
            const content = input.value.trim();

            if (!content || isLoading) return;

            // Hide welcome screen
            hideWelcomeScreen();

            // Create user message
            createUserMessage(content);

            // Send via WebSocket
            isLoading = true;
            document.getElementById('sendBtn').disabled = true;

            ws.send(JSON.stringify({
                content: content,
                agent: currentAgent?.role || null
            }));

            // Clear input
            input.value = '';
            input.style.height = 'auto';

            // Scroll to bottom
            const container = document.getElementById('messagesContainer');
            container.scrollTop = container.scrollHeight;
        }

        // Send quick message
        function sendQuickMessage(content) {
            document.getElementById('messageInput').value = content;
            sendMessage();
        }

        // New chat
        async function newChat() {
            try {
                await fetch('/api/clear', { method: 'POST' });
                document.getElementById('messagesContainer').innerHTML = `
                    <div class="welcome-screen" id="welcomeScreen">
                        <div class="welcome-icon">🚀</div>
                        <h1 class="welcome-title">THAU Team Chat</h1>
                        <p class="welcome-subtitle">
                            Un equipo completo de agentes IA especializados para desarrollar tu proyecto.
                        </p>
                        <div class="quick-actions">
                            <div class="quick-action" onclick="sendQuickMessage('Quiero crear una aplicación web de e-commerce')">
                                <div class="quick-action-title">🛒 E-commerce</div>
                                <div class="quick-action-desc">Tienda online con carrito</div>
                            </div>
                            <div class="quick-action" onclick="sendQuickMessage('Necesito una API REST con autenticación')">
                                <div class="quick-action-title">🔐 API REST</div>
                                <div class="quick-action-desc">Backend con JWT</div>
                            </div>
                            <div class="quick-action" onclick="sendQuickMessage('Diseña un dashboard de analytics')">
                                <div class="quick-action-title">📊 Dashboard</div>
                                <div class="quick-action-desc">Analytics con gráficos</div>
                            </div>
                            <div class="quick-action" onclick="sendQuickMessage('Crea una app móvil de tareas')">
                                <div class="quick-action-title">📱 App Móvil</div>
                                <div class="quick-action-desc">Gestión de tareas</div>
                            </div>
                        </div>
                    </div>
                `;
                currentAgent = null;
                document.querySelectorAll('.agent-item').forEach(el => el.classList.remove('active'));
            } catch (error) {
                console.error('Error clearing chat:', error);
            }
        }

        // Hide welcome screen
        function hideWelcomeScreen() {
            const welcome = document.getElementById('welcomeScreen');
            if (welcome) welcome.remove();
        }

        // Handle keyboard
        function handleKeyDown(event) {
            if (event.key === 'Enter' && !event.shiftKey) {
                event.preventDefault();
                sendMessage();
            }
        }

        // Handle input for mentions
        function handleInput(event) {
            const input = event.target;
            const value = input.value;
            const lastAt = value.lastIndexOf('@');

            if (lastAt !== -1 && lastAt === value.length - 1) {
                document.getElementById('mentionSuggestions').classList.add('active');
            } else {
                document.getElementById('mentionSuggestions').classList.remove('active');
            }

            // Auto resize
            input.style.height = 'auto';
            input.style.height = Math.min(input.scrollHeight, 200) + 'px';
        }

        // Insert mention
        function insertMention(role) {
            const input = document.getElementById('messageInput');
            const value = input.value;
            const lastAt = value.lastIndexOf('@');

            if (lastAt !== -1) {
                input.value = value.substring(0, lastAt) + '@' + role + ' ';
            } else {
                input.value += '@' + role + ' ';
            }

            document.getElementById('mentionSuggestions').classList.remove('active');
            input.focus();
        }

        // Setup textarea auto resize
        function setupTextareaAutoResize() {
            const textarea = document.getElementById('messageInput');
            textarea.addEventListener('input', () => {
                textarea.style.height = 'auto';
                textarea.style.height = Math.min(textarea.scrollHeight, 200) + 'px';
            });
        }

        // Highlight code
        function highlightCode(element) {
            element.querySelectorAll('pre code').forEach(block => {
                hljs.highlightElement(block);
            });
        }

        // Format time
        function formatTime(date) {
            return date.toLocaleTimeString('es-ES', { hour: '2-digit', minute: '2-digit' });
        }

        // Escape HTML
        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }

        // Initialize on load
        document.addEventListener('DOMContentLoaded', init);
    </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def get_index():
    """Serve the main chat interface."""
    return HTML_TEMPLATE


# ============ Main ============

def main():
    """Main entry point."""
    print("=" * 60)
    print("  THAU Team Chat")
    print("  Chat con equipo de agentes especializados")
    print("=" * 60)

    # Check Ollama
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.ok:
            print("\n✅ Ollama activo")
        else:
            print("\n⚠️ Ollama no disponible")
    except:
        print("\n⚠️ Ollama no disponible")

    print("\n🚀 Iniciando servidor...")
    print("   URL: http://localhost:7862")
    print("\n   Presiona Ctrl+C para detener\n")

    uvicorn.run(app, host="0.0.0.0", port=7862)


if __name__ == "__main__":
    main()
