# THAU Code Desktop

AI-Powered Code Assistant Desktop Application - Built with THAU Agent System

<div align="center">

![THAU Code](https://img.shields.io/badge/THAU-Code-blue)
![Version](https://img.shields.io/badge/version-1.0.0-green)
![License](https://img.shields.io/badge/license-MIT-blue)

**The world's first AI code assistant with self-creating tools and visual imagination**

</div>

## Overview

**THAU Code** is a desktop application that brings the power of THAU's Agent System to your development workflow. Similar to Claude Code or GitHub Copilot, but with unique capabilities:

- **11 Specialized AI Agents** for different coding tasks
- **AI-Powered Task Planning** (like Claude Code)
- **Auto-Tool Creation** from natural language (unique to THAU)
- **Visual Imagination** using THAU's proprietary VAE
- **MCP Compatible** - works with Claude/OpenAI tools
- **Real-time Agent Communication** via WebSocket

## Architecture

```
THAU Code Desktop (Electron + React + TypeScript)
├── Chat Interface (Real-time messaging)
├── Monaco Editor (VS Code's editor)
├── Agent Panel (11 specialized agents)
├── Planner View (Task decomposition)
└── Tool Factory (Auto-tool generation)
        ↕
THAU Backend (FastAPI + Python)
├── THAU-2B Model (up to 2B parameters)
├── Agent System (Orchestration)
├── THAU Visual VAE (Image generation)
└── MCP Server (Tool registry)
```

## Features

### 1. Specialized AI Agents

Choose from 11 specialized agents for different tasks:

- 💬 **General** - General-purpose assistant
- ✍️ **Code Writer** - Write new code
- 👀 **Code Reviewer** - Review and improve code
- 🐛 **Debugger** - Find and fix bugs
- 🔍 **Researcher** - Research topics and APIs
- 📋 **Planner** - Plan complex tasks
- 🏗️ **Architect** - Design system architecture
- 🧪 **Tester** - Write tests
- 📝 **Documenter** - Generate documentation
- 🔌 **API Specialist** - Work with APIs
- 📊 **Data Analyst** - Analyze data
- 🔒 **Security** - Security analysis
- 🎨 **Visual Creator** - Generate images

### 2. AI-Powered Planner

Break down complex tasks into manageable steps:

- Analyzes task complexity
- Estimates time and effort
- Identifies dependencies
- Lists required tools
- Assesses risks and assumptions
- Defines success criteria

### 3. Tool Factory (Unique!)

Create tools from natural language descriptions:

```
Description: "Send email notifications with HTML templates"
     ↓
Generated: email_notification_tool.py (ready to use!)
```

**Examples:**
- "Call REST API with retry logic and authentication"
- "Create calendar events with timezone support"
- "Query PostgreSQL database with parameterized queries"

### 4. Monaco Code Editor

Full-featured code editor powered by VS Code's Monaco:

- Syntax highlighting
- IntelliSense
- Code folding
- Multiple languages
- Dark/Light themes

### 5. Real-time Communication

WebSocket-based real-time messaging:

- Instant responses
- Streaming output
- Tool invocations
- Connection status

## Installation

### Prerequisites

- **Node.js** 18+ and npm
- **Python** 3.10+ with virtual environment
- **THAU Backend** running (see backend setup)

### 1. Clone Repository

```bash
cd thau-code-desktop
```

### 2. Install Dependencies

```bash
npm install
```

### 3. Start THAU Backend

In a separate terminal:

```bash
cd ..
source venv/bin/activate
python api/thau_code_server.py
```

The backend will start on `http://localhost:8001`

### 4. Run Development Mode

```bash
npm run dev
```

This starts:
- Vite dev server on `http://localhost:5173`
- Electron app (automatically opens)

## Building for Production

### Build All Platforms

```bash
npm run build
npm run build:electron
```

### Platform-Specific Builds

**macOS:**
```bash
npm run build:electron -- --mac
```

**Windows:**
```bash
npm run build:electron -- --win
```

**Linux:**
```bash
npm run build:electron -- --linux
```

### Installers

After building, installers will be in `dist-electron/`:

- **macOS**: `.dmg` and `.zip`
- **Windows**: `.exe` (NSIS) and portable
- **Linux**: `.AppImage` and `.deb`

## Usage

### 1. Connect to THAU Backend

On startup, THAU Code automatically connects to:
- Default: `http://localhost:8001`
- Configure in Electron settings

### 2. Chat with Agents

1. Select an agent from the sidebar or panel
2. Type your message
3. Send (Enter or click button)
4. Receive response with reasoning

**Example:**
```
You: "Write a Python function to calculate Fibonacci numbers"
Code Writer Agent: "Here's an efficient implementation..."
[Shows code with explanation]
```

### 3. Create a Plan

1. Click **Planner** icon (📋)
2. Enter task description
3. Select priority (Low/Medium/High/Critical)
4. Click "Create Plan"
5. View detailed steps, risks, and timeline

**Example:**
```
Task: "Build a REST API for user authentication"
  ↓
Plan:
  - Step 1: Design database schema (2h)
  - Step 2: Implement user model (1h)
  - Step 3: Create authentication endpoints (3h)
  - ... (15 more steps)

Estimated: 24 hours
Risks: 3 identified
Success Criteria: 5 defined
```

### 4. Generate Tools

1. Click **Tools** icon (🔧)
2. Go to "Create" tab
3. Describe tool in natural language
4. Click "Generate Tool"
5. Tool is created and ready to use!

**Example:**
```
Description: "Fetch weather data from OpenWeatherMap API"
  ↓
Generated: weather_api_tool.py
  - Parameters: city, api_key
  - Returns: temperature, humidity, conditions
  - Includes error handling and retry logic
```

### 5. Code Editing

1. Click **Editor** icon (📝)
2. Write or paste code
3. Full Monaco editor features
4. Syntax highlighting for all languages

## Project Structure

```
thau-code-desktop/
├── electron/              # Electron main process
│   ├── main.js           # Main entry point
│   └── preload.js        # Secure bridge to renderer
├── src/
│   ├── components/       # React components
│   │   ├── AgentPanel.tsx
│   │   ├── ChatInterface.tsx
│   │   ├── CodeEditor.tsx
│   │   ├── PlannerView.tsx
│   │   └── ToolFactory.tsx
│   ├── services/         # API & WebSocket services
│   │   ├── api.ts
│   │   └── websocket.ts
│   ├── types/            # TypeScript types
│   │   └── index.ts
│   ├── styles/           # Global styles
│   │   └── global.css
│   ├── App.tsx           # Main app component
│   └── main.tsx          # React entry point
├── public/               # Static assets
├── package.json
├── tsconfig.json
├── vite.config.ts
└── README.md
```

## Configuration

### Backend URL

By default, connects to `http://localhost:8001`. To change:

**Environment Variable:**
```bash
export THAU_BACKEND_URL=http://your-server:8001
```

**Or configure in Electron:**
```javascript
// electron/main.js
process.env.THAU_BACKEND_URL = 'http://your-server:8001';
```

### Agent Selection

Default agent is **General**. Change in UI or programmatically:

```typescript
setSelectedAgent(AgentRole.CODE_WRITER);
```

## API Reference

### WebSocket Messages

**Send Message:**
```json
{
  "type": "message",
  "content": "Your message here",
  "agent_role": "code_writer"
}
```

**Receive Response:**
```json
{
  "type": "message",
  "content": "Agent response",
  "agent_role": "code_writer",
  "timestamp": "2025-11-15T10:30:00Z",
  "thinking": "Reasoning process...",
  "task_id": "abc123"
}
```

**Invoke Tool:**
```json
{
  "type": "tool_call",
  "tool_name": "generate_image",
  "arguments": {
    "prompt": "a robot coding",
    "num_images": 3
  }
}
```

### REST API Endpoints

**Get Agents:**
```
GET /api/agents
```

**Create Task:**
```
POST /api/agents/task
{
  "description": "Task description",
  "role": "code_writer",
  "priority": "high"
}
```

**Create Plan:**
```
POST /api/planner/create
{
  "task_description": "Task to plan",
  "priority": "medium"
}
```

**Generate Tool:**
```
POST /api/tools/create
{
  "description": "Tool description",
  "template_name": "api_client" (optional)
}
```

See full API documentation in backend README.

## Comparison with Other Tools

| Feature | THAU Code | Claude Code | GitHub Copilot |
|---------|-----------|-------------|----------------|
| Chat Interface | ✅ | ✅ | ❌ |
| Code Editor | ✅ Monaco | ✅ | ✅ VS Code |
| Specialized Agents | ✅ 11 types | ❌ | ❌ |
| Task Planning | ✅ AI-powered | ✅ | ❌ |
| **Auto-Tool Creation** | ✅ **Unique!** | ❌ | ❌ |
| **Visual Imagination** | ✅ **Unique!** | ❌ | ❌ |
| MCP Compatible | ✅ | ✅ | ❌ |
| Self-Learning | ✅ | ❌ | ❌ |
| Desktop App | ✅ Electron | ✅ Native | ✅ Extension |
| Open Source Backend | ✅ | ❌ | ❌ |

## Keyboard Shortcuts

- `Cmd/Ctrl + Enter` - Send message
- `Cmd/Ctrl + K` - Focus chat input
- `Cmd/Ctrl + ,` - Settings (coming soon)
- `Cmd/Ctrl + Q` - Quit application

## Troubleshooting

### Connection Issues

**Problem:** Can't connect to backend

**Solution:**
1. Check backend is running: `curl http://localhost:8001/health`
2. Check firewall settings
3. Verify backend URL in settings

### WebSocket Disconnects

**Problem:** WebSocket keeps disconnecting

**Solution:**
1. Check network stability
2. Increase timeout in `websocket.ts`
3. Check backend logs for errors

### Build Errors

**Problem:** Build fails with TypeScript errors

**Solution:**
```bash
npm run lint
npm run build -- --mode development
```

### Monaco Editor Not Loading

**Problem:** Code editor doesn't appear

**Solution:**
1. Clear browser cache (in dev mode)
2. Reinstall: `rm -rf node_modules && npm install`
3. Check console for errors

## Development

### Run Tests

```bash
npm test
```

### Lint Code

```bash
npm run lint
```

### Format Code

```bash
npm run format
```

### Debug Electron

In `electron/main.js`:
```javascript
mainWindow.webContents.openDevTools();
```

### Hot Reload

Dev mode includes hot reload for React. For Electron:
```bash
npm run dev:electron
```

## Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing`)
5. Open Pull Request

## Roadmap

- [ ] Settings panel for configuration
- [ ] File system integration (open/save files)
- [ ] Git integration (commits, branches)
- [ ] Extensions system (plugins)
- [ ] Voice input/output
- [ ] Collaboration (share sessions)
- [ ] Cloud sync (cross-device)
- [ ] Mobile app (iOS/Android)

## License

MIT License - See LICENSE file for details

## Credits

**THAU Team**
- AI Model: THAU-2B (custom trained)
- Visual System: THAU VAE (proprietary)
- Agent System: Original implementation
- Tool Factory: Unique innovation

**Technologies:**
- Electron
- React + TypeScript
- Monaco Editor (Microsoft)
- FastAPI (Python)
- WebSocket

## Support

- Documentation: See `/docs`
- Issues: GitHub Issues
- Discussions: GitHub Discussions
- Email: support@thau-ai.com

---

**Built with ❤️ by the THAU Team**

*Making AI development accessible to everyone*
