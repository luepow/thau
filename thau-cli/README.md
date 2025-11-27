# THAU CLI

Command Line Interface for THAU AI System - Like Claude Code, but for THAU.

## Installation

```bash
# From source (development)
cd thau-cli
pip install -e .

# From PyPI (when published)
pip install thau-cli
```

## Quick Start

```bash
# Check if it works
thau --version

# Configure server
thau config --server http://localhost:8001

# Chat with THAU
thau chat "Explain async/await in Python"

# Explain code in a file
thau explain main.py

# Create a plan
thau plan "Build authentication system with JWT"

# Use specific agent
thau agent -r code_writer "Refactor this function"

# Interactive mode
thau
```

## Features

- **Chat**: Talk to THAU AI agents
- **Code Analysis**: Explain, refactor, and review code
- **Planning**: Create detailed plans for complex tasks
- **Multiple Agents**: Use specialized agents (code_writer, planner, reviewer, etc.)
- **Interactive Mode**: Chat with THAU in real-time
- **Configuration**: Save your preferences

## Commands

### `thau chat <message>`
Chat with THAU AI

```bash
thau chat "How do I handle errors in async functions?"
thau chat --agent code_writer "Fix this bug"
thau chat --file app.py "Analyze this code"
```

### `thau explain <file>`
Explain code in a file

```bash
thau explain src/main.py
```

### `thau plan <task>`
Create a detailed plan

```bash
thau plan "Implement user authentication"
```

### `thau agent`
Work with THAU agents

```bash
thau agent --list                           # List all agents
thau agent -r code_writer "Fix this bug"    # Use specific agent
```

### `thau config`
Configure THAU CLI

```bash
thau config --show                          # Show current config
thau config --server http://localhost:8001  # Set server URL
thau config --agent code_writer             # Set default agent
thau config --theme monokai                 # Set syntax theme
```

### `thau tools`
List available MCP tools

```bash
thau tools
```

## Configuration

Configuration is stored in `~/.thau/config.yaml`:

```yaml
server_url: http://localhost:8001
default_agent: general
theme: monokai
```

## Requirements

- Python 3.8+
- THAU backend server running (see main project)

## Starting THAU Server

```bash
# In main THAU project directory
python api/thau_code_server.py
```

## Examples

```bash
# Explain a Python file with syntax highlighting
thau explain app.py

# Create a plan for a complex task
thau plan "Build REST API with authentication and rate limiting"

# Use code_writer agent to refactor code
thau agent -r code_writer "Refactor the UserService class to use dependency injection"

# Interactive mode
thau
```

## License

MIT
