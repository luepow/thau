# THAU AGI v2 - Complete Guide in English

## What is THAU AGI?

THAU AGI v2 is a Proto-AGI (Prototype Artificial General Intelligence) system that integrates multiple advanced capabilities:

- **ReAct Cycle**: Reasons before acting (THINK -> PLAN -> ACT -> OBSERVE -> REFLECT)
- **Experiential Learning**: Learns from past interactions
- **Metacognition**: Self-evaluates to improve
- **Web Search**: Searches information on the internet
- **Multi-Agent**: Collaboration between specialized agents
- **Knowledge Base**: Knowledge base with RAG (Retrieval Augmented Generation)
- **Feedback Loop**: Continuous improvement with user feedback

---

## Quick Installation

### Step 1: Clone and Prepare

```bash
# Clone repository
git clone https://github.com/your-user/my-llm.git
cd my-llm

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Install Ollama (Optional but Recommended)

```bash
# On macOS
brew install ollama

# On Linux
curl -fsSL https://ollama.com/install.sh | sh

# Start Ollama
ollama serve

# Download a model (in another terminal)
ollama pull llama3.2
```

---

## Ways to Use THAU

### Option 1: Web Interface with Gradio (Recommended for Beginners)

```bash
# With Ollama (runs locally)
python scripts/gradio_thau_ollama.py

# With HuggingFace model
python scripts/gradio_thau_agi.py
```

Then open your browser at: **http://localhost:7860**

### Option 2: From Python (For Developers)

```python
from capabilities.proto_agi import ThauAGIv2, ThauConfig, ThauMode

# Create agent
agent = ThauAGIv2()

# Simple chat
response = agent.chat("Hello, what can you do?")
print(response)

# Execute task
result = agent.run("Calculate 25 * 4 + 100", ThauMode.TASK)
print(result["response"])

# Research topic
result = agent.run("Research about machine learning", ThauMode.RESEARCH)
print(result["response"])
```

### Option 3: Interactive Terminal Demo

```bash
python scripts/test_thau_agi_v2.py --demo
```

---

## Available Tools

THAU has 8 integrated tools:

| Tool | Description | Usage Example |
|------|-------------|---------------|
| `calculate` | Mathematical calculations | "Calculate 25 * 4 + 100" |
| `read_file` | Read files | "Read the file config.py" |
| `write_file` | Write files | "Write 'hello world' to test.txt" |
| `list_directory` | List directories | "List files in current directory" |
| `execute_python` | Execute Python code | "Execute print(2**10)" |
| `web_search` | Search on internet | "Search on internet what is Python" |
| `fetch_url` | Get URL content | "Get content from python.org" |
| `research` | Deep research | "Research about artificial intelligence" |

---

## Operation Modes

THAU can operate in 5 different modes:

### 1. Chat Mode (Default)
For casual conversations and general questions.

```python
agent.chat("Hello, how are you?")
```

### 2. Task Mode
For executing specific tasks with tools.

```python
agent.run("Calculate factorial of 10", ThauMode.TASK)
```

### 3. Research Mode
For deep information search.

```python
agent.run("Research about renewable energy", ThauMode.RESEARCH)
```

### 4. Collaborative Mode
Uses multiple specialized agents.

```python
agent.run("Develop a function to sort lists", ThauMode.COLLABORATIVE)
```

### 5. Learning Mode
Intensive learning of new concepts.

```python
agent.run("Learn about design patterns", ThauMode.LEARNING)
```

---

## Feedback System

THAU learns from your feedback:

```python
# If the response was good
agent.thumbs_up()

# If the response was incorrect
agent.thumbs_down(reason="The response was incomplete")

# To correct a response
agent.correct("The correct answer is...")
```

In the Gradio interface, use the 👍 and 👎 buttons.

---

## System Components

### 1. Experiential Learning

```python
from capabilities.proto_agi import ExperienceStore, Experience

# The system automatically saves experiences
store = ExperienceStore()

# Find similar experiences
experiences = store.find_similar_experiences("mathematical calculation")
```

### 2. Metacognition

The metacognitive engine evaluates each response:
- Response confidence (0-100%)
- Uncertainty detection
- Improvement suggestions

### 3. Knowledge Base with RAG

```python
from capabilities.proto_agi import KnowledgeStore, KnowledgeType

store = KnowledgeStore()

# Store knowledge
store.store(
    content="Python is an interpreted programming language",
    knowledge_type=KnowledgeType.FACT,
    source="manual"
)

# Retrieve relevant knowledge
results = store.retrieve("programming language")
```

### 4. Multi-Agent System

Available specialized agents:
- **CODER**: Writes and reviews code
- **REVIEWER**: Reviews code quality
- **RESEARCHER**: Researches information
- **PLANNER**: Plans complex tasks
- **TESTER**: Tests functionality

---

## Practical Examples

### Example 1: Smart Calculator

```
User: Calculate what is 15% of 250 plus 50
THAU: I'll calculate this step by step:
      1. 15% of 250 = 250 * 0.15 = 37.5
      2. 37.5 + 50 = 87.5
      The result is 87.5
```

### Example 2: Explore Files

```
User: List Python files in current directory
THAU: I found the following .py files:
      - main.py
      - config.py
      - utils.py
      - test_app.py
```

### Example 3: Web Research

```
User: Search on internet the latest news about AI
THAU: I found the following results:
      1. "OpenAI launches new GPT-5 model..."
      2. "Google announces advances in Gemini..."
      3. "Meta presents Llama 3..."
```

### Example 4: Generate Code

```
User: Write a Python function that calculates factorial
THAU: def factorial(n):
          if n <= 1:
              return 1
          return n * factorial(n - 1)
```

---

## Advanced Configuration

### Configure Components

```python
from capabilities.proto_agi import ThauConfig, ThauAGIv2

config = ThauConfig(
    # Model
    checkpoint_path="path/to/model",
    base_model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",

    # Behavior
    max_iterations=10,
    confidence_threshold=0.6,

    # Enable/Disable features
    enable_learning=True,
    enable_metacognition=True,
    enable_web_search=True,
    enable_multi_agent=True,
    enable_knowledge_base=True,
    enable_feedback=True,

    # Limits
    max_tokens=500,
    timeout_seconds=120.0,

    # Debug
    verbose=True
)

agent = ThauAGIv2(config)
```

### Configure Ollama

```python
from scripts.gradio_thau_ollama import ThauOllama, OllamaConfig

config = OllamaConfig(
    base_url="http://localhost:11434",
    model="llama3.2",
    temperature=0.7,
    max_tokens=1000,
    timeout=120
)

agent = ThauOllama(ollama_config=config)
```

---

## Troubleshooting

### Error: "Could not connect to Ollama"

```bash
# Check that Ollama is running
ollama serve

# In another terminal, check models
ollama list
```

### Error: "Model not found"

```bash
# Download the model
ollama pull llama3.2
```

### Error: "Web search not available"

```bash
# Install optional dependencies
pip install httpx beautifulsoup4 html2text
```

### Memory error

```python
# Use smaller model
config = ThauConfig(
    base_model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    max_tokens=256
)
```

---

## Tests

```bash
# Quick tests (without model)
python scripts/test_thau_agi_v2.py --quick

# Full tests
python scripts/test_thau_agi_v2.py

# Performance benchmark
python scripts/test_thau_agi_v2.py --benchmark

# Web search test
python scripts/test_web_search.py --quick
```

---

## Project Structure

```
my-llm/
├── capabilities/
│   ├── proto_agi/
│   │   ├── __init__.py           # Main exports
│   │   ├── thau_proto_agi.py     # Basic ReAct cycle
│   │   ├── thau_agi.py           # AGI v1
│   │   ├── thau_agi_v2.py        # AGI v2 (unified)
│   │   ├── experiential_learning.py  # Learning
│   │   ├── multi_agent.py        # Multi-agent
│   │   └── knowledge_base.py     # Knowledge + RAG
│   └── tools/
│       ├── web_search.py         # Web search
│       └── system_tools.py       # System tools
├── scripts/
│   ├── gradio_thau_agi.py        # UI with HuggingFace
│   ├── gradio_thau_ollama.py     # UI with Ollama
│   ├── test_thau_agi_v2.py       # Tests
│   └── test_web_search.py        # Web tests
└── docs/
    ├── THAU_AGI_GUIA_ES.md       # Spanish guide
    └── THAU_AGI_GUIDE_EN.md      # This guide
```

---

## Quick Start Summary

### 1. Install (One Time)

```bash
git clone https://github.com/your-user/my-llm.git
cd my-llm
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Install Ollama (One Time)

```bash
# macOS
brew install ollama

# Linux
curl -fsSL https://ollama.com/install.sh | sh

# Download model
ollama pull llama3.2
```

### 3. Run (Every Time)

```bash
# Terminal 1: Start Ollama
ollama serve

# Terminal 2: Start THAU
source venv/bin/activate
python scripts/gradio_thau_ollama.py
```

### 4. Use

Open http://localhost:7860 and start chatting!

---

## Contributing

1. Fork the repository
2. Create a branch: `git checkout -b my-feature`
3. Make commits: `git commit -m "Add feature"`
4. Push: `git push origin my-feature`
5. Open a Pull Request

---

## License

MIT License - See LICENSE file

---

## Credits

Developed with love for Thomas & Aurora.

**THAU** = **TH**omas + **AU**rora
