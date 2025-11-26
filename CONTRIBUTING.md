# Contributing to THAU

First off, thank you for considering contributing to THAU! It's people like you that make THAU such a great tool.

## 🌟 Ways to Contribute

### 1. 🐛 Report Bugs

Found a bug? Please create an issue with:
- Clear description of the bug
- Steps to reproduce
- Expected vs actual behavior
- Your environment (OS, Python version, GPU/CPU)
- Relevant logs or screenshots

### 2. 💡 Suggest Features

Have an idea? Open an issue with:
- Clear description of the feature
- Why it would be useful
- How it might work
- Any implementation ideas

### 3. 📝 Improve Documentation

Documentation can always be better:
- Fix typos or unclear explanations
- Add examples and tutorials
- Translate documentation
- Improve code comments

### 4. 🔧 Submit Code

Want to code? Here's how:

#### Getting Started

1. **Fork the repository**
   ```bash
   # Click "Fork" on GitHub
   ```

2. **Clone your fork**
   ```bash
   git clone https://github.com/YOUR_USERNAME/thau.git
   cd thau
   ```

3. **Set up development environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   pip install -r requirements-dev.txt  # Development dependencies
   pip install -e .  # Install in development mode
   ```

4. **Create a branch**
   ```bash
   git checkout -b feature/your-feature-name
   # OR
   git checkout -b fix/your-bug-fix
   ```

#### Development Guidelines

**Code Style:**
- Follow PEP 8
- Use type hints
- Write docstrings (Google style)
- Keep functions small and focused

**Example:**
```python
def train_model(age: int, steps: int, learning_rate: float = 1e-4) -> Dict[str, float]:
    """
    Train THAU model for a specific age.

    Args:
        age: Cognitive age (0-15)
        steps: Number of training steps
        learning_rate: Learning rate for optimizer

    Returns:
        Dictionary with training metrics (loss, accuracy, etc.)

    Raises:
        ValueError: If age is not in valid range
    """
    if not 0 <= age <= 15:
        raise ValueError(f"Age must be 0-15, got {age}")

    # Implementation...
    return {"loss": 0.5, "accuracy": 0.95}
```

**Testing:**
```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_models.py -v

# Run with coverage
pytest tests/ --cov=core --cov=memory --cov-report=html
```

**Formatting:**
```bash
# Format code
black .
isort .

# Check types
mypy core/ memory/ api/
```

#### Commit Guidelines

Use conventional commits:

```
feat: Add Age 6 training support
fix: Resolve memory leak in training loop
docs: Update installation guide
test: Add tests for self-questioning
refactor: Simplify model initialization
perf: Optimize attention mechanism
```

#### Pull Request Process

1. **Update documentation** if needed
2. **Add tests** for new features
3. **Run tests** and ensure they pass
4. **Update CHANGELOG.md**
5. **Create pull request** with:
   - Clear title
   - Description of changes
   - Link to related issue
   - Screenshots (if UI changes)

**PR Template:**
```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Refactoring

## Testing
- [ ] Tests pass locally
- [ ] Added new tests
- [ ] Documentation updated

## Related Issues
Closes #123
```

---

## 🎯 Good First Issues

New to the project? Look for issues labeled:
- `good first issue`
- `beginner friendly`
- `documentation`
- `help wanted`

---

## 🏗️ Project Structure

```
thau/
├── core/               # Core transformer models
├── thau_models/        # Specialized models (Vision, Tools)
├── thau_agents/        # Agent system
├── thau_trainer/       # Training infrastructure
├── memory/             # Memory system
├── api/                # REST API
├── tests/              # Unit and integration tests
├── docs/               # Documentation
└── examples/           # Example scripts
```

---

## 📚 Development Resources

### Understanding the Codebase

1. **Start here:**
   - `core/models/base_transformer.py` - Main model
   - `thau_trainer/own_model_manager.py` - Training manager
   - `api/thau_code_server.py` - API server

2. **Read documentation:**
   - [Architecture Overview](docs/architecture.md)
   - [Training Guide](docs/training-from-scratch.md)
   - [API Reference](docs/api-reference.md)

### Debugging

```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Use breakpoints
import pdb; pdb.set_trace()

# Or use IPython
from IPython import embed; embed()
```

---

## 🤝 Code Review Process

All submissions require review:

1. **Automated checks**: GitHub Actions runs tests
2. **Code review**: Maintainers review code
3. **Feedback**: Address review comments
4. **Approval**: Get approval from maintainer
5. **Merge**: Code is merged to main

---

## 🎓 Learning Resources

New to LLMs or transformers?

- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [Hugging Face Course](https://huggingface.co/course)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)

---

## 📞 Getting Help

Stuck? We're here to help:

- **GitHub Discussions**: Ask questions
- **Issue Comments**: Comment on relevant issues
- **Discord** (coming soon): Join our community

---

## 🌟 Recognition

Contributors are recognized in:
- README.md Contributors section
- Release notes
- Hall of Fame (coming soon)

---

## 📜 Code of Conduct

Please read and follow our [Code of Conduct](CODE_OF_CONDUCT.md).

---

## 📄 License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

## 🙏 Thank You!

Your contributions make THAU better for everyone. Thank you for being part of this journey!

*"Together, we're building the future of AI"* 🚀
