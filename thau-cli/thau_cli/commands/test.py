#!/usr/bin/env python3
"""
THAU TEST - Generate tests for code
"""

import sys
import requests
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.syntax import Syntax
from rich import print as rprint

console = Console()


class TestCommand:
    """Test command - generate tests"""

    def __init__(self):
        self.config = self._load_config()
        self.server_url = self.config.get("server_url", "http://localhost:8001")

    def _load_config(self):
        """Load configuration"""
        import yaml
        config_file = Path.home() / ".thau" / "config.yaml"
        if config_file.exists():
            with open(config_file, "r") as f:
                return yaml.safe_load(f) or {}
        return {}

    def run(self, files):
        """Run test command"""
        console.print(Panel(
            "[bold cyan]🧪 THAU Test Generator[/bold cyan]\n\n"
            "[dim]Generate comprehensive tests for your code[/dim]",
            border_style="cyan"
        ))

        if not files:
            console.print("[yellow]No files specified[/yellow]")
            console.print("Usage: thau test <file1> <file2> ...")
            sys.exit(1)

        # Generate tests for each file
        for file_path in files:
            self._generate_tests(file_path)

    def _generate_tests(self, file_path):
        """Generate tests for a file"""
        path = Path(file_path)

        if not path.exists():
            console.print(f"[red]File not found:[/red] {file_path}")
            return

        # Read file
        try:
            code = path.read_text()
        except Exception as e:
            console.print(f"[red]Error reading file:[/red] {str(e)}")
            return

        console.print(f"\n[bold cyan]Generating tests for:[/bold cyan] {file_path}\n")

        # Detect language and test framework
        lang = self._detect_language(path)
        framework = self._select_test_framework(lang)

        # Send for test generation
        with console.status("[cyan]🧠 THAU Test Writer generating tests...[/cyan]", spinner="dots"):
            try:
                response = requests.post(
                    f"{self.server_url}/api/agents/task",
                    json={
                        "description": f"Generate comprehensive {framework} tests for this {lang} code. Include unit tests, edge cases, and integration tests:\n\n```{lang}\n{code}\n```",
                        "role": "test_writer",
                    },
                    timeout=120,
                )

                if response.status_code == 200:
                    result = response.json()
                    tests = result.get("result", result.get("description", ""))

                    # Extract code if in markdown
                    if "```" in tests:
                        test_code = self._extract_code(tests)
                    else:
                        test_code = tests

                    # Show generated tests
                    console.print("\n[bold green]Generated tests:[/bold green]\n")
                    console.print(Syntax(test_code, lang, theme="monokai", line_numbers=True))
                    console.print()

                    # Ask to save
                    if Confirm.ask("Save tests?", default=True):
                        test_file = self._get_test_filename(path, framework)
                        test_path = Path(test_file)
                        test_path.parent.mkdir(parents=True, exist_ok=True)
                        test_path.write_text(test_code)
                        console.print(f"[green]✓[/green] Tests saved to: {test_file}")

                else:
                    console.print(f"[red]Error:[/red] Server returned {response.status_code}")

            except Exception as e:
                console.print(f"[red]Error:[/red] {str(e)}")

    def _detect_language(self, path):
        """Detect programming language"""
        ext = path.suffix.lower()
        lang_map = {
            ".py": "python",
            ".js": "javascript",
            ".ts": "typescript",
            ".tsx": "typescript",
            ".jsx": "javascript",
            ".java": "java",
            ".go": "go",
            ".rs": "rust",
        }
        return lang_map.get(ext, "python")

    def _select_test_framework(self, lang):
        """Select test framework based on language"""
        frameworks = {
            "python": "pytest",
            "javascript": "jest",
            "typescript": "jest",
            "java": "junit",
            "go": "go test",
            "rust": "cargo test",
        }
        return frameworks.get(lang, "pytest")

    def _get_test_filename(self, path, framework):
        """Generate test filename"""
        if framework in ["pytest", "unittest"]:
            return f"tests/test_{path.stem}.py"
        elif framework == "jest":
            return f"tests/{path.stem}.test{path.suffix}"
        elif framework == "junit":
            return f"tests/{path.stem}Test.java"
        else:
            return f"tests/test_{path.name}"

    def _extract_code(self, text):
        """Extract code from markdown code blocks"""
        lines = text.split("\n")
        in_code_block = False
        code_lines = []

        for line in lines:
            if line.strip().startswith("```"):
                in_code_block = not in_code_block
                continue
            if in_code_block:
                code_lines.append(line)

        return "\n".join(code_lines) if code_lines else text


if __name__ == "__main__":
    cmd = TestCommand()
    cmd.run(["test.py"])
