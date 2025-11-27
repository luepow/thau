#!/usr/bin/env python3
"""
THAU REVIEW - Review code for bugs and improvements
"""

import sys
import requests
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.markdown import Markdown
from rich import print as rprint

console = Console()


class ReviewCommand:
    """Review command - code review"""

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
        """Run review command"""
        console.print(Panel(
            "[bold cyan]🔍 THAU Code Review[/bold cyan]\n\n"
            "[dim]Review code for bugs and improvements[/dim]",
            border_style="cyan"
        ))

        if not files:
            console.print("[yellow]No files specified[/yellow]")
            console.print("Usage: thau review <file1> <file2> ...")
            sys.exit(1)

        # Review each file
        for file_path in files:
            self._review_file(file_path)

    def _review_file(self, file_path):
        """Review a single file"""
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

        console.print(f"\n[bold cyan]Reviewing:[/bold cyan] {file_path}\n")

        # Show code
        lang = self._detect_language(path)
        console.print(Syntax(code, lang, theme="monokai", line_numbers=True))
        console.print()

        # Send for review
        with console.status("[cyan]🧠 THAU Code Reviewer analyzing...[/cyan]", spinner="dots"):
            try:
                response = requests.post(
                    f"{self.server_url}/api/agents/task",
                    json={
                        "description": f"Review this {lang} code for bugs, security issues, and improvements:\n\n```{lang}\n{code}\n```",
                        "role": "code_reviewer",
                    },
                    timeout=120,
                )

                if response.status_code == 200:
                    result = response.json()
                    review = result.get("result", result.get("description", "No review available"))

                    console.print(Panel(
                        Markdown(review),
                        title=f"🔍 Review: {path.name}",
                        border_style="cyan"
                    ))
                    console.print()

                else:
                    console.print(f"[red]Error:[/red] Server returned {response.status_code}")

            except Exception as e:
                console.print(f"[red]Error:[/red] {str(e)}")

    def _detect_language(self, path):
        """Detect programming language from file extension"""
        ext = path.suffix.lower()
        lang_map = {
            ".py": "python",
            ".js": "javascript",
            ".ts": "typescript",
            ".tsx": "tsx",
            ".jsx": "jsx",
            ".java": "java",
            ".cpp": "cpp",
            ".c": "c",
            ".go": "go",
            ".rs": "rust",
            ".rb": "ruby",
            ".php": "php",
            ".swift": "swift",
            ".kt": "kotlin",
        }
        return lang_map.get(ext, "text")


if __name__ == "__main__":
    cmd = ReviewCommand()
    cmd.run(["test.py"])
