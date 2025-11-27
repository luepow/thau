#!/usr/bin/env python3
"""
THAU REFACTOR - Refactor code
"""

import sys
import requests
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.syntax import Syntax
from rich.markdown import Markdown
from rich import print as rprint

console = Console()


class RefactorCommand:
    """Refactor command - refactor code"""

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

    def run(self, file_path):
        """Run refactor command"""
        console.print(Panel(
            "[bold cyan]♻️  THAU Refactor[/bold cyan]\n\n"
            "[dim]Refactor code following best practices[/dim]",
            border_style="cyan"
        ))

        path = Path(file_path)

        if not path.exists():
            console.print(f"[red]File not found:[/red] {file_path}")
            sys.exit(1)

        # Read file
        try:
            code = path.read_text()
        except Exception as e:
            console.print(f"[red]Error reading file:[/red] {str(e)}")
            sys.exit(1)

        console.print(f"\n[bold cyan]Refactoring:[/bold cyan] {file_path}\n")

        # Show original code
        lang = self._detect_language(path)
        console.print("[bold]Original code:[/bold]\n")
        console.print(Syntax(code, lang, theme="monokai", line_numbers=True))
        console.print()

        # Get refactoring goals
        goals = Prompt.ask(
            "[cyan]Refactoring goals[/cyan]",
            default="Improve readability, remove code smells, follow SOLID principles"
        )

        # Send for refactoring
        with console.status("[cyan]🧠 THAU Refactorer working...[/cyan]", spinner="dots"):
            try:
                response = requests.post(
                    f"{self.server_url}/api/agents/task",
                    json={
                        "description": f"Refactor this {lang} code. Goals: {goals}\n\n```{lang}\n{code}\n```",
                        "role": "refactorer",
                    },
                    timeout=120,
                )

                if response.status_code == 200:
                    result = response.json()
                    refactored = result.get("result", result.get("description", ""))

                    # Extract code if in markdown
                    if "```" in refactored:
                        refactored_code = self._extract_code(refactored)
                    else:
                        refactored_code = refactored

                    # Show refactored code
                    console.print("\n[bold green]Refactored code:[/bold green]\n")
                    console.print(Syntax(refactored_code, lang, theme="monokai", line_numbers=True))
                    console.print()

                    # Ask to save
                    if Confirm.ask("Apply refactoring?", default=False):
                        path.write_text(refactored_code)
                        console.print(f"[green]✓[/green] File updated: {file_path}")
                    else:
                        # Save to new file
                        if Confirm.ask("Save to new file?"):
                            new_name = path.stem + "_refactored" + path.suffix
                            new_path = path.parent / new_name
                            new_path.write_text(refactored_code)
                            console.print(f"[green]✓[/green] Saved to: {new_path}")

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
        }
        return lang_map.get(ext, "text")

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
    cmd = RefactorCommand()
    cmd.run("test.py")
