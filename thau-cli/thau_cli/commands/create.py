#!/usr/bin/env python3
"""
THAU CREATE - Create files, classes, components, APIs, etc.

Intelligently creates different types of project artifacts
"""

import os
import sys
import requests
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.syntax import Syntax
from rich import print as rprint

console = Console()


class CreateCommand:
    """Create command - intelligent file/directory creation"""

    def __init__(self):
        self.config = self._load_config()
        self.server_url = self.config.get("server_url", "http://localhost:8001")
        self.templates = {
            "file": self._create_file,
            "dir": self._create_directory,
            "class": self._create_class,
            "function": self._create_function,
            "component": self._create_component,
            "api": self._create_api,
            "model": self._create_model,
            "test": self._create_test,
            "route": self._create_route,
            "service": self._create_service,
        }

    def _load_config(self):
        """Load configuration"""
        from pathlib import Path
        import yaml

        config_file = Path.home() / ".thau" / "config.yaml"
        if config_file.exists():
            with open(config_file, "r") as f:
                return yaml.safe_load(f) or {}
        return {}

    def run(self, type_name, name):
        """Run create command"""
        console.print(f"\n[cyan]Creating {type_name}: {name}[/cyan]\n")

        # Check if type is supported
        if type_name not in self.templates:
            console.print(f"[red]Unknown type:[/red] {type_name}")
            console.print("[dim]Supported types:[/dim]")
            for t in self.templates.keys():
                console.print(f"  - {t}")
            sys.exit(1)

        # Execute creation
        try:
            self.templates[type_name](name)
        except Exception as e:
            console.print(f"[red]Error:[/red] {str(e)}")
            sys.exit(1)

    def _create_file(self, name):
        """Create a simple file"""
        path = Path(name)

        if path.exists():
            if not Confirm.ask(f"File {name} exists. Overwrite?"):
                console.print("[yellow]Cancelled[/yellow]")
                return

        # Ask for content
        content = Prompt.ask("File content (or press Enter for empty)")

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content + "\n" if content else "")

        console.print(f"[green]✓[/green] Created: {name}")

    def _create_directory(self, name):
        """Create a directory"""
        path = Path(name)

        if path.exists():
            console.print(f"[yellow]Directory already exists:[/yellow] {name}")
            return

        path.mkdir(parents=True, exist_ok=True)
        console.print(f"[green]✓[/green] Created directory: {name}")

    def _create_class(self, name):
        """Create a class using THAU AI"""
        description = Prompt.ask("Describe the class")
        language = Prompt.ask("Language", choices=["python", "typescript", "java"], default="python")

        # Use THAU to generate the class
        with console.status("[cyan]Generating class with THAU...[/cyan]"):
            try:
                response = requests.post(
                    f"{self.server_url}/api/agents/task",
                    json={
                        "description": f"Create a {language} class named {name}. {description}",
                        "role": "code_writer",
                    },
                    timeout=60,
                )

                if response.status_code == 200:
                    result = response.json()
                    code = result.get("result", result.get("description", ""))

                    # Extract code from markdown if present
                    if "```" in code:
                        code = self._extract_code_from_markdown(code)

                    # Display code
                    console.print("\n[green]Generated code:[/green]\n")
                    console.print(Syntax(code, language, theme="monokai"))

                    # Ask to save
                    if Confirm.ask("\nSave to file?"):
                        ext = {"python": ".py", "typescript": ".ts", "java": ".java"}
                        filename = Prompt.ask("Filename", default=f"{name}{ext[language]}")
                        Path(filename).write_text(code)
                        console.print(f"[green]✓[/green] Saved to: {filename}")
                else:
                    console.print(f"[red]Error:[/red] Server returned {response.status_code}")

            except Exception as e:
                console.print(f"[red]Error:[/red] {str(e)}")

    def _create_function(self, name):
        """Create a function using THAU AI"""
        description = Prompt.ask("Describe the function")
        language = Prompt.ask("Language", choices=["python", "typescript", "javascript"], default="python")

        with console.status("[cyan]Generating function with THAU...[/cyan]"):
            try:
                response = requests.post(
                    f"{self.server_url}/api/agents/task",
                    json={
                        "description": f"Create a {language} function named {name}. {description}",
                        "role": "code_writer",
                    },
                    timeout=60,
                )

                if response.status_code == 200:
                    result = response.json()
                    code = result.get("result", result.get("description", ""))

                    if "```" in code:
                        code = self._extract_code_from_markdown(code)

                    console.print("\n[green]Generated code:[/green]\n")
                    console.print(Syntax(code, language, theme="monokai"))

                    if Confirm.ask("\nSave to file?"):
                        ext = {"python": ".py", "typescript": ".ts", "javascript": ".js"}
                        filename = Prompt.ask("Filename", default=f"{name}{ext[language]}")
                        Path(filename).write_text(code)
                        console.print(f"[green]✓[/green] Saved to: {filename}")

            except Exception as e:
                console.print(f"[red]Error:[/red] {str(e)}")

    def _create_component(self, name):
        """Create a React/Vue component"""
        framework = Prompt.ask("Framework", choices=["react", "vue"], default="react")
        description = Prompt.ask("Describe the component", default=f"A {name} component")

        with console.status(f"[cyan]Generating {framework} component with THAU...[/cyan]"):
            try:
                response = requests.post(
                    f"{self.server_url}/api/agents/task",
                    json={
                        "description": f"Create a {framework} component named {name}. {description}. Use TypeScript and modern best practices.",
                        "role": "code_writer",
                    },
                    timeout=60,
                )

                if response.status_code == 200:
                    result = response.json()
                    code = result.get("result", result.get("description", ""))

                    if "```" in code:
                        code = self._extract_code_from_markdown(code)

                    console.print("\n[green]Generated component:[/green]\n")
                    console.print(Syntax(code, "tsx", theme="monokai"))

                    if Confirm.ask("\nSave to file?"):
                        ext = ".tsx" if framework == "react" else ".vue"
                        filename = Prompt.ask("Filename", default=f"{name}{ext}")
                        Path(filename).write_text(code)
                        console.print(f"[green]✓[/green] Saved to: {filename}")

            except Exception as e:
                console.print(f"[red]Error:[/red] {str(e)}")

    def _create_api(self, name):
        """Create an API endpoint"""
        framework = Prompt.ask("Framework", choices=["fastapi", "express", "flask"], default="fastapi")
        description = Prompt.ask("Describe the API endpoint")

        with console.status(f"[cyan]Generating {framework} API with THAU...[/cyan]"):
            try:
                response = requests.post(
                    f"{self.server_url}/api/agents/task",
                    json={
                        "description": f"Create a {framework} API endpoint named {name}. {description}. Include proper error handling and validation.",
                        "role": "code_writer",
                    },
                    timeout=60,
                )

                if response.status_code == 200:
                    result = response.json()
                    code = result.get("result", result.get("description", ""))

                    if "```" in code:
                        code = self._extract_code_from_markdown(code)

                    lang = "python" if framework in ["fastapi", "flask"] else "javascript"
                    console.print("\n[green]Generated API:[/green]\n")
                    console.print(Syntax(code, lang, theme="monokai"))

                    if Confirm.ask("\nSave to file?"):
                        ext = ".py" if framework in ["fastapi", "flask"] else ".js"
                        filename = Prompt.ask("Filename", default=f"{name}{ext}")
                        Path(filename).write_text(code)
                        console.print(f"[green]✓[/green] Saved to: {filename}")

            except Exception as e:
                console.print(f"[red]Error:[/red] {str(e)}")

    def _create_model(self, name):
        """Create a data model"""
        framework = Prompt.ask("Framework", choices=["sqlalchemy", "django", "prisma"], default="sqlalchemy")
        description = Prompt.ask("Describe the model fields")

        with console.status(f"[cyan]Generating {framework} model with THAU...[/cyan]"):
            try:
                response = requests.post(
                    f"{self.server_url}/api/agents/task",
                    json={
                        "description": f"Create a {framework} model named {name}. Fields: {description}",
                        "role": "code_writer",
                    },
                    timeout=60,
                )

                if response.status_code == 200:
                    result = response.json()
                    code = result.get("result", result.get("description", ""))

                    if "```" in code:
                        code = self._extract_code_from_markdown(code)

                    console.print("\n[green]Generated model:[/green]\n")
                    console.print(Syntax(code, "python", theme="monokai"))

                    if Confirm.ask("\nSave to file?"):
                        filename = Prompt.ask("Filename", default=f"{name.lower()}.py")
                        Path(filename).write_text(code)
                        console.print(f"[green]✓[/green] Saved to: {filename}")

            except Exception as e:
                console.print(f"[red]Error:[/red] {str(e)}")

    def _create_test(self, name):
        """Create a test file"""
        framework = Prompt.ask("Test framework", choices=["pytest", "jest", "unittest"], default="pytest")
        description = Prompt.ask("What to test")

        with console.status(f"[cyan]Generating {framework} tests with THAU...[/cyan]"):
            try:
                response = requests.post(
                    f"{self.server_url}/api/agents/task",
                    json={
                        "description": f"Create {framework} tests for {name}. Test: {description}",
                        "role": "test_writer",
                    },
                    timeout=60,
                )

                if response.status_code == 200:
                    result = response.json()
                    code = result.get("result", result.get("description", ""))

                    if "```" in code:
                        code = self._extract_code_from_markdown(code)

                    lang = "python" if framework in ["pytest", "unittest"] else "javascript"
                    console.print("\n[green]Generated tests:[/green]\n")
                    console.print(Syntax(code, lang, theme="monokai"))

                    if Confirm.ask("\nSave to file?"):
                        ext = ".py" if framework in ["pytest", "unittest"] else ".test.js"
                        filename = Prompt.ask("Filename", default=f"test_{name}{ext}")
                        Path(filename).write_text(code)
                        console.print(f"[green]✓[/green] Saved to: {filename}")

            except Exception as e:
                console.print(f"[red]Error:[/red] {str(e)}")

    def _create_route(self, name):
        """Create a route/controller"""
        self._create_api(name)

    def _create_service(self, name):
        """Create a service class"""
        description = Prompt.ask("Describe the service")

        with console.status("[cyan]Generating service with THAU...[/cyan]"):
            try:
                response = requests.post(
                    f"{self.server_url}/api/agents/task",
                    json={
                        "description": f"Create a service class named {name}Service. {description}. Follow SOLID principles and use dependency injection.",
                        "role": "architect",
                    },
                    timeout=60,
                )

                if response.status_code == 200:
                    result = response.json()
                    code = result.get("result", result.get("description", ""))

                    if "```" in code:
                        code = self._extract_code_from_markdown(code)

                    console.print("\n[green]Generated service:[/green]\n")
                    console.print(Syntax(code, "python", theme="monokai"))

                    if Confirm.ask("\nSave to file?"):
                        filename = Prompt.ask("Filename", default=f"{name.lower()}_service.py")
                        Path(filename).write_text(code)
                        console.print(f"[green]✓[/green] Saved to: {filename}")

            except Exception as e:
                console.print(f"[red]Error:[/red] {str(e)}")

    def _extract_code_from_markdown(self, text):
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
    cmd = CreateCommand()
    cmd.run("class", "User")
