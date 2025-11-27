#!/usr/bin/env python3
"""
THAU INIT - Initialize new projects

Initialize projects with templates for different frameworks
"""

import os
import sys
import subprocess
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.table import Table
from rich import print as rprint

console = Console()


class InitCommand:
    """Initialize command - create new projects"""

    def __init__(self):
        self.templates = {
            "python": self._init_python,
            "fastapi": self._init_fastapi,
            "flask": self._init_flask,
            "django": self._init_django,
            "react": self._init_react,
            "nextjs": self._init_nextjs,
            "vue": self._init_vue,
            "node": self._init_node,
            "express": self._init_express,
        }

    def run(self, name=None, template=None):
        """Run init command"""
        console.print(Panel(
            "[bold cyan]🚀 THAU Project Initialization[/bold cyan]\n\n"
            "[dim]Create a new project with templates[/dim]",
            border_style="cyan"
        ))

        # Get project name
        if not name:
            name = Prompt.ask("\n[cyan]Project name[/cyan]")

        # Get template
        if not template:
            template = self._select_template()

        # Create project
        console.print(f"\n[cyan]Creating {template} project: {name}[/cyan]\n")

        if template not in self.templates:
            console.print(f"[red]Unknown template:[/red] {template}")
            sys.exit(1)

        try:
            self.templates[template](name)
            self._show_next_steps(name, template)
        except Exception as e:
            console.print(f"[red]Error:[/red] {str(e)}")
            sys.exit(1)

    def _select_template(self):
        """Select template interactively"""
        table = Table(title="Available Templates")
        table.add_column("Type", style="cyan")
        table.add_column("Description", style="white")

        templates_info = [
            ("python", "Python project with virtual environment"),
            ("fastapi", "FastAPI web application"),
            ("flask", "Flask web application"),
            ("django", "Django web application"),
            ("react", "React application with TypeScript"),
            ("nextjs", "Next.js application"),
            ("vue", "Vue.js application"),
            ("node", "Node.js project"),
            ("express", "Express.js API"),
        ]

        for type_name, desc in templates_info:
            table.add_row(type_name, desc)

        console.print(table)
        console.print()

        return Prompt.ask(
            "Select template",
            choices=list(self.templates.keys()),
            default="python"
        )

    def _init_python(self, name):
        """Initialize Python project"""
        project_dir = Path(name)
        project_dir.mkdir(exist_ok=True)

        # Create directory structure
        (project_dir / "src").mkdir(exist_ok=True)
        (project_dir / "tests").mkdir(exist_ok=True)
        (project_dir / "docs").mkdir(exist_ok=True)

        # Create files
        (project_dir / "README.md").write_text(f"# {name}\n\nPython project created with THAU\n")
        (project_dir / "requirements.txt").write_text("# Python dependencies\n")
        (project_dir / ".gitignore").write_text("venv/\n__pycache__/\n*.pyc\n.env\n")
        (project_dir / "src" / "__init__.py").write_text("")
        (project_dir / "src" / "main.py").write_text(
            'def main():\n    print("Hello from THAU!")\n\n'
            'if __name__ == "__main__":\n    main()\n'
        )

        console.print("[green]✓[/green] Created Python project structure")

        # Create virtual environment
        if Confirm.ask("Create virtual environment?", default=True):
            subprocess.run(["python3", "-m", "venv", str(project_dir / "venv")])
            console.print("[green]✓[/green] Created virtual environment")

    def _init_fastapi(self, name):
        """Initialize FastAPI project"""
        project_dir = Path(name)
        project_dir.mkdir(exist_ok=True)

        # Create structure
        (project_dir / "app").mkdir(exist_ok=True)
        (project_dir / "app" / "routers").mkdir(exist_ok=True)
        (project_dir / "tests").mkdir(exist_ok=True)

        # Create files
        (project_dir / "README.md").write_text(f"# {name}\n\nFastAPI application created with THAU\n")
        (project_dir / "requirements.txt").write_text(
            "fastapi>=0.100.0\nuvicorn[standard]>=0.23.0\npydantic>=2.0.0\n"
        )
        (project_dir / ".gitignore").write_text("venv/\n__pycache__/\n*.pyc\n.env\n")

        # Create main.py
        (project_dir / "app" / "__init__.py").write_text("")
        (project_dir / "app" / "main.py").write_text('''from fastapi import FastAPI

app = FastAPI(title="''' + name + '''")

@app.get("/")
async def root():
    return {"message": "Hello from THAU!"}

@app.get("/health")
async def health():
    return {"status": "ok"}
''')

        console.print("[green]✓[/green] Created FastAPI project structure")

    def _init_flask(self, name):
        """Initialize Flask project"""
        project_dir = Path(name)
        project_dir.mkdir(exist_ok=True)

        # Create structure
        (project_dir / "app").mkdir(exist_ok=True)
        (project_dir / "app" / "templates").mkdir(exist_ok=True)
        (project_dir / "app" / "static").mkdir(exist_ok=True)

        # Create files
        (project_dir / "README.md").write_text(f"# {name}\n\nFlask application created with THAU\n")
        (project_dir / "requirements.txt").write_text("Flask>=2.3.0\n")
        (project_dir / ".gitignore").write_text("venv/\n__pycache__/\n*.pyc\n.env\n")

        (project_dir / "app" / "__init__.py").write_text("""from flask import Flask

def create_app():
    app = Flask(__name__)

    @app.route('/')
    def index():
        return {"message": "Hello from THAU!"}

    return app
""")

        (project_dir / "run.py").write_text("""from app import create_app

app = create_app()

if __name__ == '__main__':
    app.run(debug=True)
""")

        console.print("[green]✓[/green] Created Flask project structure")

    def _init_django(self, name):
        """Initialize Django project"""
        try:
            subprocess.run(["django-admin", "startproject", name], check=True)
            console.print("[green]✓[/green] Created Django project")
        except subprocess.CalledProcessError:
            console.print("[yellow]Django not installed. Install with: pip install django[/yellow]")
            self._init_python(name)

    def _init_react(self, name):
        """Initialize React project"""
        try:
            subprocess.run(["npx", "create-react-app", name, "--template", "typescript"], check=True)
            console.print("[green]✓[/green] Created React project with TypeScript")
        except subprocess.CalledProcessError as e:
            console.print(f"[red]Error creating React app:[/red] {str(e)}")
            console.print("[yellow]Make sure Node.js and npm are installed[/yellow]")

    def _init_nextjs(self, name):
        """Initialize Next.js project"""
        try:
            subprocess.run(["npx", "create-next-app@latest", name, "--typescript", "--tailwind", "--app"], check=True)
            console.print("[green]✓[/green] Created Next.js project")
        except subprocess.CalledProcessError as e:
            console.print(f"[red]Error creating Next.js app:[/red] {str(e)}")

    def _init_vue(self, name):
        """Initialize Vue project"""
        try:
            subprocess.run(["npm", "create", "vue@latest", name], check=True)
            console.print("[green]✓[/green] Created Vue project")
        except subprocess.CalledProcessError as e:
            console.print(f"[red]Error creating Vue app:[/red] {str(e)}")

    def _init_node(self, name):
        """Initialize Node.js project"""
        project_dir = Path(name)
        project_dir.mkdir(exist_ok=True)

        # Create structure
        (project_dir / "src").mkdir(exist_ok=True)
        (project_dir / "tests").mkdir(exist_ok=True)

        # Create files
        (project_dir / "README.md").write_text(f"# {name}\n\nNode.js project created with THAU\n")
        (project_dir / "package.json").write_text(f'''{{
  "name": "{name}",
  "version": "1.0.0",
  "description": "Node.js project created with THAU",
  "main": "src/index.js",
  "scripts": {{
    "start": "node src/index.js",
    "dev": "nodemon src/index.js"
  }},
  "keywords": [],
  "author": "",
  "license": "ISC"
}}
''')
        (project_dir / ".gitignore").write_text("node_modules/\n.env\n")
        (project_dir / "src" / "index.js").write_text('console.log("Hello from THAU!");\n')

        console.print("[green]✓[/green] Created Node.js project structure")

    def _init_express(self, name):
        """Initialize Express project"""
        project_dir = Path(name)
        project_dir.mkdir(exist_ok=True)

        # Create structure
        (project_dir / "src").mkdir(exist_ok=True)
        (project_dir / "src" / "routes").mkdir(exist_ok=True)

        # Create files
        (project_dir / "README.md").write_text(f"# {name}\n\nExpress.js API created with THAU\n")
        (project_dir / "package.json").write_text(f'''{{
  "name": "{name}",
  "version": "1.0.0",
  "description": "Express.js API created with THAU",
  "main": "src/index.js",
  "scripts": {{
    "start": "node src/index.js",
    "dev": "nodemon src/index.js"
  }},
  "dependencies": {{
    "express": "^4.18.0",
    "cors": "^2.8.5"
  }},
  "devDependencies": {{
    "nodemon": "^3.0.0"
  }}
}}
''')
        (project_dir / ".gitignore").write_text("node_modules/\n.env\n")
        (project_dir / "src" / "index.js").write_text('''const express = require('express');
const cors = require('cors');

const app = express();
const PORT = process.env.PORT || 3000;

app.use(cors());
app.use(express.json());

app.get('/', (req, res) => {
  res.json({ message: 'Hello from THAU!' });
});

app.get('/health', (req, res) => {
  res.json({ status: 'ok' });
});

app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});
''')

        console.print("[green]✓[/green] Created Express.js project structure")

        if Confirm.ask("Install dependencies?", default=True):
            os.chdir(project_dir)
            subprocess.run(["npm", "install"])
            console.print("[green]✓[/green] Installed dependencies")

    def _show_next_steps(self, name, template):
        """Show next steps"""
        console.print(f"\n[green]✓ Project '{name}' created successfully![/green]\n")
        console.print(Panel(
            f"[bold]Next steps:[/bold]\n\n"
            f"1. cd {name}\n"
            f"2. " + self._get_run_command(template) + "\n"
            f"3. Start coding with THAU!\n\n"
            f"[dim]Use [cyan]thau code[/cyan] for interactive mode[/dim]",
            title="🎉 Success",
            border_style="green"
        ))

    def _get_run_command(self, template):
        """Get run command for template"""
        commands = {
            "python": "python src/main.py",
            "fastapi": "uvicorn app.main:app --reload",
            "flask": "python run.py",
            "django": "python manage.py runserver",
            "react": "npm start",
            "nextjs": "npm run dev",
            "vue": "npm run dev",
            "node": "npm start",
            "express": "npm run dev",
        }
        return commands.get(template, "# Check README.md")


if __name__ == "__main__":
    cmd = InitCommand()
    cmd.run()
