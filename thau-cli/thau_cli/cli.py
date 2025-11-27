#!/usr/bin/env python3
"""
THAU CODE - AI-Powered Coding Assistant CLI

Like Claude Code, but powered by THAU AI.

Commands:
    thau init              - Initialize new project
    thau code              - Interactive coding mode
    thau create <type>     - Create files/directories
    thau plan <task>       - Plan complex task
    thau exec <plan>       - Execute multi-step plan
    thau review            - Review code
    thau refactor <file>   - Refactor code
    thau test              - Generate tests
    thau deploy            - Deploy project
"""

import click
import requests
import json
import sys
import os
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.table import Table
from rich import print as rprint
from rich.markdown import Markdown
from rich.syntax import Syntax
from rich.progress import Progress, SpinnerColumn, TextColumn
import yaml

console = Console()

# Configuration
DEFAULT_CONFIG = {
    "server_url": "http://localhost:8001",
    "default_agent": "code_writer",
    "theme": "monokai",
    "auto_save": True,
    "auto_format": True,
}

CONFIG_FILE = Path.home() / ".thau" / "config.yaml"


def load_config():
    """Load configuration"""
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE, "r") as f:
            return yaml.safe_load(f) or DEFAULT_CONFIG
    return DEFAULT_CONFIG.copy()


def save_config(config):
    """Save configuration"""
    CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_FILE, "w") as f:
        yaml.dump(config, f)


class ThauClient:
    """THAU AI Client"""

    def __init__(self, server_url):
        self.server_url = server_url

    def health_check(self):
        try:
            response = requests.get(f"{self.server_url}/health", timeout=2)
            return response.status_code == 200
        except:
            return False

    def send_task(self, description, agent_role="code_writer"):
        try:
            response = requests.post(
                f"{self.server_url}/api/agents/task",
                json={"description": description, "role": agent_role},
                timeout=60
            )
            return response.json()
        except Exception as e:
            return {"error": str(e)}

    def create_plan(self, task_description):
        try:
            response = requests.post(
                f"{self.server_url}/api/planner/create",
                json={"task_description": task_description},
                timeout=60
            )
            return response.json()
        except Exception as e:
            return {"error": str(e)}


@click.group(invoke_without_command=True)
@click.pass_context
def main(ctx):
    """
    🧠 THAU CODE - AI-Powered Coding Assistant

    Like Claude Code, but powered by THAU AI
    """
    if ctx.invoked_subcommand is None:
        show_welcome()


def show_welcome():
    """Show welcome screen"""
    console.print(Panel(
        "[bold cyan]🧠 THAU CODE[/bold cyan]\n\n"
        "AI-Powered Coding Assistant\n\n"
        "[dim]Commands:[/dim]\n"
        "  thau init              - Initialize new project\n"
        "  thau code              - Interactive coding mode\n"
        "  thau create <type>     - Create files/directories\n"
        "  thau plan <task>       - Plan complex task\n"
        "  thau review            - Review code\n"
        "  thau refactor <file>   - Refactor code\n"
        "  thau test              - Generate tests\n"
        "  thau --help            - Show all commands",
        border_style="cyan",
        expand=False
    ))


@main.command()
@click.option("--name", "-n", help="Project name")
@click.option("--template", "-t", help="Template (python, react, fastapi, etc.)")
def init(name, template):
    """Initialize a new project"""
    from thau_cli.commands.init import InitCommand

    cmd = InitCommand()
    cmd.run(name, template)


@main.command()
def code():
    """Interactive coding mode (like Claude Code)"""
    from thau_cli.commands.code import CodeCommand

    cmd = CodeCommand()
    cmd.run()


@main.command()
@click.argument("type", required=True)
@click.argument("name", required=True)
def create(type, name):
    """Create files/directories (file, class, component, api, etc.)"""
    from thau_cli.commands.create import CreateCommand

    cmd = CreateCommand()
    cmd.run(type, name)


@main.command()
@click.argument("task_description", required=False)
def plan(task_description):
    """Create a detailed plan for a task"""
    from thau_cli.commands.plan import PlanCommand

    cmd = PlanCommand()
    cmd.run(task_description)


@main.command()
@click.argument("plan_file", required=False)
def exec(plan_file):
    """Execute a multi-step plan"""
    from thau_cli.commands.exec import ExecCommand

    cmd = ExecCommand()
    cmd.run(plan_file)


@main.command()
@click.argument("files", nargs=-1)
def review(files):
    """Review code for bugs and improvements"""
    from thau_cli.commands.review import ReviewCommand

    cmd = ReviewCommand()
    cmd.run(list(files))


@main.command()
@click.argument("file_path", required=True)
def refactor(file_path):
    """Refactor code"""
    from thau_cli.commands.refactor import RefactorCommand

    cmd = RefactorCommand()
    cmd.run(file_path)


@main.command()
@click.argument("files", nargs=-1)
def test(files):
    """Generate tests for code"""
    from thau_cli.commands.test import TestCommand

    cmd = TestCommand()
    cmd.run(list(files))


@main.command()
@click.option("--env", "-e", default="production", help="Environment")
def deploy(env):
    """Deploy project"""
    from thau_cli.commands.deploy import DeployCommand

    cmd = DeployCommand()
    cmd.run(env)


@main.command()
@click.argument("message", required=False)
@click.option("--agent", "-a", help="Agent to use")
@click.option("--file", "-f", type=click.Path(exists=True), help="File to analyze")
def chat(message, agent, file):
    """Chat with THAU AI"""
    config = load_config()
    client = ThauClient(config["server_url"])

    if not client.health_check():
        console.print("[red]Error: THAU server not running[/red]")
        console.print("[yellow]Start with:[/yellow] python api/thau_code_server.py")
        sys.exit(1)

    if file:
        with open(file, "r") as f:
            content = f.read()
        message = f"{message}\n\nFile: {file}\n```\n{content}\n```" if message else f"Analyze:\n```\n{content}\n```"

    agent_role = agent or config["default_agent"]

    with Progress(SpinnerColumn(), TextColumn("[cyan]{task.description}"), console=console) as progress:
        task = progress.add_task(f"Thinking with {agent_role}...", total=None)
        response = client.send_task(message, agent_role)

    if "error" in response:
        console.print(f"[red]Error:[/red] {response['error']}")
    else:
        console.print(Panel(
            response.get('description', 'Task created'),
            title=f"🤖 {agent_role.upper()}",
            border_style="cyan"
        ))


@main.command()
@click.option("--server", help="Set server URL")
@click.option("--agent", help="Set default agent")
@click.option("--show", is_flag=True, help="Show configuration")
def config(server, agent, show):
    """Configure THAU CODE"""
    cfg = load_config()

    if show:
        console.print(Panel(
            f"[cyan]Server URL:[/cyan] {cfg['server_url']}\n"
            f"[cyan]Default Agent:[/cyan] {cfg['default_agent']}\n"
            f"[cyan]Theme:[/cyan] {cfg['theme']}\n"
            f"[cyan]Auto Save:[/cyan] {cfg['auto_save']}\n"
            f"[cyan]Auto Format:[/cyan] {cfg['auto_format']}",
            title="⚙️ THAU CODE Configuration",
            border_style="blue"
        ))
        return

    if server:
        cfg["server_url"] = server
        console.print(f"[green]✓[/green] Server URL: {server}")

    if agent:
        cfg["default_agent"] = agent
        console.print(f"[green]✓[/green] Default agent: {agent}")

    save_config(cfg)


if __name__ == "__main__":
    main()
