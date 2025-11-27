#!/usr/bin/env python3
"""
THAU EXEC - Execute multi-step plans

Execute plans created by the planner
"""

import sys
import json
import requests
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich import print as rprint

console = Console()


class ExecCommand:
    """Exec command - execute multi-step plans"""

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

    def run(self, plan_file=None):
        """Run exec command"""
        console.print(Panel(
            "[bold cyan]⚡ THAU Executor[/bold cyan]\n\n"
            "[dim]Execute multi-step plans[/dim]",
            border_style="cyan"
        ))

        # Get plan file
        if not plan_file:
            plan_file = Prompt.ask("\n[cyan]Plan file (JSON)[/cyan]", default="plan.json")

        plan_path = Path(plan_file)
        if not plan_path.exists():
            console.print(f"[red]Error:[/red] File not found: {plan_file}")
            sys.exit(1)

        # Load plan
        try:
            with open(plan_path, "r") as f:
                plan = json.load(f)
        except Exception as e:
            console.print(f"[red]Error loading plan:[/red] {str(e)}")
            sys.exit(1)

        # Display plan summary
        console.print(f"\n[bold]Plan:[/bold] {plan.get('task_description', 'N/A')}")
        steps = plan.get("steps", [])
        console.print(f"[bold]Steps:[/bold] {len(steps)}\n")

        if not Confirm.ask("Execute this plan?", default=True):
            console.print("[yellow]Cancelled[/yellow]")
            return

        # Execute plan
        self._execute_plan(plan)

    def _execute_plan(self, plan):
        """Execute the plan step by step"""
        steps = plan.get("steps", [])
        results = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            console=console
        ) as progress:
            task = progress.add_task("Executing plan...", total=len(steps))

            for i, step in enumerate(steps, 1):
                if isinstance(step, dict):
                    description = step.get("description", str(step))
                else:
                    description = str(step)

                progress.update(task, description=f"Step {i}/{len(steps)}: {description}")

                # Execute step
                result = self._execute_step(description, i)
                results.append({
                    "step": i,
                    "description": description,
                    "result": result
                })

                progress.advance(task)

        # Display results
        self._display_results(results)

    def _execute_step(self, description, step_number):
        """Execute a single step"""
        try:
            response = requests.post(
                f"{self.server_url}/api/agents/task",
                json={
                    "description": f"Execute step {step_number}: {description}",
                    "role": "code_writer",
                },
                timeout=120,
            )

            if response.status_code == 200:
                result = response.json()
                return result.get("result", result.get("description", "Completed"))
            else:
                return f"Error: Server returned {response.status_code}"

        except Exception as e:
            return f"Error: {str(e)}"

    def _display_results(self, results):
        """Display execution results"""
        console.print("\n" + "=" * 70)
        console.print("[bold cyan]📊 EXECUTION RESULTS[/bold cyan]")
        console.print("=" * 70 + "\n")

        for result in results:
            console.print(f"\n[bold cyan]Step {result['step']}:[/bold cyan] {result['description']}")
            console.print(Panel(
                result.get("result", "N/A"),
                border_style="green" if "error" not in result.get("result", "").lower() else "red"
            ))

        console.print("\n[green]✓ Plan execution completed![/green]\n")


if __name__ == "__main__":
    cmd = ExecCommand()
    cmd.run()
