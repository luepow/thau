#!/usr/bin/env python3
"""
THAU PLAN - Create detailed plans for complex tasks

Like Claude Code's planning feature
"""

import sys
import requests
import json
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.markdown import Markdown
from rich.table import Table
from rich import print as rprint

console = Console()


class PlanCommand:
    """Plan command - create detailed task plans"""

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

    def run(self, task_description=None):
        """Run plan command"""
        console.print(Panel(
            "[bold cyan]📋 THAU Planner[/bold cyan]\n\n"
            "[dim]Create detailed plans for complex tasks[/dim]",
            border_style="cyan"
        ))

        # Get task description
        if not task_description:
            console.print("\n[cyan]Describe the task you want to plan:[/cyan]")
            task_description = Prompt.ask("Task")

        # Create plan
        console.print(f"\n[cyan]Creating plan for: {task_description}[/cyan]\n")

        with console.status("[cyan]🧠 THAU Planner thinking...[/cyan]", spinner="dots"):
            try:
                response = requests.post(
                    f"{self.server_url}/api/planner/create",
                    json={"task_description": task_description},
                    timeout=120,
                )

                if response.status_code == 200:
                    result = response.json()
                    self._display_plan(result)

                    # Ask to save
                    if Confirm.ask("\n[cyan]Save plan to file?[/cyan]", default=True):
                        self._save_plan(task_description, result)

                else:
                    console.print(f"[red]Error:[/red] Server returned {response.status_code}")
                    sys.exit(1)

            except requests.Timeout:
                console.print("[red]Error:[/red] Request timed out")
                sys.exit(1)
            except Exception as e:
                console.print(f"[red]Error:[/red] {str(e)}")
                sys.exit(1)

    def _display_plan(self, plan):
        """Display plan in a nice format"""
        console.print("\n" + "=" * 70)
        console.print("[bold cyan]📋 DETAILED PLAN[/bold cyan]")
        console.print("=" * 70 + "\n")

        # Display task
        console.print(Panel(
            plan.get("task_description", "N/A"),
            title="🎯 Task",
            border_style="cyan"
        ))
        console.print()

        # Display steps
        steps = plan.get("steps", [])
        if steps:
            console.print("[bold]Steps:[/bold]\n")
            table = Table(show_header=True, header_style="bold cyan")
            table.add_column("#", style="cyan", width=4)
            table.add_column("Description", style="white")
            table.add_column("Estimated Time", style="yellow", width=15)

            for i, step in enumerate(steps, 1):
                if isinstance(step, dict):
                    desc = step.get("description", str(step))
                    time = step.get("estimated_time", "N/A")
                else:
                    desc = str(step)
                    time = "N/A"

                table.add_row(str(i), desc, time)

            console.print(table)
            console.print()

        # Display dependencies
        dependencies = plan.get("dependencies", [])
        if dependencies:
            console.print(Panel(
                "\n".join(f"• {dep}" for dep in dependencies),
                title="📦 Dependencies",
                border_style="yellow"
            ))
            console.print()

        # Display risks
        risks = plan.get("risks", [])
        if risks:
            console.print(Panel(
                "\n".join(f"⚠️  {risk}" for risk in risks),
                title="⚠️ Potential Risks",
                border_style="red"
            ))
            console.print()

        # Display estimated total time
        total_time = plan.get("total_estimated_time", "N/A")
        console.print(f"[bold]Total Estimated Time:[/bold] [yellow]{total_time}[/yellow]\n")

    def _save_plan(self, task_name, plan):
        """Save plan to file"""
        # Generate filename
        safe_name = "".join(c for c in task_name if c.isalnum() or c in (' ', '-', '_')).strip()
        safe_name = safe_name.replace(' ', '_')[:50]
        filename = Prompt.ask("Filename", default=f"plan_{safe_name}.json")

        # Save as JSON
        plan_file = Path(filename)
        with open(plan_file, "w") as f:
            json.dump(plan, f, indent=2)

        console.print(f"[green]✓[/green] Plan saved to: {filename}")

        # Also save as markdown
        if Confirm.ask("Save as markdown too?", default=True):
            md_filename = plan_file.with_suffix('.md')
            self._save_as_markdown(md_filename, task_name, plan)
            console.print(f"[green]✓[/green] Markdown saved to: {md_filename}")

    def _save_as_markdown(self, filename, task_name, plan):
        """Save plan as markdown"""
        lines = [
            f"# Plan: {task_name}",
            "",
            f"**Created by:** THAU Planner",
            "",
            "## Task Description",
            "",
            plan.get("task_description", "N/A"),
            "",
            "## Steps",
            ""
        ]

        steps = plan.get("steps", [])
        for i, step in enumerate(steps, 1):
            if isinstance(step, dict):
                desc = step.get("description", str(step))
                time = step.get("estimated_time", "N/A")
                lines.append(f"{i}. **{desc}**")
                lines.append(f"   - Estimated time: {time}")
            else:
                lines.append(f"{i}. {step}")
            lines.append("")

        dependencies = plan.get("dependencies", [])
        if dependencies:
            lines.extend([
                "## Dependencies",
                ""
            ])
            for dep in dependencies:
                lines.append(f"- {dep}")
            lines.append("")

        risks = plan.get("risks", [])
        if risks:
            lines.extend([
                "## Potential Risks",
                ""
            ])
            for risk in risks:
                lines.append(f"- ⚠️ {risk}")
            lines.append("")

        total_time = plan.get("total_estimated_time", "N/A")
        lines.extend([
            "## Summary",
            "",
            f"**Total Estimated Time:** {total_time}",
            ""
        ])

        with open(filename, "w") as f:
            f.write("\n".join(lines))


if __name__ == "__main__":
    cmd = PlanCommand()
    cmd.run()
