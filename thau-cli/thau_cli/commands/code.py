#!/usr/bin/env python3
"""
THAU CODE - Interactive Coding Mode

Like Claude Code interactive mode - real-time conversation with THAU AI
"""

import sys
import os
import requests
import websocket
import json
import threading
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.markdown import Markdown
from rich.syntax import Syntax
from rich.live import Live
from rich.spinner import Spinner
from rich import print as rprint
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory

console = Console()


class CodeCommand:
    """Interactive coding mode - like Claude Code"""

    def __init__(self):
        self.config = self._load_config()
        self.server_url = self.config.get("server_url", "http://localhost:8001")
        self.session = None
        self.conversation_history = []
        self.ws = None
        self.current_agent = self.config.get("default_agent", "code_writer")

    def _load_config(self):
        """Load configuration"""
        from pathlib import Path
        import yaml

        config_file = Path.home() / ".thau" / "config.yaml"
        if config_file.exists():
            with open(config_file, "r") as f:
                return yaml.safe_load(f) or {}
        return {}

    def _check_server(self):
        """Check if THAU server is running"""
        try:
            response = requests.get(f"{self.server_url}/health", timeout=2)
            return response.status_code == 200
        except:
            return False

    def run(self):
        """Run interactive coding mode"""
        # Check server
        if not self._check_server():
            console.print("[red]❌ Error: THAU server not running[/red]")
            console.print("[yellow]Start server with:[/yellow] python api/thau_code_server.py")
            sys.exit(1)

        # Welcome screen
        self._show_welcome()

        # Create prompt session with history
        history_file = Path.home() / ".thau" / "history.txt"
        history_file.parent.mkdir(parents=True, exist_ok=True)

        session = PromptSession(
            history=FileHistory(str(history_file)),
            auto_suggest=AutoSuggestFromHistory(),
        )

        # Interactive loop
        console.print("[dim]Type your message or use commands:[/dim]")
        console.print("[dim]  /help    - Show help[/dim]")
        console.print("[dim]  /agent   - Switch agent[/dim]")
        console.print("[dim]  /clear   - Clear history[/dim]")
        console.print("[dim]  /exit    - Exit[/dim]")
        console.print()

        while True:
            try:
                # Get user input
                user_input = session.prompt("You: ", multiline=False)

                if not user_input.strip():
                    continue

                # Handle commands
                if user_input.startswith("/"):
                    self._handle_command(user_input)
                    continue

                # Send to THAU
                self._send_message(user_input)

            except KeyboardInterrupt:
                console.print("\n[yellow]Use /exit to quit[/yellow]")
                continue
            except EOFError:
                break

        console.print("\n[cyan]👋 Goodbye![/cyan]")

    def _show_welcome(self):
        """Show welcome screen"""
        console.print(
            Panel(
                "[bold cyan]🧠 THAU CODE - Interactive Mode[/bold cyan]\n\n"
                "[dim]Like Claude Code, but powered by THAU AI[/dim]\n\n"
                f"[cyan]Agent:[/cyan] {self.current_agent}\n"
                f"[cyan]Server:[/cyan] {self.server_url}\n\n"
                "[dim]Commands:[/dim]\n"
                "  /help    - Show help\n"
                "  /agent   - Switch agent\n"
                "  /clear   - Clear history\n"
                "  /exit    - Exit",
                border_style="cyan",
                expand=False,
            )
        )
        console.print()

    def _handle_command(self, command):
        """Handle slash commands"""
        cmd = command.lower().strip()

        if cmd == "/exit" or cmd == "/quit":
            console.print("[cyan]👋 Goodbye![/cyan]")
            sys.exit(0)

        elif cmd == "/help":
            self._show_help()

        elif cmd == "/agent":
            self._switch_agent()

        elif cmd == "/clear":
            self.conversation_history = []
            console.print("[green]✓[/green] Conversation history cleared")

        elif cmd == "/history":
            self._show_history()

        else:
            console.print(f"[red]Unknown command:[/red] {command}")
            console.print("[dim]Type /help for available commands[/dim]")

    def _show_help(self):
        """Show help"""
        console.print(
            Panel(
                "[bold]THAU CODE Commands[/bold]\n\n"
                "[cyan]/help[/cyan]     - Show this help\n"
                "[cyan]/agent[/cyan]    - Switch agent (code_writer, planner, reviewer, etc.)\n"
                "[cyan]/clear[/cyan]    - Clear conversation history\n"
                "[cyan]/history[/cyan]  - Show conversation history\n"
                "[cyan]/exit[/cyan]     - Exit interactive mode\n\n"
                "[bold]Usage Examples:[/bold]\n"
                "  Create a Python function to calculate fibonacci\n"
                "  Refactor this code to use dependency injection\n"
                "  Review the UserService class for bugs\n"
                "  Plan a REST API with authentication",
                title="💡 Help",
                border_style="blue",
            )
        )

    def _switch_agent(self):
        """Switch agent"""
        agents = [
            "general",
            "code_writer",
            "planner",
            "code_reviewer",
            "debugger",
            "architect",
            "test_writer",
            "refactorer",
            "explainer",
            "optimizer",
            "security",
        ]

        console.print("\n[cyan]Available agents:[/cyan]")
        for i, agent in enumerate(agents, 1):
            marker = "→" if agent == self.current_agent else " "
            console.print(f"  {marker} {i}. {agent}")

        choice = Prompt.ask(
            "\nSelect agent", choices=[str(i) for i in range(1, len(agents) + 1)]
        )
        self.current_agent = agents[int(choice) - 1]
        console.print(f"[green]✓[/green] Switched to: {self.current_agent}\n")

    def _show_history(self):
        """Show conversation history"""
        if not self.conversation_history:
            console.print("[yellow]No conversation history[/yellow]")
            return

        console.print(Panel("[bold]Conversation History[/bold]", border_style="blue"))
        for i, msg in enumerate(self.conversation_history, 1):
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            if role == "user":
                console.print(f"\n[cyan]You ({i}):[/cyan]")
                console.print(content[:200] + "..." if len(content) > 200 else content)
            elif role == "assistant":
                console.print(f"\n[green]THAU ({i}):[/green]")
                console.print(content[:200] + "..." if len(content) > 200 else content)

    def _send_message(self, message):
        """Send message to THAU and get response"""
        # Add to history
        self.conversation_history.append({"role": "user", "content": message})

        # Show spinner while waiting
        with console.status(
            f"[cyan]🧠 {self.current_agent} thinking...[/cyan]", spinner="dots"
        ):
            try:
                # Send request to THAU
                response = requests.post(
                    f"{self.server_url}/api/agents/task",
                    json={
                        "description": message,
                        "role": self.current_agent,
                        "context": self.conversation_history[-5:],  # Last 5 messages
                    },
                    timeout=120,
                )

                if response.status_code == 200:
                    result = response.json()
                    answer = result.get("result", result.get("description", "No response"))

                    # Add to history
                    self.conversation_history.append(
                        {"role": "assistant", "content": answer}
                    )

                    # Display response
                    console.print(f"\n[green]🤖 {self.current_agent}:[/green]")

                    # Try to render as markdown if it looks like it
                    if "```" in answer or "#" in answer or "*" in answer:
                        console.print(Markdown(answer))
                    else:
                        console.print(answer)

                    console.print()

                else:
                    console.print(
                        f"[red]Error:[/red] Server returned {response.status_code}"
                    )

            except requests.Timeout:
                console.print("[red]Error:[/red] Request timed out")
            except Exception as e:
                console.print(f"[red]Error:[/red] {str(e)}")


if __name__ == "__main__":
    cmd = CodeCommand()
    cmd.run()
