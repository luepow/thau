#!/usr/bin/env python3
"""
THAU DEPLOY - Deploy projects
"""

import sys
import subprocess
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich import print as rprint

console = Console()


class DeployCommand:
    """Deploy command - deploy projects"""

    def __init__(self):
        self.config = self._load_config()
        self.deployers = {
            "heroku": self._deploy_heroku,
            "vercel": self._deploy_vercel,
            "netlify": self._deploy_netlify,
            "docker": self._deploy_docker,
            "aws": self._deploy_aws,
            "gcp": self._deploy_gcp,
        }

    def _load_config(self):
        """Load configuration"""
        import yaml
        config_file = Path.home() / ".thau" / "config.yaml"
        if config_file.exists():
            with open(config_file, "r") as f:
                return yaml.safe_load(f) or {}
        return {}

    def run(self, env="production"):
        """Run deploy command"""
        console.print(Panel(
            "[bold cyan]🚀 THAU Deploy[/bold cyan]\n\n"
            "[dim]Deploy your project to the cloud[/dim]",
            border_style="cyan"
        ))

        # Select deployment platform
        platform = Prompt.ask(
            "\n[cyan]Select platform[/cyan]",
            choices=list(self.deployers.keys()),
            default="docker"
        )

        console.print(f"\n[cyan]Deploying to {platform} ({env} environment)[/cyan]\n")

        if not Confirm.ask("Continue with deployment?", default=True):
            console.print("[yellow]Cancelled[/yellow]")
            return

        # Deploy
        try:
            self.deployers[platform](env)
        except Exception as e:
            console.print(f"[red]Error:[/red] {str(e)}")
            sys.exit(1)

    def _deploy_heroku(self, env):
        """Deploy to Heroku"""
        console.print("[cyan]Deploying to Heroku...[/cyan]\n")

        # Check if heroku CLI is installed
        try:
            subprocess.run(["heroku", "--version"], check=True, capture_output=True)
        except:
            console.print("[red]Heroku CLI not installed[/red]")
            console.print("Install: https://devcenter.heroku.com/articles/heroku-cli")
            return

        app_name = Prompt.ask("App name")

        with Progress(SpinnerColumn(), TextColumn("[cyan]{task.description}[/cyan]")) as progress:
            task = progress.add_task("Deploying to Heroku...", total=None)

            # Create app if doesn't exist
            subprocess.run(["heroku", "create", app_name], capture_output=True)

            # Push to heroku
            subprocess.run(["git", "push", "heroku", "main"], check=True)

            progress.update(task, description="Deployed successfully!")

        console.print(f"\n[green]✓ Deployed to:[/green] https://{app_name}.herokuapp.com")

    def _deploy_vercel(self, env):
        """Deploy to Vercel"""
        console.print("[cyan]Deploying to Vercel...[/cyan]\n")

        try:
            subprocess.run(["vercel", "--version"], check=True, capture_output=True)
        except:
            console.print("[red]Vercel CLI not installed[/red]")
            console.print("Install: npm i -g vercel")
            return

        prod_flag = "--prod" if env == "production" else ""

        with Progress(SpinnerColumn(), TextColumn("[cyan]{task.description}[/cyan]")) as progress:
            task = progress.add_task("Deploying to Vercel...", total=None)
            subprocess.run(f"vercel {prod_flag}".split(), check=True)
            progress.update(task, description="Deployed successfully!")

        console.print("\n[green]✓ Deployed to Vercel[/green]")

    def _deploy_netlify(self, env):
        """Deploy to Netlify"""
        console.print("[cyan]Deploying to Netlify...[/cyan]\n")

        try:
            subprocess.run(["netlify", "--version"], check=True, capture_output=True)
        except:
            console.print("[red]Netlify CLI not installed[/red]")
            console.print("Install: npm i -g netlify-cli")
            return

        prod_flag = "--prod" if env == "production" else ""

        with Progress(SpinnerColumn(), TextColumn("[cyan]{task.description}[/cyan]")) as progress:
            task = progress.add_task("Deploying to Netlify...", total=None)
            subprocess.run(f"netlify deploy {prod_flag}".split(), check=True)
            progress.update(task, description="Deployed successfully!")

        console.print("\n[green]✓ Deployed to Netlify[/green]")

    def _deploy_docker(self, env):
        """Deploy with Docker"""
        console.print("[cyan]Building Docker image...[/cyan]\n")

        # Check if Dockerfile exists
        if not Path("Dockerfile").exists():
            console.print("[yellow]No Dockerfile found. Creating one...[/yellow]")
            self._create_dockerfile()

        image_name = Prompt.ask("Image name", default="my-app")
        tag = Prompt.ask("Tag", default="latest")

        with Progress(SpinnerColumn(), TextColumn("[cyan]{task.description}[/cyan]")) as progress:
            # Build
            task = progress.add_task("Building Docker image...", total=None)
            subprocess.run(["docker", "build", "-t", f"{image_name}:{tag}", "."], check=True)
            progress.update(task, description="Built successfully!")

        console.print(f"\n[green]✓ Docker image built:[/green] {image_name}:{tag}")
        console.print(f"[dim]Run with:[/dim] docker run -p 8000:8000 {image_name}:{tag}")

        # Ask to push to registry
        if Confirm.ask("\nPush to Docker registry?"):
            registry = Prompt.ask("Registry", default="docker.io")
            subprocess.run(["docker", "tag", f"{image_name}:{tag}", f"{registry}/{image_name}:{tag}"])
            subprocess.run(["docker", "push", f"{registry}/{image_name}:{tag}"], check=True)
            console.print(f"[green]✓ Pushed to:[/green] {registry}/{image_name}:{tag}")

    def _deploy_aws(self, env):
        """Deploy to AWS"""
        console.print("[yellow]AWS deployment coming soon...[/yellow]")
        console.print("Use Docker deployment and push to ECR for now")

    def _deploy_gcp(self, env):
        """Deploy to GCP"""
        console.print("[yellow]GCP deployment coming soon...[/yellow]")
        console.print("Use Docker deployment and push to GCR for now")

    def _create_dockerfile(self):
        """Create a basic Dockerfile"""
        # Detect project type
        if Path("requirements.txt").exists():
            dockerfile = '''FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "main.py"]
'''
        elif Path("package.json").exists():
            dockerfile = '''FROM node:18-alpine

WORKDIR /app

COPY package*.json ./
RUN npm ci --only=production

COPY . .

CMD ["npm", "start"]
'''
        else:
            console.print("[red]Could not detect project type[/red]")
            return

        Path("Dockerfile").write_text(dockerfile)
        console.print("[green]✓ Created Dockerfile[/green]")


if __name__ == "__main__":
    cmd = DeployCommand()
    cmd.run()
