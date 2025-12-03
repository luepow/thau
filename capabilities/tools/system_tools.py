"""System tools for THAU Code Agent.

These tools allow THAU to interact with the file system and execute commands,
enabling it to be a full code agent that can:
- Read and write files
- Execute shell commands
- Search for files and content
- Create and manage projects
"""

import os
import subprocess
import glob as glob_module
import re
from pathlib import Path
from typing import Dict, Any, List, Optional
from loguru import logger
from dataclasses import dataclass


@dataclass
class ToolResult:
    """Result from a tool execution."""
    success: bool
    output: str
    error: Optional[str] = None


class SystemTools:
    """System tools for code agent capabilities."""

    def __init__(self, working_dir: str = "."):
        """Initialize system tools.

        Args:
            working_dir: Base working directory for operations
        """
        self.working_dir = Path(working_dir).resolve()
        self.allowed_extensions = {
            '.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.kt',
            '.dart', '.go', '.rs', '.c', '.cpp', '.h', '.hpp',
            '.html', '.css', '.scss', '.json', '.yaml', '.yml',
            '.md', '.txt', '.sh', '.bash', '.sql', '.xml',
            '.dockerfile', '.gitignore', '.env.example'
        }
        logger.info(f"SystemTools initialized with working_dir: {self.working_dir}")

    def _safe_path(self, path: str) -> Path:
        """Ensure path is within working directory.

        Args:
            path: Path to validate

        Returns:
            Resolved safe path
        """
        full_path = (self.working_dir / path).resolve()

        # Security check: ensure path is within working directory
        if not str(full_path).startswith(str(self.working_dir)):
            raise ValueError(f"Path {path} is outside working directory")

        return full_path

    def read_file(self, file_path: str, limit: int = 500) -> ToolResult:
        """Read contents of a file.

        Args:
            file_path: Path to file
            limit: Maximum lines to read

        Returns:
            ToolResult with file contents
        """
        try:
            path = self._safe_path(file_path)

            if not path.exists():
                return ToolResult(
                    success=False,
                    output="",
                    error=f"File not found: {file_path}"
                )

            with open(path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            if len(lines) > limit:
                content = ''.join(lines[:limit])
                content += f"\n... [truncated, {len(lines) - limit} more lines]"
            else:
                content = ''.join(lines)

            return ToolResult(
                success=True,
                output=content
            )

        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
            return ToolResult(
                success=False,
                output="",
                error=str(e)
            )

    def write_file(self, file_path: str, content: str) -> ToolResult:
        """Write content to a file.

        Args:
            file_path: Path to file
            content: Content to write

        Returns:
            ToolResult with status
        """
        try:
            path = self._safe_path(file_path)

            # Create parent directories if needed
            path.parent.mkdir(parents=True, exist_ok=True)

            with open(path, 'w', encoding='utf-8') as f:
                f.write(content)

            return ToolResult(
                success=True,
                output=f"Successfully wrote {len(content)} characters to {file_path}"
            )

        except Exception as e:
            logger.error(f"Error writing file {file_path}: {e}")
            return ToolResult(
                success=False,
                output="",
                error=str(e)
            )

    def edit_file(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False
    ) -> ToolResult:
        """Edit a file by replacing text.

        Args:
            file_path: Path to file
            old_string: String to find
            new_string: String to replace with
            replace_all: Replace all occurrences

        Returns:
            ToolResult with status
        """
        try:
            path = self._safe_path(file_path)

            if not path.exists():
                return ToolResult(
                    success=False,
                    output="",
                    error=f"File not found: {file_path}"
                )

            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()

            if old_string not in content:
                return ToolResult(
                    success=False,
                    output="",
                    error=f"String not found in file: {old_string[:50]}..."
                )

            if replace_all:
                new_content = content.replace(old_string, new_string)
                count = content.count(old_string)
            else:
                new_content = content.replace(old_string, new_string, 1)
                count = 1

            with open(path, 'w', encoding='utf-8') as f:
                f.write(new_content)

            return ToolResult(
                success=True,
                output=f"Replaced {count} occurrence(s) in {file_path}"
            )

        except Exception as e:
            logger.error(f"Error editing file {file_path}: {e}")
            return ToolResult(
                success=False,
                output="",
                error=str(e)
            )

    def bash(
        self,
        command: str,
        timeout: int = 30,
        allow_dangerous: bool = False
    ) -> ToolResult:
        """Execute a bash command.

        Args:
            command: Command to execute
            timeout: Timeout in seconds
            allow_dangerous: Allow potentially dangerous commands

        Returns:
            ToolResult with command output
        """
        # Security: block dangerous commands unless explicitly allowed
        dangerous_patterns = [
            r'rm\s+-rf\s+/',
            r'rm\s+-rf\s+\*',
            r'mkfs',
            r'dd\s+if=',
            r'>\s*/dev/',
            r'chmod\s+777',
            r'curl.*\|\s*bash',
            r'wget.*\|\s*bash',
        ]

        if not allow_dangerous:
            for pattern in dangerous_patterns:
                if re.search(pattern, command, re.IGNORECASE):
                    return ToolResult(
                        success=False,
                        output="",
                        error=f"Dangerous command blocked: {command}"
                    )

        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(self.working_dir)
            )

            output = result.stdout
            if result.stderr:
                output += f"\n[stderr]: {result.stderr}"

            return ToolResult(
                success=result.returncode == 0,
                output=output,
                error=result.stderr if result.returncode != 0 else None
            )

        except subprocess.TimeoutExpired:
            return ToolResult(
                success=False,
                output="",
                error=f"Command timed out after {timeout}s"
            )
        except Exception as e:
            logger.error(f"Error executing command: {e}")
            return ToolResult(
                success=False,
                output="",
                error=str(e)
            )

    def glob(self, pattern: str, path: str = ".") -> ToolResult:
        """Find files matching a glob pattern.

        Args:
            pattern: Glob pattern (e.g., "**/*.py")
            path: Base path to search

        Returns:
            ToolResult with matching files
        """
        try:
            base_path = self._safe_path(path)
            full_pattern = str(base_path / pattern)

            matches = glob_module.glob(full_pattern, recursive=True)

            # Make paths relative
            relative_matches = [
                str(Path(m).relative_to(self.working_dir))
                for m in matches
            ]

            # Sort and limit
            relative_matches.sort()
            if len(relative_matches) > 100:
                relative_matches = relative_matches[:100]
                relative_matches.append(f"... and more (truncated at 100)")

            return ToolResult(
                success=True,
                output="\n".join(relative_matches) if relative_matches else "No matches found"
            )

        except Exception as e:
            logger.error(f"Error in glob: {e}")
            return ToolResult(
                success=False,
                output="",
                error=str(e)
            )

    def grep(
        self,
        pattern: str,
        path: str = ".",
        file_pattern: str = "*",
        case_insensitive: bool = True
    ) -> ToolResult:
        """Search for pattern in files.

        Args:
            pattern: Regex pattern to search
            path: Directory to search
            file_pattern: File glob pattern
            case_insensitive: Case insensitive search

        Returns:
            ToolResult with matches
        """
        try:
            base_path = self._safe_path(path)
            flags = re.IGNORECASE if case_insensitive else 0
            regex = re.compile(pattern, flags)

            matches = []

            # Find files
            for file_path in base_path.rglob(file_pattern):
                if not file_path.is_file():
                    continue
                if file_path.suffix.lower() not in self.allowed_extensions:
                    continue

                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        for line_num, line in enumerate(f, 1):
                            if regex.search(line):
                                rel_path = file_path.relative_to(self.working_dir)
                                matches.append(f"{rel_path}:{line_num}: {line.strip()}")

                                if len(matches) >= 50:
                                    matches.append("... (truncated at 50 matches)")
                                    break

                except Exception:
                    continue

                if len(matches) >= 50:
                    break

            return ToolResult(
                success=True,
                output="\n".join(matches) if matches else "No matches found"
            )

        except Exception as e:
            logger.error(f"Error in grep: {e}")
            return ToolResult(
                success=False,
                output="",
                error=str(e)
            )

    def create_directory(self, dir_path: str) -> ToolResult:
        """Create a directory.

        Args:
            dir_path: Path to directory

        Returns:
            ToolResult with status
        """
        try:
            path = self._safe_path(dir_path)
            path.mkdir(parents=True, exist_ok=True)

            return ToolResult(
                success=True,
                output=f"Created directory: {dir_path}"
            )

        except Exception as e:
            logger.error(f"Error creating directory: {e}")
            return ToolResult(
                success=False,
                output="",
                error=str(e)
            )

    def list_directory(self, dir_path: str = ".") -> ToolResult:
        """List contents of a directory.

        Args:
            dir_path: Path to directory

        Returns:
            ToolResult with directory listing
        """
        try:
            path = self._safe_path(dir_path)

            if not path.exists():
                return ToolResult(
                    success=False,
                    output="",
                    error=f"Directory not found: {dir_path}"
                )

            items = []
            for item in sorted(path.iterdir()):
                if item.is_dir():
                    items.append(f"📁 {item.name}/")
                else:
                    size = item.stat().st_size
                    items.append(f"📄 {item.name} ({size:,} bytes)")

            return ToolResult(
                success=True,
                output="\n".join(items) if items else "Empty directory"
            )

        except Exception as e:
            logger.error(f"Error listing directory: {e}")
            return ToolResult(
                success=False,
                output="",
                error=str(e)
            )

    def get_tool_schemas(self) -> List[Dict]:
        """Get OpenAI-format schemas for all tools.

        Returns:
            List of tool schemas
        """
        return [
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read the contents of a file",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "file_path": {
                                "type": "string",
                                "description": "Path to the file to read"
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Maximum lines to read",
                                "default": 500
                            }
                        },
                        "required": ["file_path"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "write_file",
                    "description": "Write content to a file, creating it if it doesn't exist",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "file_path": {
                                "type": "string",
                                "description": "Path to the file to write"
                            },
                            "content": {
                                "type": "string",
                                "description": "Content to write to the file"
                            }
                        },
                        "required": ["file_path", "content"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "edit_file",
                    "description": "Edit a file by replacing text",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "file_path": {
                                "type": "string",
                                "description": "Path to the file to edit"
                            },
                            "old_string": {
                                "type": "string",
                                "description": "String to find and replace"
                            },
                            "new_string": {
                                "type": "string",
                                "description": "Replacement string"
                            },
                            "replace_all": {
                                "type": "boolean",
                                "description": "Replace all occurrences",
                                "default": False
                            }
                        },
                        "required": ["file_path", "old_string", "new_string"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "bash",
                    "description": "Execute a bash command in the terminal",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "command": {
                                "type": "string",
                                "description": "The bash command to execute"
                            },
                            "timeout": {
                                "type": "integer",
                                "description": "Timeout in seconds",
                                "default": 30
                            }
                        },
                        "required": ["command"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "glob",
                    "description": "Find files matching a glob pattern",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "pattern": {
                                "type": "string",
                                "description": "Glob pattern (e.g., '**/*.py')"
                            },
                            "path": {
                                "type": "string",
                                "description": "Base path to search",
                                "default": "."
                            }
                        },
                        "required": ["pattern"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "grep",
                    "description": "Search for pattern in files",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "pattern": {
                                "type": "string",
                                "description": "Regex pattern to search for"
                            },
                            "path": {
                                "type": "string",
                                "description": "Directory to search",
                                "default": "."
                            },
                            "file_pattern": {
                                "type": "string",
                                "description": "File glob pattern",
                                "default": "*"
                            }
                        },
                        "required": ["pattern"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "list_directory",
                    "description": "List contents of a directory",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "dir_path": {
                                "type": "string",
                                "description": "Path to directory",
                                "default": "."
                            }
                        }
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "create_directory",
                    "description": "Create a new directory",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "dir_path": {
                                "type": "string",
                                "description": "Path to directory to create"
                            }
                        },
                        "required": ["dir_path"]
                    }
                }
            }
        ]

    def execute_tool(self, tool_name: str, **kwargs) -> ToolResult:
        """Execute a tool by name.

        Args:
            tool_name: Name of the tool
            **kwargs: Tool arguments

        Returns:
            ToolResult
        """
        tool_map = {
            "read_file": self.read_file,
            "write_file": self.write_file,
            "edit_file": self.edit_file,
            "bash": self.bash,
            "glob": self.glob,
            "grep": self.grep,
            "list_directory": self.list_directory,
            "create_directory": self.create_directory,
        }

        if tool_name not in tool_map:
            return ToolResult(
                success=False,
                output="",
                error=f"Unknown tool: {tool_name}"
            )

        return tool_map[tool_name](**kwargs)


if __name__ == "__main__":
    # Test the tools
    tools = SystemTools(working_dir=".")

    print("Testing SystemTools...")

    # Test list_directory
    result = tools.list_directory(".")
    print(f"\nlist_directory('.'):\n{result.output[:500]}")

    # Test glob
    result = tools.glob("**/*.py")
    print(f"\nglob('**/*.py'):\n{result.output[:500]}")

    # Test bash
    result = tools.bash("echo 'Hello from THAU!'")
    print(f"\nbash('echo Hello'):\n{result.output}")

    # Test read_file
    result = tools.read_file("README.md", limit=10)
    print(f"\nread_file('README.md'):\n{result.output[:300]}")

    print("\nAll tools working!")
