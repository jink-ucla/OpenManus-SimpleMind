# app/tool/terminal_execute.py

import asyncio
import subprocess
import shlex
import logging # Use standard logging
from typing import Optional, Union, Dict, Any

# Import ConfigDict for Pydantic v2 configuration.
from pydantic import ConfigDict, Field

# Import necessary sandbox components
from app.config import SandboxSettings, config # Import config to potentially use default sandbox settings
from app.sandbox.client import LocalSandboxClient # Use LocalSandboxClient directly
from app.sandbox.core.exceptions import SandboxTimeoutError # Import sandbox-specific timeout error

from app.tool.base import BaseTool, ToolResult, ToolFailure, CLIResult
# Use standard logging
logger = logging.getLogger(__name__)
# Example basic config if not configured globally:
# logging.basicConfig(level=logging.INFO) # Adjust level as needed


DEFAULT_TERMINAL_TIMEOUT = 66000 # seconds

class TerminalExecuteTool(BaseTool):
    name: str = "terminal_execute"
    description: str = "Executes commands in a sandboxed terminal environment."
    parameters: dict = {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "The terminal command to execute."
            },
            "timeout": {
                "type": "integer",
                "description": f"Optional timeout in seconds. Default is {DEFAULT_TERMINAL_TIMEOUT}."
            },
            "cwd": {
                "type": "string",
                "description": "The working directory inside the sandbox. Use '/workdir' to access the mounted volume. Defaults to the sandbox's configured work_dir (usually '/workspace')."
            }
        },
        "required": ["command"],
        "additionalProperties": False
    }

    # Pydantic v2 configuration: Allow setting additional attributes not defined as model fields.
    model_config = ConfigDict(extra='allow')

    # Sandbox related attributes (internal state)
    _sandbox_client: Optional[LocalSandboxClient] = None
    _sandbox_config: Optional[SandboxSettings] = None
    _volume_bindings: Optional[Dict[str, str]] = None
    _is_initialized: bool = False # Track initialization state
    _intended_sandbox_cwd: Optional[str] = None # The preferred sandbox cwd based on volume bindings

    def __init__(
        self,
        default_timeout: int = DEFAULT_TERMINAL_TIMEOUT,
        # Parameters to receive sandbox config
        sandbox_config: Optional[SandboxSettings] = None,
        volume_bindings: Optional[Dict[str, str]] = None,
        **data
    ):
        super().__init__(**data)

        self.default_timeout = default_timeout
        self.use_shell = data.get('use_shell', True) # Keep use_shell if passed, default True

        self._sandbox_config = sandbox_config
        self._volume_bindings = volume_bindings

        # --- ADDED: Determine the intended sandbox cwd from volume bindings ---
        self._intended_sandbox_cwd = None
        if self._volume_bindings:
            # Assume the container path of the first volume binding is the intended cwd
            # This is a heuristic, but works for the user's specific case (/cvib2/... -> /workdir)
            first_binding_container_path = next(iter(self._volume_bindings.values()), None)
            if first_binding_container_path:
                 self._intended_sandbox_cwd = first_binding_container_path
                 logger.debug(f"Determined intended sandbox cwd from volume bindings: {self._intended_sandbox_cwd}")
        # --- END ADDED ---

        logger.debug(f"TerminalExecuteTool initialized: name='{self.name}', default_timeout={self.default_timeout}, use_shell={self.use_shell}, intended_cwd={self._intended_sandbox_cwd}")

    async def init_sandbox(self):
        """Initializes the sandbox client and creates the container."""
        if self._is_initialized:
            return

        logger.info(f"Initializing sandbox for TerminalExecuteTool '{self.name}'...")

        # Use provided config or fallback to global config
        sandbox_cfg = self._sandbox_config if self._sandbox_config is not None else config.sandbox
        print(f"Sandbox config: {sandbox_cfg}") # For debugging, remove in production
        if sandbox_cfg is None:
             sandbox_cfg = SandboxSettings()
             logger.warning(f"No sandbox config provided for tool '{self.name}' and global config.sandbox is None. Using default SandboxSettings.")

        if not sandbox_cfg.use_sandbox:
             logger.warning(f"Sandbox is not enabled in config or specific config provided for tool '{self.name}'. Commands will NOT run in sandbox.")
             self._sandbox_client = None
             self._is_initialized = True
             return

        self._sandbox_client = LocalSandboxClient()
        try:
            await self._sandbox_client.create(
                config=sandbox_cfg,
                volume_bindings=self._volume_bindings
            )
            self._is_initialized = True
            logger.info(f"Sandbox initialized successfully for TerminalExecuteTool '{self.name}'.")
        except Exception as e:
            logger.error(f"Failed to initialize sandbox for TerminalExecuteTool '{self.name}': {e}", exc_info=True)
            self._sandbox_client = None
            self._is_initialized = True
            raise RuntimeError(f"Sandbox initialization failed: {e}") from e


    async def execute(self, **kwargs) -> Union[ToolResult, ToolFailure, CLIResult]:
        command: Optional[str] = kwargs.get("command")
        timeout: Optional[int] = kwargs.get("timeout")
        # Get the cwd argument provided by the LLM
        llm_provided_cwd: Optional[str] = kwargs.get("cwd")

        if not command:
            logger.error("Command parameter is missing or empty.")
            return ToolFailure(error="Command parameter is required.")

        if not self._is_initialized:
             try:
                 await self.init_sandbox()
             except RuntimeError as e:
                 return ToolFailure(error=f"Sandbox not available: {e}")

        if not self._sandbox_client or not self._sandbox_client.sandbox:
             logger.error(f"Sandbox client is not available for tool '{self.name}'. Cannot execute command in sandbox.")
             return ToolFailure(error="Sandbox environment is not available for this tool.")

        actual_timeout = timeout if timeout is not None else self.default_timeout

        # --- MODIFIED: Determine the effective cwd for the command ---
        effective_cwd = None
        sandbox_default_workdir = self._sandbox_config.work_dir if self._sandbox_config else "/workspace" # Fallback default

        if llm_provided_cwd is None or llm_provided_cwd.strip() == "":
            # If LLM didn't provide cwd, use the intended sandbox cwd from bindings,
            # or fallback to the sandbox's default work_dir.
            effective_cwd = self._intended_sandbox_cwd if self._intended_sandbox_cwd else sandbox_default_workdir
            logger.debug(f"LLM provided no cwd. Using effective_cwd: {effective_cwd} (intended: {self._intended_sandbox_cwd}, sandbox default: {sandbox_default_workdir})")
        else:
            # If LLM provided a cwd, use it.
            effective_cwd = llm_provided_cwd
            # Optional: Add a warning if the provided cwd looks like a host path
            # This check is heuristic and might need adjustment.
            if effective_cwd.startswith('/cvib2/') or effective_cwd.startswith('/path/on/host/'): # Add other host path prefixes if known
                 logger.warning(f"LLM provided a cwd that looks like a host path: '{effective_cwd}'. Using it, but this might fail in the sandbox.")
            logger.debug(f"LLM provided cwd: '{llm_provided_cwd}'. Using effective_cwd: {effective_cwd}")

        # Ensure effective_cwd is not None before using it in the command
        if effective_cwd is None:
             effective_cwd = sandbox_default_workdir
             logger.warning(f"Effective cwd is None after logic, falling back to sandbox default: {effective_cwd}")

        logger.info(f"Executing sandboxed terminal command: {command} (Timeout: {actual_timeout}s)")
        logger.info(f"Effective working directory inside sandbox: {effective_cwd}")

        try:
            # Construct the command string with the effective_cwd
            # Use shlex.quote to safely handle paths with spaces or special characters in the cd command.
            command_to_run = f"cd {shlex.quote(effective_cwd)} && {command}"

            stdout = await self._sandbox_client.run_command(
                command_to_run,
                timeout=actual_timeout
            )

            logger.info(f"Sandboxed command execution completed.")
            logger.debug(f"STDOUT (from sandbox): {stdout}")
            print(f"STDOUT (from sandbox): {stdout}") # For debugging, remove in production
            # Assuming success if run_command didn't raise an exception
            return CLIResult(output=stdout, error="", system="Exit Code: 0 (Sandbox)")

        except SandboxTimeoutError:
            logger.warning(f"Sandboxed command timed out: {command}")
            return ToolFailure(error=f"Sandboxed command timed out after {actual_timeout} seconds.", system="Timeout")
        except Exception as e:
             logger.error(f"An error occurred during sandboxed command execution: {e}", exc_info=True)
             return ToolFailure(error=f"Error during sandboxed command execution: {e}")

    async def cleanup(self):
        """Cleans up the sandbox resources managed by this tool instance."""
        if self._sandbox_client:
            logger.info(f"Cleaning up sandbox for TerminalExecuteTool '{self.name}'...")
            try:
                await self._sandbox_client.cleanup()
                logger.info(f"Sandbox cleanup complete for TerminalExecuteTool '{self.name}'.")
            except Exception as e:
                 logger.error(f"Error during sandbox cleanup for TerminalExecuteTool '{self.name}': {e}", exc_info=True)
            finally:
                self._sandbox_client = None
                self._is_initialized = False

# Note: The __del__ method from BrowserUseTool is not included here.
# Rely on the agent's cleanup method to call this tool's cleanup.