# --- Start of file: app/agent/manus.py ---

from typing import Dict, List, Optional

# Add LLM to imports if not already there
from pydantic import Field, model_validator

from app.agent.browser import BrowserContextHelper
from app.agent.toolcall import ToolCallAgent
from app.config import config
from app.logger import logger
from app.prompt.manus import NEXT_STEP_PROMPT, SYSTEM_PROMPT
# Import LLM class
from app.llm import LLM
# Make sure all needed tools are imported
from app.tool.ask_human import AskHuman
from app.tool.browser_use_tool import BrowserUseTool
from app.tool.mcp import MCPClients, MCPClientTool
from app.tool.python_execute import PythonExecute
from app.tool.str_replace_editor import StrReplaceEditor
from app.tool.terminal_execute import TerminalExecuteTool
from app.tool.sm_json_generation import SM_JsonGenerationTool
from app.tool.mcp import MCPClients, MCPClientTool
from app.tool import Terminate, ToolCollection
from app.config import config, SandboxSettings 

class Manus(ToolCallAgent):
    """A versatile general-purpose agent with support for both local and MCP tools."""

    name: str = "Manus"
    description: str = "A versatile agent that can solve various tasks using multiple tools including MCP-based tools"

    system_prompt: str = SYSTEM_PROMPT.format(directory=config.workspace_root)
    next_step_prompt: str = NEXT_STEP_PROMPT

    max_observe: int = 10000
    max_steps: int = 20

    # MCP clients for remote tool access
    mcp_clients: MCPClients = Field(default_factory=MCPClients)

    # --- CHANGE 1: Initialize available_tools as None initially ---
    available_tools: Optional[ToolCollection] = None

    special_tool_names: list[str] = Field(default_factory=lambda: [Terminate().name])
    browser_context_helper: Optional[BrowserContextHelper] = None

    # Track connected MCP servers
    connected_servers: Dict[str, str] = Field(
        default_factory=dict
    )  # server_id -> url/command
    _initialized: bool = False # Tracks MCP initialization

    # --- CHANGE 2: Modify the existing validator to also initialize tools ---
    @model_validator(mode="after")
    def initialize_dependencies(self) -> "Manus":
        """Initialize basic components and tools after agent creation."""
        # Ensure LLM is initialized (BaseAgent/ToolCallAgent likely handles this)
        # Add a check just in case
        if not hasattr(self, 'llm') or not self.llm:
             logger.warning("LLM not found during validator, attempting default initialization.")
             self.llm = LLM() # Or load from config specific to Manus if needed

        # Initialize Browser Helper
        self.browser_context_helper = BrowserContextHelper(self)

        # Initialize the ToolCollection, passing self.llm where needed
        # Note: This defines the *base* set of tools before MCP tools are added.

        
        self.available_tools = ToolCollection(
            PythonExecute(),
            BrowserUseTool(), # Does BrowserUseTool need LLM? Pass if needed: BrowserUseTool(llm=self.llm)
            StrReplaceEditor(),
            AskHuman(),
            Terminate(),
            TerminalExecuteTool(
                # name="syp_sandboxed_terminal", # 도구 이름 (선택 사항)
                # config.py에서 로드된 기본 샌드박스 설정을 사용하되, 이미지와 볼륨만 오버라이드
                sandbox_config=SandboxSettings(
                    # 기본 설정 복사 (필요하다면)
                    use_sandbox=True,
                    memory_limit=config.sandbox.memory_limit,
                    cpu_limit=config.sandbox.cpu_limit,
                    timeout=config.sandbox.timeout,
                    network_enabled=True,
                    # 특정 이미지와 작업 디렉토리 지정
                    image="registry.cvib.ucla.edu/sm:everything", # <-- 원하는 Docker 이미지
                    work_dir="/workspace", # <-- 샌드박스 내 작업 디렉토리
                    # auto_remove=False,
                ),
                volume_bindings={ # <-- 원하는 볼륨 바인딩 설정 
                    # "/cvib2/apps/personal/wasil/agentic_ai_sprint": "/workdir",
                    "/radraid2/../radraid2/mwahianwar/agentic_ai": "/workdir",
                    "/radraid2/mwahianwar/agentic_ai": "/radraid2/mwahianwar/agentic_ai",                 
                },
                default_timeout=3000 # 이 도구 인스턴스의 기본 타임아웃
            ),
            SM_JsonGenerationTool(llm=self.llm) # <-- PASS self.llm HERE
        )

        logger.debug(f"Manus base tools initialized: {[t.name for t in self.available_tools.tools]}")
        # MCP tools will be added later by initialize_mcp_servers

        # Ensure _initialized reflects the agent's readiness before MCP connection attempt
        # We keep self._initialized = False until MCP servers are attempted in create()

        return self

    @classmethod
    async def create(cls, **kwargs) -> "Manus":
        """Factory method to create and properly initialize a Manus instance."""
        # This initializes the agent, including running the model_validator above
        instance = cls(**kwargs)
        # Now connect to MCP servers and add their tools
        await instance.initialize_mcp_servers()
        # Mark as fully initialized *after* MCP attempt
        instance._initialized = True
        return instance

    async def initialize_mcp_servers(self) -> None:
        """Initialize connections to configured MCP servers and add their tools."""
        # Ensure base tools are initialized first
        if not self.available_tools:
             raise RuntimeError("Base tools not initialized before attempting MCP connection.")

        initial_tool_count = len(self.available_tools.tools)
        logger.debug(f"Attempting to connect MCP servers. Base tools count: {initial_tool_count}")

        for server_id, server_config in config.mcp_config.servers.items():
            try:
                if server_config.type == "sse":
                    if server_config.url:
                        # Pass self.available_tools to be potentially updated
                        await self.connect_mcp_server(server_config.url, server_id)
                        logger.info(
                            f"Connected to MCP server {server_id} at {server_config.url}"
                        )
                elif server_config.type == "stdio":
                    if server_config.command:
                        # Pass self.available_tools to be potentially updated
                        await self.connect_mcp_server(
                            server_config.command,
                            server_id,
                            use_stdio=True,
                            stdio_args=server_config.args,
                        )
                        logger.info(
                            f"Connected to MCP server {server_id} using command {server_config.command}"
                        )
            except Exception as e:
                logger.error(f"Failed to connect to MCP server {server_id}: {e}")

        final_tool_count = len(self.available_tools.tools)
        logger.debug(f"MCP server connection attempt finished. Total tools: {final_tool_count}")


    async def connect_mcp_server(
        self,
        server_url: str,
        server_id: str = "",
        use_stdio: bool = False,
        stdio_args: List[str] = None,
    ) -> None:
        """Connect to an MCP server and add its tools."""
        # Ensure base tools are initialized
        if not self.available_tools:
             raise RuntimeError("Base tools must be initialized before connecting to MCP servers.")

        if use_stdio:
            await self.mcp_clients.connect_stdio(
                server_url, stdio_args or [], server_id
            )
            self.connected_servers[server_id or server_url] = server_url
        else:
            await self.mcp_clients.connect_sse(server_url, server_id)
            self.connected_servers[server_id or server_url] = server_url

        # Get NEW tools specific to this MCP server connection
        # Use the mcp_clients instance which now holds the connection and tools
        new_mcp_tools = [
            tool for tool in self.mcp_clients.tools
            # Filter tools belonging to the server we just connected
            if isinstance(tool, MCPClientTool) and tool.server_id == (server_id or (server_url if not use_stdio else command))
            # Avoid adding duplicates if re-connecting
            and tool.name not in self.available_tools.tool_map
        ]

        if new_mcp_tools:
             logger.info(f"Adding {len(new_mcp_tools)} new tools from MCP server {server_id or server_url}")
             # Add *only* the new tools to the existing collection
             self.available_tools.add_tools(*new_mcp_tools)
        else:
             logger.info(f"No new tools found or added from MCP server {server_id or server_url} (already connected or no tools).")


    async def disconnect_mcp_server(self, server_id: str = "") -> None:
        """Disconnect from an MCP server and remove its tools."""
        # Ensure base tools collection exists
        if not self.available_tools:
             logger.warning("Disconnect called but available_tools not initialized.")
             return

        server_id_to_disconnect = server_id
        if not server_id_to_disconnect:
             logger.info("Disconnecting from all MCP servers.")
             all_server_ids = list(self.connected_servers.keys())
             await self.mcp_clients.disconnect() # Disconnect all internal sessions
             self.connected_servers.clear()
             # Remove all MCP tools
             self.available_tools.tools = tuple(
                 tool for tool in self.available_tools.tools if not isinstance(tool, MCPClientTool)
             )
             self.available_tools.tool_map = {t.name: t for t in self.available_tools.tools}
             logger.info(f"Removed all MCP tools. Current tool count: {len(self.available_tools.tools)}")

        elif server_id_to_disconnect in self.connected_servers:
            logger.info(f"Disconnecting from MCP server: {server_id_to_disconnect}")
            await self.mcp_clients.disconnect(server_id_to_disconnect) # Disconnect specific session
            self.connected_servers.pop(server_id_to_disconnect, None)
            # Remove tools associated ONLY with this server
            self.available_tools.tools = tuple(
                tool for tool in self.available_tools.tools
                if not (isinstance(tool, MCPClientTool) and tool.server_id == server_id_to_disconnect)
            )
            self.available_tools.tool_map = {t.name: t for t in self.available_tools.tools}
            logger.info(f"Removed tools for {server_id_to_disconnect}. Current tool count: {len(self.available_tools.tools)}")
        else:
             logger.warning(f"Attempted to disconnect from unknown MCP server: {server_id_to_disconnect}")

    async def cleanup(self):
        """Clean up Manus agent resources."""
        logger.debug(f"Starting cleanup for Manus agent {self.name}...")
        if self.browser_context_helper:
            await self.browser_context_helper.cleanup_browser()
            logger.debug("Browser helper cleaned up.")
        # Disconnect from all MCP servers
        # No need to check self._initialized here, just disconnect if connected
        await self.disconnect_mcp_server() # Disconnects all if server_id is empty
        logger.debug("MCP connections closed.")
        logger.info(f"Manus agent {self.name} cleanup complete.")


    async def think(self) -> bool:
        """Process current state and decide next actions with appropriate context."""
        # MCP Initialization is now handled by the create classmethod
        # Ensure tools are available before thinking
        if not self.available_tools:
             logger.error("Think called before tools were initialized. Attempting recovery.")
             # Attempt re-initialization (might be problematic depending on state)
             self.initialize_dependencies()
             if not self.available_tools:
                  raise RuntimeError("Failed to initialize tools before think method.")


        original_prompt = self.next_step_prompt
        # Check if browser was used recently (logic remains the same)
        recent_messages = self.memory.messages[-3:] if self.memory.messages else []
        browser_in_use = any(
            hasattr(tc, 'function') and tc.function.name == BrowserUseTool().name
            for msg in recent_messages
            if msg.tool_calls
            for tc in msg.tool_calls
        )

        if browser_in_use and self.browser_context_helper:
            logger.debug("Browser was recently used, formatting next step prompt with browser state.")
            self.next_step_prompt = (
                await self.browser_context_helper.format_next_step_prompt()
            )
        else:
             logger.debug("Browser not recently used or helper not available.")


        # Call the parent think method which handles the LLM call with tools
        result = await super().think()

        # Restore original prompt
        self.next_step_prompt = original_prompt

        return result

# --- End of file: app/agent/manus.py ---