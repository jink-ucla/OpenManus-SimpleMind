# --- Start of file: app/agent/swe.py ---

# ---> ADD THIS IMPORT AT THE TOP <---
from typing import List, Optional # Add Optional here

from pydantic import Field, model_validator
from app.agent.toolcall import ToolCallAgent
from app.prompt.swe import SYSTEM_PROMPT
from app.tool import Bash, StrReplaceEditor, Terminate, ToolCollection, TerminalExecuteTool
from app.tool.sm_json_generation import SM_JsonGenerationTool # Import your tool
from app.llm import LLM # Import LLM

class SWEAgent(ToolCallAgent):
    name: str = "swe"
    description: str = "an autonomous AI programmer that interacts directly with the computer to solve tasks."

    system_prompt: str = SYSTEM_PROMPT
    next_step_prompt: str = ""

    # Define available_tools but initialize it later or make it Optional
    # Use the imported Optional type hint
    available_tools: Optional[ToolCollection] = None
    special_tool_names: List[str] = Field(default_factory=lambda: [Terminate().name])

    max_steps: int = 20

    # Use a Pydantic validator to initialize tools *after* the agent (and its LLM) is created
    @model_validator(mode="after")
    def initialize_tools(self) -> "SWEAgent":
        # Ensure the agent's LLM instance exists (ToolCallAgent should handle this)
        if not hasattr(self, 'llm') or not self.llm:
             self.llm = LLM() # Or load from config

        # Now initialize the ToolCollection, passing self.llm to the tool
        self.available_tools = ToolCollection(
            Bash(),
            StrReplaceEditor(),
            Terminate(),
            TerminalExecuteTool(),
            SM_JsonGenerationTool(llm=self.llm) # Pass self.llm
        )
        return self

# --- End of file: app/agent/swe.py ---