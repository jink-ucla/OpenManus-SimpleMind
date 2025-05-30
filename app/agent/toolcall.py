# --- Start of file: app/agent/toolcall.py ---

import asyncio
import json
import re
import logging
import ast # Import ast for literal_eval fallback
import uuid # Import uuid for generating tool call IDs
from typing import Any, Dict, List, Optional, Union, get_type_hints

from pydantic import Field, ValidationError

from app.agent.react import ReActAgent
from app.exceptions import TokenLimitExceeded
# Use standard logging
logger = logging.getLogger(__name__)
# Example basic config if not configured globally:
# logging.basicConfig(level=logging.INFO) # Adjust level as needed

from app.prompt.toolcall import NEXT_STEP_PROMPT, SYSTEM_PROMPT # Assuming these are correct
from app.schema import TOOL_CHOICE_TYPE, AgentState, Message, ToolCall, ToolChoice, Function
from app.tool.base import ToolResult, BaseTool # Import ToolResult and BaseTool
from app.tool import CreateChatCompletion, Terminate, ToolCollection

# Import or define unescape_code
try:
    from app.utils import unescape_code
except ImportError:
    import html
    logger.warning("Could not import unescape_code from app.utils, defining basic version.")
    def unescape_code(raw_code: str) -> str:
        if not isinstance(raw_code, str): return raw_code
        try:
            # More robust unescaping might be needed depending on LLM output
            code_html_unescaped = html.unescape(raw_code)
            # Basic handling, might miss complex escapes handled by unicode_escape
            decoded_code = code_html_unescaped.replace("\\n", "\n").replace("\\t", "\t").replace('\\"', '"').replace("\\'", "'").replace("\\\\", "\\")
            return decoded_code
        except Exception as e:
            logger.error(f"Basic unescape_code error: {e}")
            return raw_code # Fallback


TOOL_CALL_REQUIRED = "Tool calls required but none provided"

# Regex patterns for Pythonic parsing (kept for fallback)
PYTHONIC_CALL_PATTERN = re.compile(r"(\b[a-zA-Z_]\w*)\((.*?)\)", re.DOTALL)
MARKER_CALL_PATTERN = re.compile(r"<\|python_start\|>(.*?)<\|python_end\|>", re.DOTALL)
MARKDOWN_PYTHON_BLOCK_PATTERN = re.compile(r"```python\s*\n(.*?)\n```", re.DOTALL | re.IGNORECASE)

# Regex for parsing key=value arguments (improved for pythonic fallback)
ARG_PATTERN = re.compile(r"""
    \b([a-zA-Z_]\w*)\s*=\s* # Capture the key (identifier) followed by =
    (
        ('[^']*?(?<!\\)') |  # Single-quoted string (non-greedy, handles escaped quotes)
        ("[^"]*?(?<!\\)") |  # Double-quoted string (non-greedy, handles escaped quotes)
        (True|False|None) |  # Booleans / None
        ([\d\.-]+) |         # Numbers (int/float, allows negatives)
        # Fallback for complex values or unquoted strings - capture until next known delimiter
        (.*?)(?=,\s*\b[a-zA-Z_]\w*\s*=|\)$) # Match non-greedily until next key= or end parenthesis
    )
""", re.VERBOSE | re.DOTALL)


def _parse_pythonic_args_regex(args_str: str) -> Dict[str, Any]:
    """
    Parses a string of Python-like keyword arguments using regex.
    Improved handling for quoted strings and basic types. (Used for pythonic fallback)
    """
    args_dict = {}
    if not args_str.strip():
        return args_dict

    # Iterate through matches found by the ARG_PATTERN
    for match in ARG_PATTERN.finditer(args_str):
        key = match.group(1)
        # Extract the matched value components - check groups in order of specificity
        single_quoted = match.group(3)
        double_quoted = match.group(4)
        bool_none = match.group(5)
        number = match.group(6)
        fallback_value = match.group(7) # Capture the fallback group

        value = None
        value_processed = False

        if single_quoted is not None:
            value = single_quoted[1:-1].replace("\\'", "'") # Remove quotes and unescape internal quotes
            value_processed = True
        elif double_quoted is not None:
            value = double_quoted[1:-1].replace('\\"', '"') # Remove quotes and unescape internal quotes
            value_processed = True
        elif bool_none is not None:
            if bool_none == 'True': value = True
            elif bool_none == 'False': value = False
            elif bool_none == 'None': value = None
            value_processed = True
        elif number is not None:
            try: value = int(number)
            except ValueError:
                try: value = float(number)
                except ValueError: value = number # Keep as string if conversion fails
            value_processed = True
        elif fallback_value is not None:
            # Use the fallback value, stripping surrounding whitespace
            value = fallback_value.strip()
            # Attempt to evaluate if it looks like a simple literal, otherwise keep as string
            try:
                value = ast.literal_eval(value) # Safer eval for basic types
            except (ValueError, SyntaxError, TypeError):
                # If literal_eval fails, keep the stripped string
                pass
            value_processed = True
        else:
             # This case should ideally not be hit if the regex is correct, but handle defensively
             raw_value_part = match.group(2).strip() # The whole value part matched by group 2
             try:
                 value = ast.literal_eval(raw_value_part)
             except (ValueError, SyntaxError, TypeError):
                  value = raw_value_part # Keep raw string if all else fails
             logger.debug(f"Processed arg '{key}' using general value capture (Group 2), result: {value} (type: {type(value)})")
             value_processed = True # Mark as processed

        if key: # Ensure key was captured
             args_dict[key] = value
             logger.debug(f"Parsed pythonic arg: {key} = {value} (type: {type(value)})")

    if not args_dict and args_str.strip():
        logger.warning(f"Pythonic regex parsing yielded no key=value pairs for non-empty args: '{args_str}'")

    return args_dict


def _parse_pythonic_tool_calls(content: Optional[str]) -> tuple[List[ToolCall], str]:
    """
    Parses a string potentially containing pythonic tool calls using specific markers
    OR within markdown ```python blocks. Prioritizes marker format. (Used for pythonic fallback)
    Returns a list of validated ToolCall objects and the sanitized content string.
    """
    if not content:
        return [], ""

    tool_calls: List[ToolCall] = []
    sanitized_content = content # Start with original content
    blocks_to_remove = set() # Use set to track unique blocks

    # 1. Check for marker-based calls first (<|python_start|>...<|python_end|>)
    marker_matches = list(MARKER_CALL_PATTERN.finditer(content))
    processed_marker_block = False
    if marker_matches:
        logger.debug(f"Found {len(marker_matches)} potential marker-based call blocks.")
        for match in marker_matches:
            call_str_with_args = match.group(1).strip()
            raw_block = match.group(0)
            logger.debug(f"Processing marker block content: '{call_str_with_args}'")

            call_match = PYTHONIC_CALL_PATTERN.search(call_str_with_args)
            if call_match:
                func_name = call_match.group(1)
                args_str = call_match.group(2).strip()
                logger.debug(f"Parsed marker call: name='{func_name}', args='{args_str}'")
                args_dict = _parse_pythonic_args_regex(args_str)
                try: arguments_json = json.dumps(args_dict)
                except TypeError as e: logger.error(f"JSON serialization failed for marker tool '{func_name}': {e}. Args: {args_dict}", exc_info=True); arguments_json = "{}"

                tool_call_id = f"call_marker_{uuid.uuid4().hex[:8]}"
                tool_call_dict = {"id": tool_call_id, "type": "function", "function": {"name": func_name, "arguments": arguments_json}}

                try:
                    validated_tool_call = ToolCall(**tool_call_dict)
                    tool_calls.append(validated_tool_call)
                    blocks_to_remove.add(raw_block) # Mark the <|python_start|>...<|python_end|> block
                    processed_marker_block = True # Flag that we found at least one marker call
                    logger.debug(f"Successfully parsed and validated marker call: {validated_tool_call}")
                except ValidationError as e: logger.error(f"Validation failed for parsed marker tool call '{func_name}': {e}. Data: {tool_call_dict}")
                except Exception as e: logger.error(f"Unexpected error creating ToolCall object for marker call '{func_name}': {e}", exc_info=True)
            else:
                logger.warning(f"Could not parse function call syntax within markers: '{call_str_with_args}'")

    # 2. If no marker calls were processed, check for markdown python blocks
    if not processed_marker_block:
        markdown_matches = list(MARKDOWN_PYTHON_BLOCK_PATTERN.finditer(content))
        if markdown_matches:
            logger.debug(f"No marker calls found. Found {len(markdown_matches)} potential markdown python blocks.")
            for match in markdown_matches:
                code_block_content = match.group(1).strip()
                raw_block = match.group(0)
                logger.debug(f"Processing markdown python block content:\n```python\n{code_block_content}\n```")

                call_match = PYTHONIC_CALL_PATTERN.fullmatch(code_block_content)
                if call_match:
                    func_name = call_match.group(1)
                    args_str = call_match.group(2).strip()
                    logger.debug(f"Parsed markdown call: name='{func_name}', args='{args_str}'")
                    args_dict = _parse_pythonic_args_regex(args_str)
                    try: arguments_json = json.dumps(args_dict)
                    except TypeError as e: logger.error(f"JSON serialization failed for markdown tool '{func_name}': {e}. Args: {args_dict}", exc_info=True); arguments_json = "{}"

                    tool_call_id = f"call_markdown_{uuid.uuid4().hex[:8]}"
                    tool_call_dict = {"id": tool_call_id, "type": "function", "function": {"name": func_name, "arguments": arguments_json}}

                    try:
                        validated_tool_call = ToolCall(**tool_call_dict)
                        tool_calls.append(validated_tool_call)
                        blocks_to_remove.add(raw_block) # Mark the ```python...``` block
                        logger.debug(f"Successfully parsed and validated markdown call: {validated_tool_call}")
                    except ValidationError as e: logger.error(f"Validation failed for parsed markdown tool call '{func_name}': {e}. Data: {tool_call_dict}")
                    except Exception as e: logger.error(f"Unexpected error creating ToolCall object for markdown call '{func_name}': {e}", exc_info=True)
                else:
                    logger.debug(f"Markdown python block content does not match single function call pattern: '{code_block_content}'")


    # Remove the identified blocks from the original content
    if blocks_to_remove:
        sorted_blocks = sorted(list(blocks_to_remove), key=len, reverse=True)
        for block in sorted_blocks:
             sanitized_content = sanitized_content.replace(block, "", 1) # Replace only first occurrence

    # Final cleanup of the content string
    sanitized_content = sanitized_content.strip(" ,\t\n[]")
    if not sanitized_content.strip():
        sanitized_content = "" # Treat as no remaining text content

    if tool_calls: logger.info(f"Pythonic fallback parser extracted {len(tool_calls)} tool calls. Sanitized content length: {len(sanitized_content)}")
    else: logger.debug("Pythonic fallback parser did not extract any tool calls.")

    return tool_calls, sanitized_content


class ToolCallAgent(ReActAgent):
    """Agent that handles standard OpenAI-style tool calls, with pythonic parsing as a fallback."""

    name: str = "Manus"
    description: str = "an agent that can execute tool calls."

    system_prompt: str = SYSTEM_PROMPT
    next_step_prompt: str = NEXT_STEP_PROMPT

    available_tools: ToolCollection = Field(default_factory=lambda: ToolCollection(CreateChatCompletion(), Terminate()))
    tool_choices: TOOL_CHOICE_TYPE = ToolChoice.AUTO
    special_tool_names: List[str] = Field(default_factory=lambda: [Terminate().name])

    tool_calls: List[ToolCall] = Field(default_factory=list)
    _current_base64_image: Optional[str] = None

    max_steps: int = 30
    max_observe: Optional[Union[int, bool]] = None


    async def think(self) -> bool:
        """
        Process current state, ask LLM for next step (text or tool calls),
        and parse the response, prioritizing standard 'tool_calls' field.
        """
        if self.next_step_prompt:
            user_msg = Message.user_message(self.next_step_prompt)
            if not hasattr(self, 'messages') or self.messages is None: self.messages = []
            elif not isinstance(self.messages, list): logger.warning(f"self.messages is not a list ({type(self.messages)}), resetting."); self.messages = []

            if not self.messages or self.messages[-1].content != self.next_step_prompt:
                 logger.debug("Adding next step prompt to messages.")
                 self.messages.append(user_msg)
            else:
                 logger.debug("Next step prompt already present at end of messages, skipping.")

        try:
            # Check if the LLM instance configuration suggests using the pythonic parser (for fallback)
            use_pythonic_fallback = getattr(self.llm, 'use_pythonic_tool_parser', False)
            logger.info(f"Think: Requesting tools from LLM. Pythonic fallback enabled: {use_pythonic_fallback}")

            # Request uses standard `tools` and `tool_choice` format
            response_dict = await self.llm.ask_tool(
                messages=self.messages,
                system_msgs=(
                    [Message.system_message(self.system_prompt)]
                    if self.system_prompt
                    else None
                ),
                tools=self.available_tools.to_params(),
                tool_choice=self.tool_choices,
                use_pythonic_parser=use_pythonic_fallback, # Pass flag (informational for ask_tool, used here for fallback logic)
            )

            if response_dict is None:
                 logger.error("llm.ask_tool returned None, cannot proceed.")
                 self.memory.add_message(Message.assistant_message(content="Error: Failed to get a valid response from the language model."))
                 self.state = AgentState.ERROR
                 return False
            logger.debug(f"LLM Response Raw Dictionary: {response_dict}")

        # --- Exception Handling ---
        except TokenLimitExceeded as e: logger.error(f"🚨 Token limit error: {e}"); self.memory.add_message(Message.assistant_message(f"Max token limit reached: {str(e)}")); self.state = AgentState.FINISHED; return False
        except Exception as e:
             if isinstance(getattr(e, "__cause__", None), TokenLimitExceeded): token_limit_error = e.__cause__; logger.error(f"🚨 Token limit error (from RetryError): {token_limit_error}"); self.memory.add_message(Message.assistant_message(f"Max token limit reached: {str(token_limit_error)}")); self.state = AgentState.FINISHED; return False
             if isinstance(e, asyncio.exceptions.CancelledError):
                 logger.warning("LLM request cancelled, likely due to KeyboardInterrupt.")
                 self.memory.add_message(Message.assistant_message(content="LLM request cancelled."))
                 self.state = AgentState.ERROR
                 return False
             logger.exception(f"Unhandled exception during LLM tool request: {e}")
             self.memory.add_message(Message.assistant_message(content=f"Error during language model request: {str(e)}"))
             self.state = AgentState.ERROR; return False
        # --- End Exception Handling ---


        # --- Response Parsing Logic (Prioritize Standard Tool Calls) ---
        self.tool_calls = []
        assistant_content_text: Optional[str] = None
        finish_reason = response_dict.get("finish_reason", "stop") # Default to stop
        raw_content_str = response_dict.get("content") # Text content from LLM

        # 1. Check standard 'tool_calls' field first (Expected for Qwen/hermes)
        standard_tool_calls = response_dict.get("tool_calls")

        if standard_tool_calls and isinstance(standard_tool_calls, list):
            logger.info(f"Found {len(standard_tool_calls)} tool calls in standard 'tool_calls' field. Parsing...")
            parsed_list = []
            validation_errors_encountered = False
            for i, tc_data in enumerate(standard_tool_calls):
                 if isinstance(tc_data, dict) and tc_data.get('type') == 'function' and isinstance(tc_data.get('function'), dict):
                     try:
                          func_data = tc_data['function']
                          # Ensure 'arguments' exists and is a string before validation
                          if 'arguments' not in func_data:
                              logger.warning(f"Standard tool call item {i} missing 'arguments', defaulting to empty JSON string.")
                              func_data['arguments'] = "{}"
                          elif not isinstance(func_data['arguments'], str):
                              logger.warning(f"Standard tool call item {i} 'arguments' is not a string (type: {type(func_data['arguments'])}), attempting to dump: {func_data['arguments']}")
                              try:
                                  func_data['arguments'] = json.dumps(func_data['arguments'])
                              except (TypeError, Exception) as dump_err:
                                   logger.error(f"Failed to dump arguments to JSON string for tool call item {i}: {dump_err}. Setting arguments to empty JSON string.")
                                   func_data['arguments'] = "{}"
                                   validation_errors_encountered = True
                                   continue # Skip this tool call

                          # Validate arguments string is valid JSON before creating ToolCall obj
                          try:
                              _ = json.loads(func_data['arguments'])
                          except json.JSONDecodeError as json_err:
                              logger.error(f"Standard tool call item {i} 'arguments' is not valid JSON: {json_err}. Args: '{func_data['arguments']}'")
                              func_data['arguments'] = "{}" # Reset to valid empty JSON on error
                              validation_errors_encountered = True
                              # Continue processing the tool call with empty args if possible, or mark error

                          # Create ToolCall object (Pydantic handles internal validation)
                          validated_call = ToolCall(**tc_data)
                          parsed_list.append(validated_call)
                          logger.debug(f"Successfully parsed standard tool call {i}: {validated_call.function.name}")

                     except ValidationError as e_val:
                          logger.error(f"Pydantic validation failed for standard tool_call item {i}: {e_val}. Data: {tc_data}")
                          validation_errors_encountered = True
                     except Exception as e_parse:
                          logger.error(f"Unexpected error parsing standard tool_call item {i}: {e_parse}. Data: {tc_data}")
                          validation_errors_encountered = True
                 else:
                      logger.error(f"Invalid structure for standard tool call item {i}, skipping: {tc_data}")
                      validation_errors_encountered = True

            self.tool_calls = parsed_list
            assistant_content_text = raw_content_str # Keep any text content alongside standard tools
            # If we successfully parsed standard calls, the finish reason should be 'tool_calls'
            if self.tool_calls:
                 finish_reason = "tool_calls"
            # If validation errors occurred, append info to content? Or rely on logs? Let's log.
            if validation_errors_encountered:
                 logger.warning("Some standard tool calls failed validation/parsing.")
                 # Optionally add error note to content:
                 # assistant_content_text = (assistant_content_text or "") + "\n[Agent Note: Some tool calls had parsing errors, check logs.]"

        # 2. If no standard calls AND pythonic fallback enabled AND content exists, try pythonic parse
        elif use_pythonic_fallback and raw_content_str:
            logger.info("Standard 'tool_calls' empty/missing. Attempting pythonic parsing of 'content' as fallback.")
            parsed_calls, sanitized_content = _parse_pythonic_tool_calls(raw_content_str)
            if parsed_calls:
                self.tool_calls = parsed_calls
                assistant_content_text = sanitized_content if sanitized_content else None
                finish_reason = "tool_calls" # Override finish reason if pythonic calls found
                logger.info(f"Successfully parsed {len(self.tool_calls)} calls using pythonic fallback.")
            else:
                logger.info("No valid pythonic tool calls found in content during fallback.")
                assistant_content_text = raw_content_str # Use original content if no calls parsed

        # 3. Otherwise (no standard calls, pythonic fallback off or failed), treat content as text
        else:
            logger.info("No tool calls found via standard field or pythonic fallback. Treating content as text response.")
            assistant_content_text = raw_content_str

        # Ensure self.tool_calls is always a list
        if not isinstance(self.tool_calls, list):
            logger.error(f"Internal state error: self.tool_calls became non-list ({type(self.tool_calls)}). Resetting.")
            self.tool_calls = []

        # Log final decisions before adding to memory
        logger.info(f"✨ {self.name}'s final thoughts for memory: {assistant_content_text or '<no text content>'}")
        if self.tool_calls:
            logger.info(f"🛠️ {self.name} proceeding with {len(self.tool_calls)} parsed tools")
            logger.info(f"🧰 Parsed Tools: {[getattr(call.function, 'name', 'N/A') for call in self.tool_calls]}")
        else:
             logger.info(f"🛠️ {self.name} proceeding with no tool calls.")

        # Add the final assistant message to memory
        try:
            final_content_str = str(assistant_content_text) if assistant_content_text is not None else None
            # Convert ToolCall objects back to dicts for storage in Message
            tool_calls_dicts = [tc.model_dump(exclude_unset=True) for tc in self.tool_calls] if self.tool_calls else None

            assistant_msg_dict = {
                "role": "assistant",
                "content": final_content_str,
                # Only include tool_calls key if it's not None/empty
                **({"tool_calls": tool_calls_dicts} if tool_calls_dicts else {})
            }
            # Clean dict from None values before creating Message
            assistant_msg_dict_cleaned = {k: v for k, v in assistant_msg_dict.items() if v is not None}

            # Validate with Pydantic Message model before adding
            assistant_msg = Message(**assistant_msg_dict_cleaned)
            self.memory.add_message(assistant_msg)
            logger.debug(f"Added assistant message to memory: {assistant_msg_dict_cleaned}")

        except ValidationError as e:
            logger.error(f"Pydantic validation failed creating assistant message: {e}. Data: {assistant_msg_dict_cleaned}", exc_info=True)
            self.state = AgentState.ERROR; return False
        except Exception as e:
             logger.error(f"Error creating or adding assistant message to memory: {e}. Data: {assistant_msg_dict_cleaned}", exc_info=True)
             self.state = AgentState.ERROR; return False

        # Decide if agent should act based on having tools or producing text content
        # The LLM might naturally stop without doing anything, which is valid.
        should_act = bool(self.tool_calls or assistant_content_text)

        # Check finish reason: if 'stop' and no action planned, agent might be done or confused.
        if finish_reason == "stop" and not should_act:
            logger.info(f"Finish reason is 'stop' and no tools or content generated by LLM. Agent will not act.")
        elif finish_reason != "tool_calls" and self.tool_calls:
            logger.warning(f"Finish reason was '{finish_reason}' but tool calls were parsed. Proceeding with tools.")
        elif finish_reason == "tool_calls" and not self.tool_calls:
             logger.warning(f"Finish reason was 'tool_calls' but no valid tools were parsed (check validation errors). Treating as text response.")
             should_act = bool(assistant_content_text) # Act only if there's text content


        logger.info(f"Agent proceeding to next step (act)? {should_act} (Finish Reason: {finish_reason}, Has Tools: {bool(self.tool_calls)}, Has Text: {bool(assistant_content_text)})")
        return should_act


    async def act(self) -> str:
        """Execute tool calls stored in self.tool_calls"""
        if not self.tool_calls:
            if self.tool_choices == ToolChoice.REQUIRED:
                logger.error("Tool choice was 'required' but no valid tool calls were available to execute.")
                last_assistant_message = next((msg for msg in reversed(self.messages) if msg.role == 'assistant'), None)
                error_content = last_assistant_message.content if last_assistant_message else TOOL_CALL_REQUIRED
                # Add error message to memory? Or just return? Let's return for now.
                return f"Error: {error_content}"
            # If not required, check if there was text content from the assistant
            last_assistant_message = next((msg for msg in reversed(self.messages) if msg.role == 'assistant'), None)
            # If the last message had content but no tools, return that content.
            if last_assistant_message and last_assistant_message.content and not last_assistant_message.tool_calls:
                 logger.info("No tools to execute, returning last assistant text content.")
                 return last_assistant_message.content
            else:
                 logger.info("No tools to execute and no final text content from assistant.")
                 return "No action taken." # Or maybe a more specific "Task finished."?

        results_str_parts = []
        if not isinstance(self.tool_calls, list):
             logger.error(f"act: self.tool_calls is not a list ({type(self.tool_calls)}), cannot execute.")
             return "Error: Internal agent state inconsistency regarding tool calls."

        # Execute parsed tool calls
        for command in self.tool_calls:
            if not isinstance(command, ToolCall):
                 logger.error(f"act: Skipping invalid item in self.tool_calls, expected ToolCall, got {type(command)}: {command}")
                 results_str_parts.append("Error: Invalid command structure encountered.")
                 continue

            self._current_base64_image = None # Reset image for each tool call
            tool_output_or_error = await self.execute_tool(command)

            logger.info(f"Tool '{getattr(command.function, 'name', 'unknown')}' completed.")
            logger.debug(f"Observation from tool '{getattr(command.function, 'name', 'unknown')}': {tool_output_or_error[:500]}...")

            # Add the tool result message to memory
            try:
                tool_msg = Message.tool_message(
                    content=tool_output_or_error,
                    tool_call_id=command.id,
                    name=command.function.name, # Optional but can be helpful
                    base64_image=getattr(self, '_current_base64_image', None)
                )
                self.memory.add_message(tool_msg)
                results_str_parts.append(tool_output_or_error) # Append result for potential final summary
            except (AttributeError, ValidationError) as e:
                 logger.error(f"Error creating tool message, likely invalid ToolCall structure or result: {e}. Command was: {command}", exc_info=True)
                 # Add an error message to memory instead?
                 error_tool_msg = Message(role="tool", tool_call_id=getattr(command, 'id', 'UNKNOWN_ID'), content=f"Error processing result for this tool call: {e}")
                 self.memory.add_message(error_tool_msg)
                 results_str_parts.append(f"Error processing result for tool call ID {getattr(command, 'id', 'UNKNOWN_ID')}")

            # Check if the agent state was changed to FINISHED by the tool
            if self.state == AgentState.FINISHED:
                logger.info(f"Agent state set to FINISHED by tool '{getattr(command.function, 'name', 'unknown')}'. Stopping action loop.")
                break

        # Return a combined string of results (or a status message)
        # This return value might not be directly used if the loop continues, but useful if it's the final step.
        return "\n\n".join(results_str_parts) if results_str_parts else "Tool execution completed with no results."


    async def execute_tool(self, command: ToolCall) -> str:
            """Execute a single tool call with validation and robust argument handling"""
            # Basic validation of the input command structure
            if not isinstance(command, ToolCall) or not command.function or not command.function.name or not command.id:
                logger.error(f"execute_tool received invalid command object structure: {command}")
                return "Error: Invalid command format provided to execute_tool"

            name = command.function.name
            tool_call_id = command.id
            # Ensure arguments is treated as a string, default to empty JSON object string if None/missing
            arguments_str = command.function.arguments if command.function.arguments is not None else "{}"

            logger.debug(f"Executing tool '{name}' (ID: {tool_call_id}) with raw arguments string: '{arguments_str}'")

            # Find the tool instance
            tool_instance = self.available_tools.get_tool(name)
            if not tool_instance:
                logger.error(f"Tool '{name}' (ID: {tool_call_id}) not found in available tools.")
                return f"Error: Unknown tool '{name}'"

            args_dict: Dict[str, Any] = {}
            exec_result: Any = None
            observation: str = "" # Initialize observation string

            try:
                # Argument Parsing (Expects valid JSON string from standard tool calls)
                try:
                    args_dict = json.loads(arguments_str)
                    if not isinstance(args_dict, dict):
                        raise ValueError("Parsed arguments are not a dictionary.")
                except (json.JSONDecodeError, ValueError) as e:
                    logger.error(f"Argument JSON parsing failed for tool '{name}' (ID: {tool_call_id}): {e}. Args string: '{arguments_str}'", exc_info=True)
                    return f"Error: Arguments for tool '{name}' were not valid JSON or not a dictionary. Received: '{arguments_str[:100]}...'"

                # Argument Validation/Correction (Based on tool's Pydantic model or schema)
                # (Keep your existing validation logic here - using args_schema or required params)
                if hasattr(tool_instance, 'args_schema') and callable(getattr(tool_instance.args_schema, "model_validate", None)):
                    try:
                        validated_args = tool_instance.args_schema.model_validate(args_dict)
                        args_dict = validated_args.model_dump()
                        logger.debug(f"Arguments successfully validated against schema for tool '{name}'.")
                    except ValidationError as e:
                        logger.error(f"Argument validation failed for tool '{name}' (ID: {tool_call_id}) against its schema: {e}. Provided args: {args_dict}", exc_info=True)
                        error_details = "; ".join([f"{err['loc'][0]}: {err['msg']}" for err in e.errors()])
                        return f"Error: Invalid arguments for tool '{name}'. Details: {error_details}. Provided: {json.dumps(args_dict)}"
                else:
                    if hasattr(tool_instance, 'parameters') and isinstance(tool_instance.parameters, dict):
                        required_params = tool_instance.parameters.get("required", [])
                        missing_required = [rp for rp in required_params if rp not in args_dict]
                        if missing_required:
                            error_msg = f"Error: Missing required argument(s) for tool '{name}': {', '.join(missing_required)}. Provided: {list(args_dict.keys())}"
                            logger.error(f"{error_msg} (ID: {tool_call_id})")
                            return error_msg


                # Special Handling for Python Code (Unescape *after* JSON parsing/validation)
                if name == "python_execute":
                    if "code" not in args_dict or not isinstance(args_dict.get("code"), str):
                        logger.error(
                            "Missing or invalid 'code' argument (string expected) for python_execute. "
                            f"Args: {args_dict}"
                        )
                        return (
                            "Error: Missing or invalid 'code' argument "
                            "(must be a string) for python_execute."
                        )

                    try:
                        # ⬇️  **FIXED INDENTATION – these two lines must be nested inside the `if` block**
                        if args_dict["code"]:
                            args_dict["code"] = unescape_code(args_dict["code"])
                            logger.debug(
                                f"Unescaped code for python_execute: {args_dict['code'][:100]}..."
                            )
                    except Exception as unescape_err:
                        logger.error(
                            f"Error during code unescaping for python_execute: {unescape_err}",
                            exc_info=True,
                        )
                        return (
                            "Error: Failed to process 'code' argument for "
                            f"python_execute: {unescape_err}"
                        )

                # Tool Execution
                logger.info(f"🔧 Executing tool: '{name}' (ID: {tool_call_id}) with validated args: {args_dict}...")
                exec_result = await tool_instance.execute(**args_dict)

                # -------- START: Modified Result Handling --------
                await self._handle_special_tool(name=name, result=exec_result) # Check if tool finishes the task

                if isinstance(exec_result, ToolResult):
                    # CRITICAL CHANGE: Use str(exec_result) to get the observation/error string.
                    # Assumes ToolResult.__str__ provides the necessary output for the LLM.
                    observation = str(exec_result)
                    if exec_result.base64_image:
                        self._current_base64_image = exec_result.base64_image
                        logger.debug(f"Tool '{name}' returned base64 image data.")
                    # Log the error specifically if the ToolResult indicates one
                    if exec_result.error:
                        logger.warning(f"Tool '{name}' (ID: {tool_call_id}) execution resulted in an error state: {exec_result.error}")
                        # Ensure the observation *is* the error message if the tool failed
                        observation = f"Tool Error: {exec_result.error}"
                elif exec_result is None:
                    observation = f"Tool '{name}' completed successfully with no output."
                else:
                    observation = str(exec_result) # Convert other return types to string

                # -------- END: Modified Result Handling --------


                # Apply observation length limit if configured
                if isinstance(self.max_observe, int) and self.max_observe > 0 and len(observation) > self.max_observe:
                    truncated_len = self.max_observe
                    observation = observation[:truncated_len] + f"... (truncated from {len(observation)} chars)"
                    logger.info(f"Observation for tool '{name}' truncated to {truncated_len} chars.")

                logger.info(f"✅ Tool '{name}' (ID: {tool_call_id}) execution status logged. Observation prepared.")
                logger.debug(f"Observation for LLM:\n{observation}")
                return observation

            except ValidationError as e: # Catch Pydantic validation errors during tool execution
                error_msg = f"Internal validation error during execution of tool '{name}' (ID: {tool_call_id}): {e}"
                logger.error(error_msg, exc_info=True)
                return f"Error: Tool '{name}' encountered an internal validation issue. Details: {str(e)[:200]}"
            except ValueError as e: # Catch common arg processing errors within tool execution
                error_msg = f"Value error during execution of tool '{name}' (ID: {tool_call_id}): {e}. Args: {args_dict}"
                logger.error(error_msg, exc_info=True)
                return f"Error: Invalid value provided during execution of tool '{name}'. Details: {str(e)[:100]}"
            except TypeError as e: # Catch type errors
                error_msg = f"Type error during execution of tool '{name}' (ID: {tool_call_id}): {e}. Args: {args_dict}"
                logger.error(error_msg, exc_info=True)
                return f"Error: Type mismatch during execution of tool '{name}'. Details: {str(e)[:100]}"
            except Exception as e: # Catch other unexpected errors *during* tool execution
                error_msg = f"Unexpected error during execution of tool '{name}' (ID: {tool_call_id}): {str(e)}"
                if isinstance(exec_result, ToolResult) and exec_result.error: # Check if ToolResult captured the error
                    error_msg = f"Tool '{name}' failed: {exec_result.error}"
                logger.error(f"⚠️ {error_msg}", exc_info=True)
                return f"Error: Tool '{name}' encountered an unexpected issue during execution. Details: {str(e)[:200]}"
            

    # --- Helper methods for special tools, cleanup, run ---
    # --- These remain unchanged from your original code ---
    async def _handle_special_tool(self, name: str, result: Any, **kwargs):
        if not self._is_special_tool(name): return
        if self._should_finish_execution(name=name, result=result, **kwargs):
            logger.info(f"🏁 Special tool '{name}' has completed the task!")
            self.state = AgentState.FINISHED

    @staticmethod
    def _should_finish_execution(**kwargs) -> bool: return True

    def _is_special_tool(self, name: str) -> bool:
        return name.lower() in [n.lower() for n in self.special_tool_names]

    async def cleanup(self):
        logger.info(f"🧹 Cleaning up resources for agent '{self.name}'...")
        if hasattr(self, 'available_tools') and isinstance(self.available_tools, ToolCollection):
            for tool_name, tool_instance in self.available_tools.tool_map.items():
                if hasattr(tool_instance, "cleanup") and asyncio.iscoroutinefunction(getattr(tool_instance, "cleanup", None)):
                    try:
                        logger.debug(f"🧼 Cleaning up tool: {tool_name}")
                        await tool_instance.cleanup()
                    except Exception as e:
                        logger.error(f"🚨 Error cleaning up tool '{tool_name}': {e}", exc_info=True)
        else:
            logger.warning("Cannot perform cleanup: 'available_tools' not found or not a ToolCollection.")
        logger.info(f"✨ Cleanup complete for agent '{self.name}'.")

    async def run(self, request: Optional[str] = None) -> str:
        try:
            return await super().run(request)
        except Exception as e:
            logger.error(f"Agent run loop encountered an unhandled error: {e}", exc_info=True)
            raise
        finally:
            await self.cleanup()

# --- End of file: app/agent/toolcall.py ---