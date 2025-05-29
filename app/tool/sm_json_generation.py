# Inside app/tool/json_generation.py

import json
import re
import yaml
import tempfile
import os
import logging # Make sure logging is imported if not already
import shutil # Import the shutil module for moving files
import subprocess # <-- ADDED IMPORT

from app.tool.base import BaseTool, ToolResult
from app.llm import LLM
from app.logger import logger # Assuming your logger is set up here

# Ensure these prompts are loaded correctly in your application context
try:
    from app.prompt.sm_json_guideline import SYSTEM_PROMPT_1, SYSTEM_PROMPT_2
except ImportError:
    logger.error("Failed to import system prompts for JSON generation. Using placeholder text.")
    # Define placeholders if import fails to avoid NameError, but log the issue.
    SYSTEM_PROMPT_1 = "Placeholder Guideline 1: Define JSON structure..."
    SYSTEM_PROMPT_2 = "Placeholder Guideline 2: Follow these rules..."

from typing import Optional # Union was imported but not used, can be removed if not needed elsewhere

# --- The integrated verifier logic previously here should be REMOVED ---

class SM_JsonGenerationTool(BaseTool):
    name: str = "sm_json_generator_and_verifier"
    description: str = ("Generates a structured JSON configuration based on specific guidelines "
                        "and a user request, then converts it to YAML and verifies it against "
                        "SimpleMind rules by calling an external verifier script. Attempts to self-correct based on errors up to a limit. " # MODIFIED description
                        "Returns the JSON only if verification succeeds.")
    parameters: dict = {
        "type": "object",
        "properties": {
            "user_request": {
                "type": "string",
                "description": "The core topic or entity for JSON generation (e.g., 'heart segmentation', 'trachea analysis'). Do not include specific parameters, only the topic.",
            }
            # If you want to pass check_tf2/check_dt flags from the tool's caller:
            # "check_tf2": {"type": "boolean", "description": "Enable TF2 YAML checks for the external verifier.", "default": True},
            # "check_dt": {"type": "boolean", "description": "Enable Decision Tree YAML checks for the external verifier.", "default": True},
        },
        "required": ["user_request"],
    }

    llm: LLM

    # Define path to the agent inventory
    inventory_path: str = "/cvib2/apps/personal/jink/OpenManus_vllm/app/tool/detailed_agent_inventory.yaml"
    # Path to the external verifier script
    verifier_script_path: str = "/cvib2/apps/personal/wasil/lib/sm/sm_verifier/simplemind/utils/verifier_v2.py" # <-- ADDED
    max_retries: int = 5

    def _extract_json_from_markdown(self, text: str) -> Optional[str]:
        """Attempts to extract JSON content from a markdown code block."""
        logger.debug(f"Raw text received for JSON extraction: {text[:500]}...")
        match = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text, re.DOTALL | re.MULTILINE)
        if match:
            logger.debug("Found JSON within markdown block.")
            return match.group(1).strip()
        else:
            stripped_text = text.strip()
            if stripped_text.startswith('{') and stripped_text.endswith('}'):
                try:
                    json.loads(stripped_text)
                    logger.debug("Assuming raw text is JSON (starts/ends with {} and parses).")
                    return stripped_text
                except json.JSONDecodeError:
                    logger.warning("Text starts/ends with {} but failed to parse as JSON.")
                    return None
            else:
                logger.warning("Could not find JSON in markdown block or as raw object.")
                return None

    def _create_feedback_prompt(self, original_request: str, error_message: str, problematic_content: Optional[str]) -> str:
        """Creates a prompt for the LLM to self-correct based on feedback."""
        if problematic_content is None:
            problematic_content = "[No specific content could be extracted or generated]"

        guideline1 = getattr(self, 'SYSTEM_PROMPT_1', SYSTEM_PROMPT_1)
        guideline2 = getattr(self, 'SYSTEM_PROMPT_2', SYSTEM_PROMPT_2)

        feedback_prompt = f"""The previous attempt to generate a SimpleMind configuration JSON for the request '{original_request}' failed the verification checks.

Verification Error:
--- ERROR ---
{error_message}
--- END ERROR ---

The problematic JSON/YAML content that caused the error was (or the raw LLM output if extraction failed):
--- PROBLEMATIC CONTENT ---
{problematic_content}
--- END PROBLEMATIC CONTENT ---

Please analyze the error message and the problematic content carefully.
You MUST strictly follow the original guidelines provided:
--- GUIDELINE 1 ---
{guideline1}
--- END GUIDELINE 1 ---
--- GUIDELINE 2 ---
{guideline2}
--- END GUIDELINE 2 ---

Correct the identified error based on the feedback AND the guidelines. Ensure the structure, agent names, parameters, inputs, and ordering adhere to the SimpleMind rules described in the guidelines and implied by the error message.
Provide the complete, corrected, valid JSON object.
Generate *only* the raw JSON object without any markdown formatting, comments, or explanatory text surrounding it.
"""
        return feedback_prompt

    def _parse_verifier_output(self, output_str: str, error_str: str) -> tuple[bool, list[str]]: # <-- ADDED HELPER
        """Parses the stdout and stderr from the external verifier script."""
        lines = output_str.strip().split('\n')
        validation_passed = False
        feedback_msgs = []
        
        logger.debug(f"Verifier stdout to parse: {output_str}")
        if error_str:
            logger.warning(f"Verifier stderr: {error_str}")
            # Prepend stderr to feedback as it might contain critical errors from the script itself
            feedback_msgs.append(f"Verifier script error output (stderr):\n{error_str.strip()}")

        if not lines or not lines[0].strip(): # Handle empty or blank stdout
            if not error_str.strip(): # If stderr is also empty
                 feedback_msgs.append("Verifier script produced no output (stdout and stderr were empty).")
            # If only stderr had content, it's already added. Treat as failure.
            return False, feedback_msgs

        first_line_stdout = lines[0].strip()
        if first_line_stdout.startswith("Checks passed:"):
            try:
                status_str = first_line_stdout.split(":", 1)[1].strip()
                if status_str.lower() == 'true':
                    validation_passed = True
                elif status_str.lower() == 'false':
                    validation_passed = False
                else:
                    feedback_msgs.append(f"Could not parse 'Checks passed:' status value. Line: '{first_line_stdout}'")
                    # validation_passed remains False
            except IndexError:
                feedback_msgs.append(f"Malformed 'Checks passed:' line. Line: '{first_line_stdout}'")
                # validation_passed remains False
        else:
            feedback_msgs.append(f"Verifier stdout did not start with 'Checks passed:'. First line: '{first_line_stdout}'")
            # Add all stdout as feedback if format is unexpected
            feedback_msgs.extend([line.strip() for line in lines if line.strip()])
            return False, feedback_msgs # Unexpected format, treat as failure

        # Look for "Feedback:" section in the rest of stdout
        feedback_section_started = False
        for i in range(1, len(lines)):
            if lines[i].strip() == "Feedback:":
                feedback_section_started = True
                feedback_msgs.extend([line.strip() for line in lines[i+1:] if line.strip()])
                break
        
        if not validation_passed and not feedback_section_started:
            # If checks failed but no "Feedback:" line in stdout, remaining stdout lines (if any after the first) might be relevant.
            # Avoid duplicating the first line if it wasn't "Feedback:"
            additional_feedback_from_stdout = [line.strip() for line in lines[1:] if line.strip()]
            if additional_feedback_from_stdout:
                feedback_msgs.extend(additional_feedback_from_stdout)
        
        # Ensure there's at least one message if checks failed and nothing else was gathered
        if not validation_passed and not [msg for msg in feedback_msgs if msg and not msg.startswith("Verifier script error output (stderr):")]:
            feedback_msgs.append("Verification failed according to 'Checks passed: False', but no further specific feedback messages were found in stdout.")
        elif validation_passed and not feedback_msgs:
            feedback_msgs.append("Verification passed (no specific feedback messages from verifier).")
            
        return validation_passed, feedback_msgs

    async def execute(self, user_request: str, **kwargs) -> ToolResult:
        # Default to True for these flags if not provided by the caller
        check_tf2_yaml_flag = kwargs.get("check_tf2", False)
        check_dt_yaml_flag = kwargs.get("check_dt", False)

        logger.info(f"Executing {self.name} with request: {user_request[:100]}...")
        logger.info(f"External Verifier Flags -- TF2 checks: {'Enabled' if check_tf2_yaml_flag else 'Disabled'}, DT checks: {'Enabled' if check_dt_yaml_flag else 'Disabled'}")
        print(f"Executing {self.name} with request: {user_request[:100]}...")

        if not os.path.exists(self.inventory_path):
            logger.error(f"Agent inventory file not found at specified path: {self.inventory_path}")
            print(f"ERROR: Agent inventory file not found: {self.inventory_path}")
            return ToolResult(error=f"Configuration error: Agent inventory file missing at {self.inventory_path}")

        if not os.path.exists(self.verifier_script_path): # <-- ADDED CHECK
            logger.error(f"Verifier script not found at specified path: {self.verifier_script_path}")
            print(f"ERROR: Verifier script not found: {self.verifier_script_path}")
            return ToolResult(error=f"Configuration error: Verifier script missing at {self.verifier_script_path}")

        if not SYSTEM_PROMPT_1 or SYSTEM_PROMPT_1.startswith("Placeholder") or \
           not SYSTEM_PROMPT_2 or SYSTEM_PROMPT_2.startswith("Placeholder"):
            logger.warning("JSON generation guideline prompts might be missing or using placeholders.")

        last_error = "No attempts made."
        last_generated_json_str = None
        last_raw_llm_output = None
        current_prompt = f"{SYSTEM_PROMPT_1}\n {user_request}\n\n{SYSTEM_PROMPT_2}\n\nGenerate *only* the raw JSON object without any markdown formatting or explanatory text."

        for attempt in range(self.max_retries):
            logger.info(f"Attempt {attempt + 1}/{self.max_retries} for request: {user_request[:50]}...")
            print(f"Attempt {attempt + 1}/{self.max_retries} for request: {user_request[:50]}...")
            logger.debug(f"Prompt for attempt {attempt + 1}:\n{current_prompt[:500]}...")

            extracted_json_str = None
            # json_data = None # Removed as yaml_str is generated directly
            temp_yaml_path = None

            try:
                print(f"Calling LLM (Attempt {attempt+1})...")
                logger.info(f"Calling LLM with prompt length: {len(current_prompt)}")
                generated_raw_output = await self.llm.ask(
                    messages=[{"role": "user", "content": current_prompt}],
                    temperature=0.1 + (attempt * 0.1)
                )
                last_raw_llm_output = generated_raw_output
                logger.debug(f"Raw generated output from LLM (Attempt {attempt + 1}):\n{generated_raw_output}")
                print(f"LLM Raw Output (Attempt {attempt+1}) received.")

                print(f"Extracting JSON (Attempt {attempt+1})...")
                extracted_json_str = self._extract_json_from_markdown(generated_raw_output)
                last_generated_json_str = extracted_json_str

                if not extracted_json_str:
                    last_error = "Could not extract JSON object from LLM response."
                    logger.warning(f"Attempt {attempt + 1}: {last_error} LLM Output: {generated_raw_output[:300]}")
                    print(f"Attempt {attempt + 1} FAILED: {last_error}")
                    current_prompt = self._create_feedback_prompt(user_request, last_error, generated_raw_output)
                    continue
                
                print(f"JSON extracted successfully (Attempt {attempt+1}).")
                print(f"Validating JSON syntax (Attempt {attempt+1})...")
                try:
                    # Load JSON to validate and to prepare for YAML dump
                    json_data_for_yaml = json.loads(extracted_json_str)
                    logger.info(f"Attempt {attempt + 1}: JSON syntax validation successful.")
                    print(f"JSON syntax valid (Attempt {attempt+1}).")
                except json.JSONDecodeError as json_err:
                    last_error = f"Generated content is not valid JSON: {json_err}"
                    logger.warning(f"Attempt {attempt + 1}: {last_error}. Extracted: {extracted_json_str[:300]}")
                    print(f"Attempt {attempt + 1} FAILED: {last_error}")
                    current_prompt = self._create_feedback_prompt(user_request, last_error, extracted_json_str)
                    continue

                print(f"Converting to YAML and saving temporarily (Attempt {attempt+1})...")
                try:
                    yaml_str = yaml.dump(json_data_for_yaml, default_flow_style=False, sort_keys=False, allow_unicode=True, indent=2)
                    
                    persistent_yaml_filename = f"generated_config_attempt_{attempt + 1}.yaml"
                    try:
                        with open(persistent_yaml_filename, 'w', encoding='utf-8') as outfile:
                            outfile.write(yaml_str)
                        logger.info(f"Attempt {attempt + 1}: Saved generated YAML to ./{persistent_yaml_filename}")
                    except IOError as e_io:
                        logger.error(f"Attempt {attempt + 1}: Failed to save YAML to ./{persistent_yaml_filename}: {e_io}")

                    with tempfile.NamedTemporaryFile(mode='w', suffix=".yaml", delete=False, encoding='utf-8') as temp_yaml_file:
                        temp_yaml_path = temp_yaml_file.name
                        temp_yaml_file.write(yaml_str)
                    
                    logger.info(f"Attempt {attempt + 1}: YAML conversion successful. Written to temp file: {temp_yaml_path}")
                    print(f"YAML written to {temp_yaml_path} (Attempt {attempt+1}).")
                except Exception as convert_err:
                    last_error = f"Failed during JSON to YAML conversion or temp file saving: {convert_err}"
                    logger.exception(f"Attempt {attempt + 1}: {last_error}") # Log full exception
                    print(f"Attempt {attempt + 1} FAILED: {last_error}")
                    current_prompt = self._create_feedback_prompt(user_request, last_error, extracted_json_str)
                    if temp_yaml_path and os.path.exists(temp_yaml_path):
                        try: os.remove(temp_yaml_path)
                        except OSError: pass
                    continue

                # --- MODIFIED VERIFICATION STEP ---
                print(f"Running SimpleMind YAML verification via external script (Attempt {attempt+1})...")
                logger.info(f"Attempt {attempt + 1}: Calling external verifier: {self.verifier_script_path} on {temp_yaml_path} with inventory: {self.inventory_path}")
                
                cmd = ['python', self.verifier_script_path, temp_yaml_path, self.inventory_path]
                if check_tf2_yaml_flag:
                    cmd.append('--check_tf2_yaml')
                if check_dt_yaml_flag:
                    cmd.append('--check_dt_yaml')
                
                logger.debug(f"Verifier command: {' '.join(cmd)}")
                validation_passed = False # Default to ensure it's set
                feedback_msgs = []    # Default to ensure it's set

                try:
                    # Increased timeout, adjust as needed
                    process = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=120) 
                    
                    logger.info(f"Verifier script stdout:\n{process.stdout}")
                    if process.stderr:
                        logger.warning(f"Verifier script stderr:\n{process.stderr}")

                    validation_passed, feedback_msgs = self._parse_verifier_output(process.stdout, process.stderr)

                    # Additional check if script itself had an issue not caught by parsing
                    if process.returncode != 0 and validation_passed:
                        logger.warning(f"Verifier script exited with code {process.returncode} but parsing indicated success. Overriding to failure.")
                        validation_passed = False
                        feedback_msgs.append(f"Warning: Verifier script exited with error code {process.returncode} despite parseable 'Checks passed: True'. Treating as failure.")
                    elif process.returncode !=0 and not feedback_msgs: # If return code indicates error and no feedback msgs parsed
                        feedback_msgs.append(f"Verifier script exited with error code {process.returncode}. Stderr: {process.stderr or 'empty'}")


                    if validation_passed:
                        logger.info(f"Attempt {attempt + 1}: External SimpleMind YAML verification successful. Feedback: {'; '.join(feedback_msgs)}")
                        print(f"SUCCESS: External SimpleMind YAML verification passed (Attempt {attempt+1})!")

                        final_yaml_path = "/radraid2/mwahianwar/agentic_ai/simplemind/simplemind/example/example.yaml"
                        try:
                            shutil.copy(temp_yaml_path, final_yaml_path) 
                            logger.info(f"Successfully copied final verified YAML to: {final_yaml_path} from temp file {temp_yaml_path}")
                        except Exception as e_move:
                            logger.error(f"Error copying verified YAML from {temp_yaml_path} to {final_yaml_path}: {e_move}")
                            print(f"Error saving final YAML: {e_move}")
                        
                        return ToolResult(output=extracted_json_str) 
                    else:
                        error_str_from_validation = "; ".join(feedback_msgs) if feedback_msgs else "Verification failed with no specific messages."
                        last_error = f"Generated configuration failed SimpleMind YAML verification (external script): {error_str_from_validation}"
                        logger.warning(f"Attempt {attempt + 1}: {last_error}")
                        print(f"Attempt {attempt + 1} FAILED VERIFICATION: {last_error}")
                        current_prompt = self._create_feedback_prompt(user_request, last_error, extracted_json_str)

                except subprocess.TimeoutExpired:
                    last_error = "External verifier script timed out after 120 seconds."
                    logger.error(f"Attempt {attempt + 1}: {last_error}")
                    print(f"Attempt {attempt + 1} FAILED: {last_error}")
                    current_prompt = self._create_feedback_prompt(user_request, last_error, extracted_json_str)
                except FileNotFoundError: # e.g. python interpreter or script itself not found
                    last_error = f"Error calling external verifier: 'python' or script '{self.verifier_script_path}' not found."
                    logger.exception(f"Attempt {attempt + 1}: {last_error}")
                    print(f"Attempt {attempt + 1} FAILED: {last_error}")
                    # This is a fatal setup error, retrying might not help
                    return ToolResult(error=last_error)
                except Exception as verify_err: # Other subprocess or parsing errors
                    last_error = f"Unexpected error during external SimpleMind YAML verification call: {verify_err}"
                    logger.exception(f"Attempt {attempt + 1}: {last_error}")
                    print(f"Attempt {attempt + 1} FAILED during verification call: {last_error}")
                    current_prompt = self._create_feedback_prompt(user_request, last_error, extracted_json_str)
                # --- END MODIFIED VERIFICATION STEP ---

            except Exception as outer_err:
                last_error = f"Unexpected error during generation/processing in attempt {attempt + 1}: {outer_err}"
                logger.exception(last_error) # Log full exception
                print(f"Attempt {attempt + 1} FAILED (Outer Exception): {last_error}")
                problematic_content_for_feedback = last_generated_json_str if last_generated_json_str else last_raw_llm_output
                current_prompt = self._create_feedback_prompt(user_request, last_error, problematic_content_for_feedback)
            
            finally:
                if temp_yaml_path and os.path.exists(temp_yaml_path):
                    try:
                        os.remove(temp_yaml_path)
                        logger.debug(f"Cleaned up temporary YAML file: {temp_yaml_path}")
                    except OSError as e_remove:
                        logger.error(f"Error removing temporary file {temp_yaml_path}: {e_remove}")
                        print(f"Error removing temporary file {temp_yaml_path}: {e_remove}")
            
            print("-" * 20) 

        final_error_message = f"Failed to generate valid and verified SimpleMind configuration after {self.max_retries} attempts. Last error: {last_error}"
        logger.error(final_error_message)
        if last_generated_json_str:
            logger.error(f"Last generated JSON content that failed: {last_generated_json_str[:1000]}")
        elif last_raw_llm_output:
            logger.error(f"Last raw LLM output that failed extraction or parsing: {last_raw_llm_output[:1000]}")

        print(f"FINAL FAILURE: {final_error_message}")
        return ToolResult(
            error=final_error_message,
            output={"last_error": last_error, "last_failed_json_or_raw_output": last_generated_json_str if last_generated_json_str else last_raw_llm_output}
        )

# Example usage block (if __name__ == '__main__': ...) should remain unchanged if present