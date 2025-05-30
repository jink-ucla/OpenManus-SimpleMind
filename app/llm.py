# --- Start of file: app/llm.py ---

import math
import json
import asyncio
from typing import Dict, List, Optional, Union, AsyncGenerator, Any

import httpx # Use httpx for async requests
# Use transformers for tokenizer - make sure it's installed
from transformers import AutoTokenizer, PreTrainedTokenizerBase #
from tenacity import ( #
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    retry_if_exception,   # <-- add this import
    wait_random_exponential,
)

# Assuming these are still relevant or adapted from your project structure
from app.config import LLMSettings, config # Assuming LLMSettings is defined here
from app.exceptions import TokenLimitExceeded
from app.logger import logger # Assuming a logger is set up
from app.schema import (
    ROLE_VALUES,
    TOOL_CHOICE_TYPE,
    TOOL_CHOICE_VALUES,
    Message,
    ToolChoice,
)

# VLLM API suffix (should likely be empty as base_url includes /v1)
# If base_url is http://host:port/v1, this should be ""
# If base_url is http://host:port, this should be "/v1/chat/completions" #
# Assuming base_url includes /v1 based on typical OpenAI-compatible servers.
VLLM_API_SUFFIX = "/chat/completions" # Set path relative to base_url #

# --- Token Counter with Conditional Image Counting ---
class TokenCounter: #
    # Using OpenAI constants for image token counting when enabled
    BASE_MESSAGE_TOKENS = 4 # Base overhead per message #
    FORMAT_TOKENS = 2 # Tokens for final prompt structure (e.g., <|im_start|>) #
    LOW_DETAIL_IMAGE_TOKENS = 85 #
    HIGH_DETAIL_TILE_TOKENS = 170 #
    MAX_SIZE = 2048 #
    HIGH_DETAIL_TARGET_SHORT_SIDE = 768 #
    TILE_SIZE = 512 #

    # Store multimodal support flag
    def __init__(self, tokenizer: PreTrainedTokenizerBase, supports_multimodal: bool): #
        if tokenizer is None: # #
            raise ValueError("Tokenizer cannot be None for TokenCounter") # #
        self.tokenizer = tokenizer #
        self.supports_multimodal = supports_multimodal #
        logger.info(f"TokenCounter initialized with tokenizer: {tokenizer.name_or_path}, Multimodal support: {supports_multimodal}") #

    def count_text(self, text: str) -> int: #
        """Calculate tokens for a text string using the loaded tokenizer"""
        if not text: #
            return 0 # #
        # Encode without adding special tokens here, as they are handled per message/format
        # Using encode directly gives token IDs, len() gives the count.
        return len(self.tokenizer.encode(text, add_special_tokens=False)) # #

    def _calculate_high_detail_tokens(self, width: int, height: int) -> int: #
        """Calculate tokens for high detail images based on dimensions (OpenAI logic)."""
        if width <= 0 or height <= 0: return 0 #
        # Step 1: Scale to fit in MAX_SIZE x MAX_SIZE square
        if width > self.MAX_SIZE or height > self.MAX_SIZE: # #
            scale = self.MAX_SIZE / max(width, height) #
            width = int(width * scale) #
            height = int(height * scale) # #

        # Step 2: Scale so shortest side is HIGH_DETAIL_TARGET_SHORT_SIDE
        # Avoid division by zero if image is extremely thin/short after first scaling
        if min(width, height) == 0: return self.LOW_DETAIL_IMAGE_TOKENS #
        scale = self.HIGH_DETAIL_TARGET_SHORT_SIDE / min(width, height) # #
        scaled_width = int(width * scale) #
        scaled_height = int(height * scale) #

        # Step 3: Count number of 512px tiles
        tiles_x = math.ceil(scaled_width / self.TILE_SIZE) #
        tiles_y = math.ceil(scaled_height / self.TILE_SIZE) # #
        total_tiles = tiles_x * tiles_y #

        # Step 4: Calculate final token count
        return (total_tiles * self.HIGH_DETAIL_TILE_TOKENS) + self.LOW_DETAIL_IMAGE_TOKENS # #

    def count_image(self, image_item: dict) -> int: #
        """
        Calculate tokens for an image based on detail level and dimensions (OpenAI logic). # #
        Only applies if supports_multimodal is True.
        Assumes image_item structure like {"type": "image_url", "image_url": {"url": "...", "detail": "high", "dimensions": [w, h]}}
        """
        # If multimodal is not supported by the config, count images as 0 tokens.
        if not self.supports_multimodal: # #
            return 0 #

        # --- Use OpenAI's counting logic ---
        # Default detail to 'high' as per common practice if not specified
        detail = image_item.get("image_url", {}).get("detail", "high") #

        if detail == "low": #
            return self.LOW_DETAIL_IMAGE_TOKENS # #

        # For 'high' or unspecified detail, attempt calculation based on dimensions
        dimensions = image_item.get("image_url", {}).get("dimensions") #

        if dimensions and isinstance(dimensions, (list, tuple)) and len(dimensions) == 2: #
            width, height = dimensions # #
            if isinstance(width, int) and isinstance(height, int): #
                logger.debug(f"Calculating image tokens using provided dimensions: {width}x{height}") #
                return self._calculate_high_detail_tokens(width, height) # #
            else: #
                logger.warning(f"Image dimensions provided but not integers: {dimensions}. Using default calculation.") #
                return self._calculate_high_detail_tokens(1024, 1024) # Default if format is wrong #
        else: #
            # If dimensions not provided or invalid, use a default (e.g., 1024x1024 for high)
            logger.warning("Image dimensions not provided or invalid in message item, using default high-detail calculation (1024x1024).") # #
            return self._calculate_high_detail_tokens(1024, 1024) # ~765 tokens as default #


    def count_content(self, content: Union[str, List[Union[str, dict]]]) -> int: #
        """Calculate tokens for message content, including images if multimodal is supported."""
        if content is None: return 0 #
        if isinstance(content, str): #
            return self.count_text(content) #

        if isinstance(content, list): # #
            token_count = 0 #
            for item in content: #
                if isinstance(item, dict): #
                    item_type = item.get("type") #
                    if item_type == "text": #
                        token_count += self.count_text(item.get("text", "")) # #
                    elif item_type == "image_url": #
                        # count_image will return 0 if not self.supports_multimodal
                        token_count += self.count_image(item) # Pass the whole item dict # #
                    else: #
                        logger.warning(f"Unknown content item type: {item_type}") # #
                elif isinstance(item, str): # Handle plain strings mixed in list (should ideally be dicts) #
                    logger.warning(f"Found raw string '{item[:50]}...' in content list, treating as text.") #
                    token_count += self.count_text(item) # #
                else: #
                    logger.warning(f"Unexpected item type in content list: {type(item)}") #

            return token_count #
        else: #
            logger.warning(f"Unexpected content type: {type(content)}. Converting to string.") #
            return self.count_text(str(content)) # Fallback # #


    def count_tool_calls(self, tool_calls: Optional[List[dict]]) -> int: #
        """Calculate tokens for tool calls (list of dicts). Based on OpenAI estimates."""
        if not tool_calls: return 0 #
        token_count = 0 # #
        # Estimate overhead for the list structure itself? Maybe handled by message base tokens.

        for tool_call in tool_calls: #
            # Roughly 4 tokens overhead per function call object {}
            token_count += 4 #
            if isinstance(tool_call, dict): #
                # Add tokens for 'id', 'type', 'function' keys
                token_count += 3 #
                token_count += self.count_text(tool_call.get("id","")) #
                token_count += self.count_text(tool_call.get("type","function")) # Default type #
                function_data = tool_call.get("function") #
                if isinstance(function_data, dict): #
                    # Add tokens for 'name', 'arguments' keys
                    token_count += 2 #
                    token_count += self.count_text(function_data.get("name", "")) #
                    # Arguments should be a string, count that string
                    token_count += self.count_text(function_data.get("arguments", "")) # #
                else: #
                    token_count += 5 # Estimate if function missing/malformed #
            else: # #
                token_count += 10 # Estimate for malformed tool call item # #
        return token_count #

    def count_message_tokens(self, messages: List[dict]) -> int: #
        """
        Calculate the total number of tokens in a message list.
        Uses a manual counting approach based on OpenAI's cookbook estimates.
        """
        # Alternative: Use HuggingFace tokenizer's apply_chat_template if available and accurate for the model.
        # try:
        #     # Ensure add_generation_prompt=False to count only the input messages
        #     encoded = self.tokenizer.apply_chat_template(messages, add_generation_prompt=False, return_dict=True, return_tensors="pt")
        #     count = encoded['input_ids'].shape[1]
        #     logger.debug(f"Token count using apply_chat_template: {count}")
        #     return count
        # except Exception as e:
        #     logger.warning(f"Failed to use apply_chat_template for token counting ({e}), falling back to manual method.")

        # Manual counting:
        total_tokens = 0 #
        for message in messages: #
            # Base overhead per message (e.g., role marker, newlines)
            tokens_per_message = self.BASE_MESSAGE_TOKENS #
            role = message.get("role", "") #
            tokens_per_message += self.count_text(role) #

            # --- Content (Text/Image) ---
            tokens_per_message += self.count_content(message.get("content")) #

            # --- Tool Calls (Assistant Message) ---
            if role == "assistant": #
                tokens_per_message += self.count_tool_calls(message.get("tool_calls")) #

            # --- Tool Response (Tool Message) ---
            if role == "tool": #
                 tokens_per_message += self.count_text(message.get("tool_call_id", "")) #
                 # 'name' is sometimes used with tool role, count if present.
                 # It's often implicitly part of the context rather than explicit field now.
                 # if message.get("name"): tokens_per_message += self.count_text(message["name"]) + 1 # Add 1 for key

            total_tokens += tokens_per_message #

        # Add final format tokens (e.g., for overall prompt structure like <|im_start|>assistant)
        total_tokens += self.FORMAT_TOKENS #
        logger.debug(f"Manual token count: {total_tokens}") #
        return total_tokens #


# --- Modified LLM Class ---
class LLM: #
    _instances: Dict[str, "LLM"] = {} #

    def __new__( #
        cls, config_name: str = "default", llm_config: Optional[LLMSettings] = None
    ): #
        if config_name not in cls._instances: #
            instance = super().__new__(cls) #
            instance._initialized = False #
            cls._instances[config_name] = instance #
        return cls._instances[config_name] # #

    def __init__( #
        self, config_name: str = "default", llm_config: Optional[LLMSettings] = None
    ): #
        if hasattr(self, "_initialized") and self._initialized: #
            return #

        # Resolve configuration
        llm_config_dict = llm_config or config.llm #
        default_llm_settings = llm_config_dict.get("default") #
        specific_llm_settings = llm_config_dict.get(config_name) #

        if not default_llm_settings and not specific_llm_settings: #
            raise ValueError(f"LLM configuration not found for '{config_name}' or 'default'") #

        # Start with default, override with specific if it exists
        llm_settings: LLMSettings #
        base_settings_dict = default_llm_settings.model_dump() if default_llm_settings else {} #
        specific_settings_dict = specific_llm_settings.model_dump(exclude_unset=True) if specific_llm_settings else {} #
        final_settings_dict = {**base_settings_dict, **specific_settings_dict} #

        try: #
            llm_settings = LLMSettings(**final_settings_dict) #
        except Exception as e: #
            logger.error(f"Failed to parse LLM settings for '{config_name}'. Error: {e}. Raw settings: {final_settings_dict}", exc_info=True) #
            raise ValueError(f"Could not load LLM configuration for '{config_name}': {e}") from e #

        self.model = llm_settings.model #
        self.max_tokens = llm_settings.max_tokens # Max *completion* tokens #
        self.temperature = llm_settings.temperature # #
        self.base_url = str(llm_settings.base_url).rstrip('/') if llm_settings.base_url else None # Handle None base_url #
        if self.base_url: #
            logger.info(f"Raw base_url from settings: {llm_settings.base_url}, Processed base_url: {self.base_url}") #
        else: #
            logger.warning("No base_url configured for LLM.") #

        self.api_key = llm_settings.api_key #
        self.request_timeout = 600 #

        # Read the flag indicating if this LLM config expects pythonic tool calls
        # This is now primarily informational, as ask_tool logic is standardized.
        self.use_pythonic_tool_parser = getattr(llm_settings, "use_pythonic_tool_parser", False) # Default to False (standard) #
        logger.info(f"LLM configuration indicates pythonic tool calls expected: {self.use_pythonic_tool_parser}") #

        # Multimodal support flag
        self.supports_multimodal = getattr(llm_settings, "supports_multimodal", False) # Default False if not specified #
        logger.info(f"Multimodal support enabled: {self.supports_multimodal}") # #

        # Token counting attributes
        self.total_input_tokens = 0 #
        self.total_completion_tokens = 0 #
        # Max *input* tokens (context window size)
        self.max_input_tokens = getattr(llm_settings, "max_input_tokens", None) #
        logger.info(f"Max Input Tokens (Context Window): {self.max_input_tokens}") #

        # Initialize HuggingFace Tokenizer
        # Use specific tokenizer if provided, else fallback to model name
        tokenizer_id = getattr(llm_settings, "tokenizer_name_or_path", None) or self.model #
        logger.info(f"Attempting to load tokenizer: '{tokenizer_id}'") #
        try: #
            trust_remote_code = getattr(llm_settings, "trust_remote_code", True) # Default True for flexibility #
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, trust_remote_code=trust_remote_code) # #
            if self.tokenizer.pad_token is None: #
                if self.tokenizer.eos_token: #
                     self.tokenizer.pad_token = self.tokenizer.eos_token #
                     logger.warning(f"Tokenizer '{tokenizer_id}' missing pad_token, setting to eos_token ('{self.tokenizer.eos_token}').") #
                else: #
                     # Add a default pad token if EOS is also missing (less common)
                     self.tokenizer.add_special_tokens({'pad_token': '[PAD]'}) #
                     logger.warning(f"Tokenizer '{tokenizer_id}' missing both pad_token and eos_token. Added '[PAD]' as pad_token.") #

            # Check for chat template existence (useful for apply_chat_template)
            if not getattr(self.tokenizer, 'chat_template', None) and not getattr(self.tokenizer, 'default_chat_template', None): #
                logger.warning(f"Tokenizer '{tokenizer_id}' may not have a default chat template defined. Manual token counting will be used.") #

        except Exception as e: #
            logger.error(f"Failed to load tokenizer '{tokenizer_id}'. Error: {e}", exc_info=True) #
            raise ValueError(f"Could not load tokenizer: {tokenizer_id}") from e # #

        # Initialize TokenCounter with tokenizer AND multimodal support flag
        self.token_counter = TokenCounter(self.tokenizer, self.supports_multimodal) #

        # Initialize HTTPX Async Client
        headers = {"Accept": "application/json"} #
        # vLLM typically uses a dummy key, but include if provided
        if self.api_key and self.api_key != "EMPTY": #
            headers["Authorization"] = f"Bearer {self.api_key}" #

        timeouts = httpx.Timeout(self.request_timeout, connect=30.0) #
        # Set retries=0 in transport, handle retries via tenacity decorator
        transport = httpx.AsyncHTTPTransport(retries=0) #

        # Only initialize client if base_url is set
        if self.base_url: #
            self.client = httpx.AsyncClient( #
                base_url=self.base_url, #
                headers=headers, #
                timeout=timeouts, #
                follow_redirects=True, #
                transport=transport # #
            ) #
            logger.info(f"LLM instance '{config_name}' initialized for model '{self.model}' at '{self.base_url}'") # #
        else: #
            logger.error(f"Cannot initialize HTTP client for LLM instance '{config_name}'. No base_url configured.") #
            self.client = None # Set client to None if no base_url #

        self._initialized = True #

    async def close(self): #
        """Close the httpx client."""
        if hasattr(self, 'client') and isinstance(self.client, httpx.AsyncClient): #
            await self.client.aclose() #
            logger.info(f"Closed httpx client for LLM instance connected to {self.base_url}") #

    # --- Token Counting Methods ---
    def count_tokens(self, text: str) -> int: # #
        """Calculate the number of tokens in a text using the instance's tokenizer."""
        return self.token_counter.count_text(text) # #

    def count_message_tokens(self, messages: List[dict]) -> int: #
        """Calculate the total number of tokens in a message list."""
        return self.token_counter.count_message_tokens(messages) #

    def update_token_count(self, input_tokens: int, completion_tokens: int = 0) -> None: # #
        """Update token counts."""
        self.total_input_tokens += input_tokens #
        self.total_completion_tokens += completion_tokens #
        # Use debug level for frequent token updates
        logger.debug( #
            f"Token usage update: Input={input_tokens}, Completion={completion_tokens}, " #
            f"Cumulative Input={self.total_input_tokens}, Cumulative Completion={self.total_completion_tokens}, " #
            f"Cumulative Total={self.total_input_tokens + self.total_completion_tokens}" #
        ) # #

    def check_token_limit(self, input_tokens: int) -> bool: #
        """Check if adding input_tokens exceeds the max_input_tokens limit."""
        if self.max_input_tokens is not None and self.max_input_tokens > 0: #
            # Check against configured max_input_tokens
            return (self.total_input_tokens + input_tokens) <= self.max_input_tokens #
        # If no limit set or limit is zero/negative, assume no limit check needed
        return True # No limit set or invalid limit #

    def get_limit_error_message(self, input_tokens: int) -> str: #
        """Generate error message for token limit exceeded"""
        if self.max_input_tokens is not None: # #
            return (f"Request may exceed input token limit (Current Cumulative: {self.total_input_tokens}, " #
                    f"Request Tokens: {input_tokens}, Max Limit: {self.max_input_tokens})") #
        return "Token limit exceeded (Max limit not configured)" # #


    # --- Message Formatting ---
    def format_messages( #
        self,
        messages: List[Union[dict, Message]]
    ) -> List[dict]: #
        """
        Format messages into the standard OpenAI dictionary list format.
        Handles Message objects and basic validation.
        Uses self.supports_multimodal to determine image handling.
        Adds base64 images to the *last user message* if supported.
        """
        formatted_messages = [] #
        supports_images = self.supports_multimodal #
        # --- FIX: Track image inclusion across the entire message list ---
        # Base64 images should ideally only be in the last user message per OpenAI spec
        # This flag isn't strictly needed here if we enforce adding only to the last user message later.
        # --- End FIX ---

        processed_messages = [] #
        for idx, message in enumerate(messages): #
            if isinstance(message, Message): #
                msg_dict = message.to_dict() #
            elif isinstance(message, dict): #
                msg_dict = message.copy() # Avoid modifying original #
            else: #
                logger.error(f"Unsupported message type encountered at index {idx}: {type(message)}") #
                raise TypeError(f"Unsupported message type: {type(message)}") # #

            if "role" not in msg_dict or msg_dict["role"] not in ROLE_VALUES: #
                logger.error(f"Invalid or missing role in message at index {idx}: {msg_dict}") #
                raise ValueError(f"Invalid or missing role in message: {msg_dict}") #

            processed_messages.append(msg_dict) #

        # --- Image Handling: Add images ONLY to the last user message ---
        last_user_msg_index = -1 #
        for i in range(len(processed_messages) - 1, -1, -1): #
            if processed_messages[i]['role'] == 'user': #
                last_user_msg_index = i #
                break #

        images_to_add = [] #
        # Extract base64_image from *all* messages for potential inclusion
        for idx, msg_dict in enumerate(processed_messages): #
            base64_image = msg_dict.pop("base64_image", None) #
            if base64_image: #
                if supports_images: #
                    # Store image data along with its original message index
                    images_to_add.append({'data': base64_image, 'original_index': idx}) #
                    logger.debug(f"Found base64 image in message {idx} to potentially add.") #
                else: #
                     logger.warning(f"Ignoring base64_image found in message {idx} as multimodal support is disabled.") #

        # Now, add collected images to the last user message if found
        if last_user_msg_index != -1 and images_to_add: #
            logger.info(f"Adding {len(images_to_add)} images to the last user message (index {last_user_msg_index}).") #
            last_user_message = processed_messages[last_user_msg_index] #
            content = last_user_message.get("content") #

            # Ensure content is a list for multimodal input
            if content is None: content_list = [] #
            elif isinstance(content, str): content_list = [{"type": "text", "text": content}] # #
            elif isinstance(content, list): #
                # Ensure all items in existing list are dicts
                content_list = [] #
                for item in content: #
                    if isinstance(item, str): content_list.append({"type": "text", "text": item}) #
                    elif isinstance(item, dict): content_list.append(item) #
                    else: logger.warning(f"Skipping invalid item in existing content list: {type(item)}") #
            else: #
                logger.error(f"Invalid existing content type in last user message: {type(content)}. Cannot add images.") #
                # Decide: raise error or just skip adding images? Let's skip.
                content_list = [{"type": "text", "text": str(content)}] # Fallback #

            # Add image parts
            for img_info in images_to_add: #
                image_url_str = f"data:image/jpeg;base64,{img_info['data']}" # Assume JPEG for now #
                image_part = {"type": "image_url", "image_url": {"url": image_url_str}} # #
                # Add dimensions if available (requires parsing image or getting from source)
                # image_part["image_url"]["dimensions"] = [width, height]
                content_list.append(image_part) #
                logger.debug(f"Added image from original message {img_info['original_index']} to last user message.") #

            last_user_message["content"] = content_list # Update the message content #

        # --- Final Message Creation ---
        for msg_dict in processed_messages: #
            # Construct the final message dict according to OpenAI schema
            final_msg = { # #
                "role": msg_dict["role"], #
                # Ensure content is not empty string if other fields exist, but allow null/None
                "content": msg_dict.get("content") if msg_dict.get("content") is not None else None, #
                # Include tool calls/ids only if they exist and are not None/empty
                **({"tool_calls": msg_dict["tool_calls"]} if msg_dict.get("tool_calls") else {}), #
                **({"tool_call_id": msg_dict["tool_call_id"]} if msg_dict.get("tool_call_id") else {}), #
                # 'name' field is deprecated for 'tool' role, use tool_call_id instead.
                # **({"name": msg_dict["name"]} if msg_dict.get("name") and msg_dict["role"] == "tool" else {}),
            } # #

            # Filter out keys with None values before adding
            final_msg_filtered = {k: v for k, v in final_msg.items() if v is not None} #

            # Add the message if it has content OR tool info OR is a system message (which can be empty)
            if final_msg_filtered.get("content") is not None or \
               final_msg_filtered.get("tool_calls") or \
               final_msg_filtered.get("tool_call_id") or \
               final_msg_filtered["role"] == "system": #
                formatted_messages.append(final_msg_filtered) #
            else: #
                logger.debug(f"Skipping message with no content/tool info and not system role: {final_msg_filtered}") # #

        return formatted_messages #


    # --- Core API Interaction Methods ---

    def _prepare_payload( #
        self,
        messages: List[dict],
        stream: bool,
        temperature: Optional[float],
        max_tokens: Optional[int], # Max *completion* tokens #
        tools: Optional[List[dict]] = None, # Standard tool schema #
        tool_choice: Optional[TOOL_CHOICE_TYPE] = None, # Standard tool choice #
        **kwargs # Allow additional kwargs (e.g., top_p, frequency_penalty) #
    ) -> Dict[str, Any]: #
        """Helper to create the JSON payload for the API request.""" # #
        payload = { #
            "model": self.model, #
            "messages": messages, #
            # Use instance default max_tokens if not overridden
            "max_tokens": max_tokens if max_tokens is not None else self.max_tokens, #
            # Use instance default temperature if not overridden
            "temperature": temperature if temperature is not None else self.temperature, #
            "stream": stream, # #
            # Add other common parameters if provided in kwargs
            **{k: v for k, v in kwargs.items() if k in ['top_p', 'presence_penalty', 'frequency_penalty', 'stop'] and v is not None} #
        } #

        # Add standard tools/tool_choice if they are provided
        # This logic now runs unconditionally if tools/tool_choice are passed to ask_tool
        if tools: #
            payload["tools"] = tools #
            # Default tool_choice to 'auto' if tools are provided but no choice specified
            # OpenAI/vLLM default is typically 'auto' when tools are present.
            # Explicitly setting it might be needed for some models or older vLLM versions.
            effective_tool_choice = tool_choice if tool_choice is not None else ToolChoice.AUTO # Use constant #
            # Only include 'tool_choice' if it's NOT 'auto' (to rely on server default)
            # OR if it's explicitly 'none' or 'required' or a specific function dict.
            if effective_tool_choice != ToolChoice.AUTO: #
                 payload["tool_choice"] = effective_tool_choice #
            elif effective_tool_choice in [ToolChoice.NONE, ToolChoice.REQUIRED]: # Explicit none/required #
                 payload["tool_choice"] = effective_tool_choice #
            elif isinstance(effective_tool_choice, dict): # Specific function choice #
                 payload["tool_choice"] = effective_tool_choice #
            # If 'auto', we omit it to rely on the server's default behavior when tools are present.

        # Filter out any top-level keys with None values before returning
        payload = {k: v for k, v in payload.items() if v is not None} #
        return payload #

    async def _process_response( #
        self, # #
        response: httpx.Response,
        input_tokens: int # Calculated input tokens passed for comparison/update #
    ) -> Dict[str, Any]: # Return the full message dict from the response #
        """Process non-streaming HTTP response, return assistant message dict."""
        response.raise_for_status() # Raise HTTPStatusError for 4xx/5xx #
        try: #
            data = response.json() #
            logger.debug(f"[_process_response] Parsed JSON data:\n{json.dumps(data, indent=2)}") #

        except json.JSONDecodeError as e: #
            logger.error(f"Failed to decode JSON response from vLLM: {e}. Response text: {response.text[:500]}") # #
            raise ValueError(f"Invalid JSON response received from LLM API: {e}") from e #

        # Validate structure - expecting OpenAI compatible format
        if not data.get("choices") or not isinstance(data["choices"], list) or len(data["choices"]) == 0: #
             logger.error(f"Invalid response structure: 'choices' array missing or empty. Response: {data}") #
             raise ValueError("Received invalid response structure from LLM API (missing/empty 'choices')") #

        choice = data["choices"][0] #
        if not isinstance(choice, dict): #
            logger.error(f"Invalid response structure: 'choices' item is not a dictionary. Response: {data}") #
            raise ValueError("Received invalid response structure from LLM API (invalid 'choices' item)") #

        message = choice.get("message") #
        if not isinstance(message, dict): #
            logger.error(f"Invalid response structure: 'message' object missing or not a dictionary in choice. Response: {data}") #
            raise ValueError("Received invalid response structure from LLM API (missing/invalid 'message')") #

        # --- Token Usage ---
        usage = data.get("usage") #
        completion_tokens = 0 #
        prompt_tokens_api = input_tokens # Default to calculated if API doesn't provide # #

        if usage and isinstance(usage, dict): #
            completion_tokens = usage.get("completion_tokens", 0) #
            prompt_tokens_api_val = usage.get("prompt_tokens") #
            if prompt_tokens_api_val is not None: #
                # Compare API prompt tokens with calculated tokens
                if abs(prompt_tokens_api_val - input_tokens) > 20: # Tolerance for slight variations #
                    logger.warning(f"API prompt_tokens ({prompt_tokens_api_val}) significantly differs " # #
                                   f"from calculated ({input_tokens}). Using API value for update.") # #
                prompt_tokens_api = prompt_tokens_api_val # Trust API if available and different #
            else: #
                logger.warning("Usage data found, but 'prompt_tokens' missing. Using calculated input tokens for update.") # #
        else: #
            logger.warning("Token usage information ('usage' field) not found in response. Estimating completion tokens.") #
            # Estimate completion tokens based on the returned message content/tool calls
            # Note: This requires the TokenCounter instance (self.token_counter)
            est_content = message.get("content", "") or "" #
            est_tool_calls = message.get("tool_calls") # Can be None or list #
            completion_tokens = self.token_counter.count_text(est_content) + self.token_counter.count_tool_calls(est_tool_calls) #
            logger.warning(f"Estimated completion tokens: {completion_tokens}") #


        # Update token counts using the potentially API-provided prompt tokens
        self.update_token_count(prompt_tokens_api, completion_tokens) # #
        # --- End Token Usage ---

        # Add finish reason to the message dict for convenience
        finish_reason = choice.get("finish_reason") #
        if finish_reason: #
             message["finish_reason"] = finish_reason #

        # Return the assistant message dictionary directly
        return message #


    async def _process_streaming_response( #
        self,
        response: httpx.Response,
        input_tokens: int # Calculated input tokens (already accounted for pre-stream) #
    ) -> AsyncGenerator[Dict[str, Any], None]: # Yield chunk delta dictionaries #
        """Process streaming HTTP response (SSE), yielding delta dictionaries."""
        # Initialize tracking variables
        completion_tokens_api = 0 #
        prompt_tokens_api = input_tokens # Start with calculated input #
        processed_chunk_count = 0 #
        accumulated_content = "" # For estimating completion tokens if needed #
        accumulated_tool_call_chunks = [] # For estimating tool call tokens #

        try: #
            async for line in response.aiter_lines(): #
                if line.startswith("data:"): #
                    data_str = line[len("data:") :].strip() #
                    if data_str == "[DONE]": #
                        break # End of stream marker #
                    try: #
                        chunk = json.loads(data_str) #
                        processed_chunk_count += 1 #

                        if not chunk.get("choices") or not isinstance(chunk["choices"], list) or len(chunk["choices"]) == 0: #
                            logger.warning(f"Stream chunk missing 'choices' array or empty: {chunk}") #
                            continue #
                        choice = chunk["choices"][0] #
                        if not isinstance(choice, dict): #
                             logger.warning(f"Stream chunk 'choices' item is not a dict: {choice}") #
                             continue #

                        delta = choice.get("delta") #
                        if not isinstance(delta, dict): # Should be a dictionary #
                            logger.warning(f"Stream chunk missing 'delta' object or not a dict: {chunk}") #
                            continue #

                        # --- Yield the delta --- #
                        yield delta #
                        # --- Accumulate for estimation ---
                        if delta.get("content"): #
                            accumulated_content += delta["content"] #
                        if delta.get("tool_calls"): #
                             # Note: Tool calls in streams often come in chunks per tool
                             # Need more sophisticated logic to reconstruct full calls for accurate token counting
                             accumulated_tool_call_chunks.extend(delta.get("tool_calls", [])) #

                        # Check for usage data (might be in the final chunk)
                        usage = chunk.get("usage") # vLLM might include usage in the last chunk #
                        if usage and isinstance(usage, dict): #
                            completion_tokens_api = usage.get("completion_tokens", completion_tokens_api) #
                            prompt_tokens_api_val = usage.get("prompt_tokens") #
                            if prompt_tokens_api_val is not None: #
                                prompt_tokens_api = prompt_tokens_api_val # Update if provided #

                    except json.JSONDecodeError: #
                        logger.error(f"Failed to decode stream chunk JSON: {data_str}") # #
                    except Exception as e: #
                        logger.error(f"Error processing stream chunk: {e} - Chunk Data: {data_str}", exc_info=True) # #

            logger.debug(f"Processed {processed_chunk_count} stream data chunks.") #

        finally: #
            # --- Update tokens post-stream ---
            completion_tokens_final = 0 #
            # Prefer API provided completion tokens if available
            if completion_tokens_api > 0: #
                completion_tokens_final = completion_tokens_api #
                logger.info(f"Received completion_tokens ({completion_tokens_api}) via stream.") #
                # Compare prompt tokens if API provided them
                if prompt_tokens_api != input_tokens: #
                     if abs(prompt_tokens_api - input_tokens) > 20: #
                         logger.warning(f"API prompt_tokens ({prompt_tokens_api}) differs from calculated ({input_tokens}) in stream.") #
                     # Note: Input tokens were already added pre-stream based on calculation.
                     # We only update the *completion* tokens here.
            else: #
                # Estimate completion tokens based on accumulated content/tools if API didn't provide
                logger.warning("Completion token usage not found in stream. Estimating based on accumulated deltas.") #
                # This estimation is rough, especially for tool calls
                est_content_tokens = self.token_counter.count_text(accumulated_content) #
                # Crude estimation for tool call chunks - needs improvement
                est_tool_tokens = self.token_counter.count_text(json.dumps(accumulated_tool_call_chunks)) #
                completion_tokens_final = est_content_tokens + est_tool_tokens #
                logger.warning(f"Estimated stream completion tokens: {completion_tokens_final} (Content: {est_content_tokens}, Tools: {est_tool_tokens})") #

            # Update *only* completion tokens post-stream (input updated pre-stream)
            # Avoid double-counting input tokens.
            self.total_completion_tokens += completion_tokens_final #
            logger.info( #
                f"Stream ended. Final Token Count Update: Cumulative Input={self.total_input_tokens}, Cumulative Completion={self.total_completion_tokens}" #
            ) # #
            # --- End Token Update ---


    @retry( #
        wait=wait_random_exponential(min=1, max=30, multiplier=1.5), # Exponential backoff #
        stop=stop_after_attempt(5), # Max 5 attempts #
        # Retry only on specific network/server errors (5xx) or timeouts
        retry=( #
            retry_if_exception_type( #
                (httpx.RequestError, httpx.TimeoutException, httpx.HTTPStatusError) #
            ) #
            | #
            retry_if_exception( #
                lambda e: isinstance(e, httpx.HTTPStatusError) and e.response.status_code >= 500 #
            ) #
        ), #
        retry_error_callback=lambda retry_state: logger.warning(f"Retrying LLM request after error: {retry_state.outcome.exception()} (Attempt {retry_state.attempt_number})") # #
    ) #
    async def ask( #
        self,
        messages: List[Union[dict, Message]], # #
        system_msgs: Optional[List[Union[dict, Message]]] = None, #
        stream: bool = False, #
        temperature: Optional[float] = None, #
        max_tokens_override: Optional[int] = None, # Max *completion* tokens #
        **kwargs # Pass extra args to payload (e.g., top_p, stop sequences, request_id) #
    ) -> Union[str, AsyncGenerator[Dict[str, Any], None]]: # Return text content string OR stream generator of delta dicts #
        """
        Send a prompt to the vLLM API. Returns text content string or async generator of delta dicts. # #
        Handles standard text generation without tools.
        """ # #
        request_id = kwargs.pop("request_id", None) # Allow passing request ID for logging #
        log_prefix = f"[Req-{request_id}] " if request_id else "" #

        # Check for client initialization
        if not self.client: #
            logger.error(f"{log_prefix}HTTP client not initialized (likely missing base_url). Cannot make request.") #
            raise RuntimeError("LLM client not initialized. Check configuration (base_url).") #

        try: #
            # --- Message Formatting & Token Calculation ---
            # Format messages using the instance method (handles multimodal if applicable)
            if system_msgs: # #
                formatted_system = self.format_messages(system_msgs) #
                formatted_user = self.format_messages(messages) #
                final_messages = formatted_system + formatted_user #
            else: #
                final_messages = self.format_messages(messages) # #

            if not final_messages: #
                logger.error(f"{log_prefix}No valid messages to send after formatting.") #
                raise ValueError("Cannot send empty message list to LLM.") # #

            logger.debug(f"{log_prefix}Formatted messages count: {len(final_messages)}") #

            input_tokens = self.count_message_tokens(final_messages) #
            logger.info(f"{log_prefix}Calculated input tokens for 'ask': {input_tokens}") #

            # Check token limit against max_input_tokens
            if not self.check_token_limit(input_tokens): # #
                error_message = self.get_limit_error_message(input_tokens) #
                logger.error(f"{log_prefix}{error_message}") #
                raise TokenLimitExceeded(error_message) #

            # Update input tokens *before* the request (consistent for stream/non-stream)
            # Completion tokens updated in response processing.
            self.update_token_count(input_tokens=input_tokens, completion_tokens=0) # #

            # --- Prepare Payload ---
            payload = self._prepare_payload( #
                messages=final_messages, #
                stream=stream, #
                temperature=temperature, #
                max_tokens=max_tokens_override, # Max completion tokens #
                # Do NOT pass tools/tool_choice for standard 'ask'
                **kwargs # Pass other args like top_p, stop, etc. #
            ) #
            logger.debug(f"{log_prefix}Sending payload to vLLM endpoint '{VLLM_API_SUFFIX}' for 'ask': {json.dumps(payload, indent=2)}") #

            endpoint = VLLM_API_SUFFIX #
            timeout = self.request_timeout # Use instance default #

            # --- Make API Call & Process Response ---
            if not stream: #
                response = await self.client.post(endpoint, json=payload, timeout=timeout) # #
                # Process response returns the full assistant message dict
                assistant_message_dict = await self._process_response(response, input_tokens) # input_tokens needed for token update logic #
                # Extract just the content for the 'ask' method's string return type
                content = assistant_message_dict.get("content", "") #
                logger.debug(f"{log_prefix}Received non-streamed response content snippet: {str(content)[:100]}...") # Use str() for safety #
                return content or "" # Return empty string if content is None or empty #
            else: #
                # Return the async generator directly
                logger.debug(f"{log_prefix}Initiating stream request for 'ask'...") # #
                async def stream_generator(): #
                    try: #
                        async with self.client.stream("POST", endpoint, json=payload, timeout=timeout) as response: # #
                            # Raise status error early if stream fails to start
                            response.raise_for_status() #
                            # Process the stream, yielding deltas
                            async for delta in self._process_streaming_response(response, input_tokens): #
                                yield delta # Yield the delta dictionary #
                    except httpx.HTTPStatusError as e: #
                        # Attempt to read body for detailed error message from server
                        error_body = "<failed to read error response body>" # #
                        try: #
                            error_body = await e.response.aread() # #
                            error_body = error_body.decode() #
                        except Exception: pass #
                        logger.error(f"{log_prefix}HTTP error starting/during stream: {e}. Response: {error_body}") # #
                        # Propagate error through generator by raising
                        raise ValueError(f"vLLM stream request failed ({e.response.status_code}): {error_body}") from e # #
                    except Exception as e: #
                        logger.error(f"{log_prefix}Unexpected error during streaming: {e}", exc_info=True) # #
                        raise # Re-raise other exceptions #

                return stream_generator() # Return the async generator #

        # --- Exception Handling ---
        except TokenLimitExceeded: # Catch specific error #
            logger.error(f"{log_prefix}TokenLimitExceeded in 'ask'") #
            raise # Re-raise for calling code #
        except httpx.HTTPStatusError as e: # Catch HTTP errors (4xx/5xx) #
            status_code = e.response.status_code if hasattr(e, 'response') else 'N/A' #
            error_body = "<failed to read error response body>" #
            try: #
                if hasattr(e, 'response'): # #
                    error_body = await e.response.aread() #
                    error_body = error_body.decode() #
            except Exception: pass #
            logger.error(f"{log_prefix}HTTP Error {status_code} from vLLM API: {error_body}", exc_info=True) # #
            # Check if it's a client error (4xx) - likely bad request, don't retry via tenacity usually
            if isinstance(status_code, int) and 400 <= status_code < 500: #
                 # If it's specifically 429 (Too Many Requests), tenacity might handle it if configured, otherwise raise specific error
                 # if status_code == 429: raise RateLimitError(...)
                 raise ValueError(f"vLLM API request failed (Client Error {status_code}): {error_body}") from e #
            else: # Includes connection errors where response might not exist, or 5xx errors #
                 raise # Re-raise for tenacity or caller # #
        except (httpx.RequestError, httpx.TimeoutException) as e: # Catch network/timeout errors #
            logger.exception(f"{log_prefix}Network/Connection error communicating with vLLM API: {e}") #
            raise # Re-raise for tenacity or caller #
        except ValueError as ve: # Catch our validation errors (e.g., empty messages, JSON decode) #
            logger.exception(f"{log_prefix}Data validation or processing error: {ve}") #
            raise # Re-raise #
        except Exception as e: # Catch unexpected errors #
            logger.exception(f"{log_prefix}Unexpected error in LLM ask method: {e}") #
            raise # Re-raise #


    @retry( #
        wait=wait_random_exponential(min=1, max=30, multiplier=1.5), #
        stop=stop_after_attempt(5), #
        # Retry only on specific network/server errors (5xx) or timeouts
        # retry=retry_if_exception_type((httpx.RequestError, httpx.TimeoutException, httpx.HTTPStatusError)) | (lambda e: isinstance(e, httpx.HTTPStatusError) and e.response.status_code >= 500),
        retry=( #
            retry_if_exception_type( #
                (httpx.RequestError, httpx.TimeoutException, httpx.HTTPStatusError) #
            ) #
            | #
            retry_if_exception( #
                lambda e: isinstance(e, httpx.HTTPStatusError) and e.response.status_code >= 500 #
            ) #
        ), #
        retry_error_callback=lambda retry_state: logger.warning(f"Retrying LLM tool request after error: {retry_state.outcome.exception()} (Attempt {retry_state.attempt_number})") # #
    ) #
    async def ask_tool( #
        self,
        messages: List[Union[dict, Message]],
        system_msgs: Optional[List[Union[dict, Message]]] = None,
        # timeout: Optional[int] = None, # Timeout now handled by httpx client default/retry
        tools: Optional[List[dict]] = None, # Standard OpenAI tools schema #
        tool_choice: Optional[TOOL_CHOICE_TYPE] = None, # Standard OpenAI tool choice #
        temperature: Optional[float] = None, #
        max_tokens_override: Optional[int] = None, # Max completion tokens #
        # *** Flag from signature (passed from think method) - now informational ***
        use_pythonic_parser: bool = False, # Default False, read from instance if needed #
        **kwargs, #
    ) -> Optional[Dict[str, Any]]: # Return the raw assistant message dictionary (content + tool_calls) # #
        """
        Ask LLM to decide between responding directly or using tools (non-streaming). # #
        Uses the standard OpenAI/vLLM tool calling mechanism by sending 'tools' and 'tool_choice'.
        Returns the assistant's message dictionary, including 'content' and/or 'tool_calls'.
        """ # #
        request_id = kwargs.pop("request_id", None) #
        log_prefix = f"[Req-{request_id}] " if request_id else "" #

        # Check for client initialization
        if not self.client: #
            logger.error(f"{log_prefix}HTTP client not initialized (likely missing base_url). Cannot make tool request.") #
            raise RuntimeError("LLM client not initialized. Check configuration (base_url).") #

        # Log the mode based on the instance config (informational)
        logger.debug(f"{log_prefix}ask_tool called. Instance configured for pythonic parser: {self.use_pythonic_tool_parser}. Sending standard tool payload.") #

        # Validation: Tools are required for tool calling to function meaningfully.
        if not tools: #
             logger.warning(f"{log_prefix}ask_tool called without providing 'tools'. LLM cannot use tools.") #
             # If tool_choice is 'required', this is an error.
             if tool_choice == ToolChoice.REQUIRED: #
                  raise ValueError("ask_tool requires 'tools' when tool_choice is 'required'.") #
             # Otherwise, proceed, but LLM will likely just generate text.

        try: #
            # Validate tool_choice format if provided
            if tool_choice: # #
                if not isinstance(tool_choice, (str, dict)): #
                     raise ValueError(f"Invalid tool_choice type: {type(tool_choice)}") #
                if isinstance(tool_choice, str) and tool_choice not in TOOL_CHOICE_VALUES: #
                     raise ValueError(f"Invalid tool_choice string value: {tool_choice}") #
                if isinstance(tool_choice, dict) and (tool_choice.get("type") != "function" or not isinstance(tool_choice.get("function"), dict) or not tool_choice["function"].get("name")): #
                     raise ValueError(f"Invalid tool_choice dictionary structure: {tool_choice}") #


            # --- Message Formatting & Token Calculation ---
            # Use instance multimodal flag for formatting
            # Format messages using instance method
            if system_msgs: # #
                formatted_system = self.format_messages(system_msgs) #
                formatted_user = self.format_messages(messages) #
                final_messages = formatted_system + formatted_user #
            else: #
                final_messages = self.format_messages(messages) # #

            if not final_messages: #
                logger.error(f"{log_prefix}No valid messages for tool request after formatting.") #
                raise ValueError("Cannot send empty message list for tool request.") # #

            logger.debug(f"{log_prefix}Formatted messages count for tool request: {len(final_messages)}") #

            # Calculate Tokens (Include tool definitions)
            message_tokens = self.count_message_tokens(final_messages) # #
            tools_tokens = 0 #
            if tools: # Only count if sending standard tools #
                try: #
                    # Estimate based on JSON representation
                    tools_json_str = json.dumps(tools) #
                    tools_tokens = self.count_tokens(tools_json_str) #
                    # Add a small buffer for structural overhead
                    tools_tokens += len(tools) * 5 # ~5 extra tokens per tool definition #
                except Exception: # #
                    logger.exception(f"{log_prefix}Failed to calculate token size for tool definitions.") #
                    tools_tokens = len(tools) * 50 # Rough fallback estimate # #

            input_tokens = message_tokens + tools_tokens #
            logger.info(f"{log_prefix}Calculated input tokens for tool request: {input_tokens} (Messages: {message_tokens}, Tools: {tools_tokens})") # #

            # Check token limit
            if not self.check_token_limit(input_tokens): #
                error_message = self.get_limit_error_message(input_tokens) #
                logger.error(f"{log_prefix}{error_message}") #
                raise TokenLimitExceeded(error_message) #

            # Update input tokens before the request
            self.update_token_count(input_tokens=input_tokens, completion_tokens=0) #

            # --- Prepare Payload ---
            # Tool requests are non-streaming.
            # Pass the tools and tool_choice directly.
            payload = self._prepare_payload( # #
                messages=final_messages, #
                stream=False, # Tool requests are typically non-streaming #
                temperature=temperature, #
                max_tokens=max_tokens_override, # Max completion tokens #
                tools=tools, #
                tool_choice=tool_choice, # #
                **kwargs # Pass extra args like top_p etc. #
            ) #
            logger.debug(f"{log_prefix}Sending tool request payload: {json.dumps(payload, indent=2)}") # #

            # --- Make the API Call ---
            endpoint = VLLM_API_SUFFIX #
            request_timeout = self.request_timeout # Use instance default timeout #

            response = await self.client.post(endpoint, json=payload, timeout=request_timeout) #

            # --- Log Raw Response ---
            raw_response_text = "<failed to read response text>" #
            try: raw_response_text = response.text #
            except Exception: pass #
            logger.debug(f"{log_prefix}Raw HTTP Response Status: {response.status_code}") #
            logger.debug(f"{log_prefix}Raw HTTP Response Text (Preview): {raw_response_text[:1000]}...") #
            # --- End Log ---

            # --- Process Response ---
            # Process response returns the assistant message dict directly
            assistant_message_dict = await self._process_response(response, input_tokens) #
            logger.debug(f"{log_prefix}Received tool response message dict: {assistant_message_dict}") # #

            # Finish reason should be available in assistant_message_dict from _process_response
            finish_reason = assistant_message_dict.get("finish_reason", "unknown") #
            logger.info(f"{log_prefix}LLM finish reason for tool request: {finish_reason}") #

            # Return the complete assistant message dictionary
            return assistant_message_dict # Contains role, content, tool_calls, finish_reason #


        # --- Exception Handling ---
        except TokenLimitExceeded: # #
            logger.error(f"{log_prefix}TokenLimitExceeded in ask_tool") #
            raise #
        except httpx.HTTPStatusError as e: # #
            status_code = e.response.status_code if hasattr(e, 'response') else 'N/A' #
            error_body = "<failed to read error response body>" #
            try: #
                if hasattr(e, 'response'): # #
                    # FIX: Read body safely and decode with error handling
                    error_body_bytes = await e.response.aread() #
                    error_body = error_body_bytes.decode('utf-8', errors='replace') # Decode safely #
            except Exception as read_err: #
                logger.warning(f"Failed to read/decode error response body: {read_err}") #
                pass #

            # FIX: Use safer logging format (e.g., %s or repr())
            # logger.error(f"{log_prefix}HTTP Error {status_code} from vLLM API (tool request): {error_body}", exc_info=True) <-- Problematic f-string
            logger.error(f"{log_prefix}HTTP Error {status_code} from vLLM API (tool request): %s", repr(error_body), exc_info=True) # Use repr() for safety

            if isinstance(status_code, int) and 400 <= status_code < 500: #
                # If 400 Bad Request, often indicates issues with tool definitions or parameters
                if status_code == 400: #
                    logger.error(f"{log_prefix}Received 400 Bad Request. Check tool definitions and payload structure.") #
                raise ValueError(f"vLLM API tool request failed (Client Error {status_code}): {error_body}") from e #
            else: #
                raise # Re-raise for tenacity or caller #
        except (httpx.RequestError, httpx.TimeoutException) as e: # #
            logger.exception(f"{log_prefix}Network/Connection error during tool request: {e}") #
            raise # Re-raise for tenacity or caller #
        except ValueError as ve: # Catch our validation errors #
            logger.exception(f"{log_prefix}Data validation or processing error in ask_tool: {ve}") #
            raise #
        except Exception as e: # Catch unexpected errors #
            logger.exception(f"{log_prefix}Unexpected error in ask_tool: {e}") #
            raise # #


    # --- ask_with_images method remains largely the same ---
    # It uses the standard 'ask' method internally after formatting messages.
    @retry( #
        wait=wait_random_exponential(min=1, max=30, multiplier=1.5), #
        stop=stop_after_attempt(5), #
        # Retry only on specific network/server errors (5xx) or timeouts
        # retry=retry_if_exception_type((httpx.RequestError, httpx.TimeoutException, httpx.HTTPStatusError)) | (lambda e: isinstance(e, httpx.HTTPStatusError) and e.response.status_code >= 500),
        retry=( #
            retry_if_exception_type( #
                (httpx.RequestError, httpx.TimeoutException, httpx.HTTPStatusError) #
            ) #
            | #
            retry_if_exception( #
                lambda e: isinstance(e, httpx.HTTPStatusError) and e.response.status_code >= 500 #
            ) #
        ), #
        retry_error_callback=lambda retry_state: logger.warning(f"Retrying LLM image request after error: {retry_state.outcome.exception()} (Attempt {retry_state.attempt_number})") #
    ) #
    async def ask_with_images( #
        self,
        messages: List[Union[dict, Message]],
        images: List[Union[str, dict]], # List of base64 strings or image dicts {'url': 'data:...'} #
        system_msgs: Optional[List[Union[dict, Message]]] = None, #
        stream: bool = False, #
        temperature: Optional[float] = None, #
        max_tokens_override: Optional[int] = None, #
        **kwargs #
    ) -> Union[str, AsyncGenerator[Dict[str, Any], None]]: # Return string OR stream generator #
        """
        Send a prompt with images to the LLM. # #
        Requires multimodal support to be enabled in config.
        Images are added to the content list of the last user message.
        """ # #
        request_id = kwargs.pop("request_id", None) #
        log_prefix = f"[Req-{request_id}] " if request_id else "" #

        # Check for client initialization
        if not self.client: #
            logger.error(f"{log_prefix}HTTP client not initialized (likely missing base_url). Cannot make image request.") #
            raise RuntimeError("LLM client not initialized. Check configuration (base_url).") #

        # Check if multimodal is supported by this instance
        if not self.supports_multimodal: #
            logger.error(f"{log_prefix}ask_with_images called, but multimodal support is not enabled in configuration.") #
            raise ValueError("Multimodal support is not enabled for this LLM instance.") #
        if not images: #
            logger.warning(f"{log_prefix}ask_with_images called with no images. Falling back to standard text 'ask'.") # #
            # Fallback to standard ask if no images provided
            return await self.ask( #
                messages=messages, system_msgs=system_msgs, stream=stream, #
                temperature=temperature, max_tokens_override=max_tokens_override, #
                request_id=request_id, **kwargs # #
            ) #

        logger.info(f"{log_prefix}ask_with_images called with {len(images)} images.") #

        try: #
            # --- Prepare Messages with Images ---
            # Format text messages first
            if system_msgs: #
                formatted_system = self.format_messages(system_msgs) # #
                formatted_user = self.format_messages(messages) #
                formatted_text_messages = formatted_system + formatted_user #
            else: #
                formatted_text_messages = self.format_messages(messages) # #

            # Find the last user message to append images to
            last_user_msg_index = -1 #
            for i in range(len(formatted_text_messages) - 1, -1, -1): #
                 if formatted_text_messages[i].get("role") == "user": #
                     last_user_msg_index = i #
                     break #

            if last_user_msg_index == -1: #
                 raise ValueError("Cannot add images: No 'user' role message found in the provided messages.") # #

            last_user_message = formatted_text_messages[last_user_msg_index] #
            content = last_user_message.get("content") #

            # Ensure content is a list for multimodal input
            if content is None: content_list = [] #
            elif isinstance(content, str): content_list = [{"type": "text", "text": content}] # #
            elif isinstance(content, list): #
                # Ensure all items are dicts
                content_list = [] #
                for item in content: #
                    if isinstance(item, str): content_list.append({"type": "text", "text": item}) #
                    elif isinstance(item, dict): content_list.append(item) #
                    else: logger.warning(f"Skipping invalid item in existing content list: {type(item)}") #
            else: #
                raise ValueError(f"Invalid existing content type ({type(content)}) in user message for images.") # #

            # Add image parts to the content list
            for img_idx, img_data in enumerate(images): #
                image_part = {"type": "image_url"} # #
                url = None #
                if isinstance(img_data, str): # Assume base64 string #
                    # Add data URI prefix - ASSUME JPEG FOR NOW
                    url = f"data:image/jpeg;base64,{img_data}" # #
                elif isinstance(img_data, dict) and "url" in img_data: #
                    url = img_data["url"] # Allow passing pre-formatted {'url': 'data:...'} or external URL #
                    # Potentially add detail field if provided in img_data dict
                    if "detail" in img_data: image_part.setdefault("image_url", {})["detail"] = img_data["detail"] #
                else: #
                    raise ValueError(f"Unsupported image format in 'images' list at index {img_idx}: {type(img_data)}") # #

                if url: #
                    image_part.setdefault("image_url", {})["url"] = url #
                    content_list.append(image_part) #
                    logger.debug(f"{log_prefix}Added image part {img_idx+1}/{len(images)} to message content.") # #
                else: #
                     logger.warning(f"{log_prefix}Could not determine URL for image at index {img_idx}.") #

            last_user_message["content"] = content_list # Update the last user message #
            final_messages = formatted_text_messages # Use the modified list #
            logger.debug(f"{log_prefix}Final messages with images count: {len(final_messages)}") #


            # --- Calculate Tokens & Check Limits ---
            input_tokens = self.count_message_tokens(final_messages) #
            logger.info(f"{log_prefix}Calculated input tokens (with images): {input_tokens}") # #

            if not self.check_token_limit(input_tokens): # #
                error_message = self.get_limit_error_message(input_tokens) #
                logger.error(f"{log_prefix}{error_message}") #
                raise TokenLimitExceeded(error_message) # #

            # Update input tokens before the request
            self.update_token_count(input_tokens=input_tokens, completion_tokens=0) #

            # --- Prepare Payload & Call API --- #
            payload = self._prepare_payload( #
                messages=final_messages, #
                stream=stream, #
                temperature=temperature, #
                max_tokens=max_tokens_override, # Max completion tokens #
                # Do not pass tools/tool_choice here
                **kwargs # #
            ) #
            logger.debug(f"{log_prefix}Sending image request payload: {json.dumps(payload, indent=2)}") #

            endpoint = VLLM_API_SUFFIX #
            timeout = self.request_timeout # #

            # --- Handle Response (Similar to ask method) ---
            if not stream: #
                response = await self.client.post(endpoint, json=payload, timeout=timeout) # #
                assistant_message_dict = await self._process_response(response, input_tokens) #
                content = assistant_message_dict.get("content", "") #
                logger.debug(f"{log_prefix}Received non-streamed image response content snippet: {str(content)[:100]}...") #
                return content or "" # #
            else: #
                logger.debug(f"{log_prefix}Initiating image stream request...") # #
                async def stream_generator(): #
                    try: #
                        async with self.client.stream("POST", endpoint, json=payload, timeout=timeout) as response: # #
                            response.raise_for_status() #
                            async for delta in self._process_streaming_response(response, input_tokens): #
                                yield delta # #
                    except httpx.HTTPStatusError as e: #
                        error_body = "<failed to read error response body>" #
                        try: error_body = (await e.response.aread()).decode() # #
                        except Exception: pass #
                        logger.error(f"{log_prefix}HTTP error starting/during image stream: {e}. Response: {error_body}") # #
                        raise ValueError(f"vLLM image stream request failed ({e.response.status_code}): {error_body}") from e #
                    except Exception as e: #
                        logger.error(f"{log_prefix}Unexpected error during image streaming: {e}", exc_info=True) #
                        raise # #

                return stream_generator() #

        # --- Exception Handling ---
        except TokenLimitExceeded: #
            logger.error(f"{log_prefix}TokenLimitExceeded in ask_with_images") #
            raise # #
        except httpx.HTTPStatusError as e: # #
            status_code = e.response.status_code if hasattr(e, 'response') else 'N/A' #
            error_body = "<failed to read error response body>" #
            try: #
                if hasattr(e, 'response'): #
                    error_body = await e.response.aread() #
                    error_body = error_body.decode() # #
            except Exception: pass #
            logger.error(f"{log_prefix}HTTP Error {status_code} from vLLM API (image request): {error_body}", exc_info=True) #
            if isinstance(status_code, int) and 400 <= status_code < 500: # #
                raise ValueError(f"vLLM API image request failed (Client Error {status_code}): {error_body}") from e #
            else: #
                raise # Re-raise for retry or caller #
        except (httpx.RequestError, httpx.TimeoutException) as e: #
            logger.exception(f"{log_prefix}Network/Connection error during image request: {e}") #
            raise # Re-raise for retry or caller #
        except ValueError as ve: #
            logger.exception(f"{log_prefix}Data validation or processing error in ask_with_images: {ve}") #
            raise #