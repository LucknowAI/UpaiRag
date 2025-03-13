import os
import tenacity
from litellm import batch_completion, completion

class UpaiLLM:
    def __init__(self, llm_identifier=None, access_token=None, llm_params=None):
        """
        Initialize a client for batch processing requests to language models.
        
        Args:
            access_token: Authentication token for the LLM service
            llm_identifier: Specific LLM model to use (defaults to value in llm_params)
            llm_params: Dictionary of parameters like {"access_token": "", "temperature": ""}
        """
        # Initialize configuration parameters
        self.llm_params = llm_params or {}
        
        # Set model identifier with fallback to configuration
        self.llm_identifier = llm_identifier or self.llm_params.get("model")
        
        # Set access token with priority order
        self.access_token = access_token or self.llm_params.get("access_token")
        
        # Validate essential parameters
        if not self.access_token:
            raise ValueError(
                "Access token must be provided either directly or in the llm_params."
            )
            
        if not self.llm_identifier:
            raise ValueError(
                "LLM model identifier must be provided either directly or in the llm_params."
            )
    
    def construct_request_payload(self, prompt_sequences):
        """
        Construct the complete payload for LLM API requests.
        
        Args:
            prompt_sequences: List of message sequences to be processed
            
        Returns:
            Dictionary with the complete request configuration
        """
        # Create the base request configuration
        request_payload = {
            "model": self.llm_identifier,
            "messages": prompt_sequences,
            "temperature": 0.0,  # Default to deterministic responses
            "api_key": self.access_token,
        }
        
        # Merge with any additional parameters
        if self.llm_params:
            request_payload.update(self.llm_params)
            
        return request_payload

  
    def process_batch_prompts(self, prompt_sequences, debug_mode=False):
        """
        Send multiple prompt sequences to the LLM and collect responses.
        
        Args:
            prompt_sequences: List of message sequences for batch processing
            debug_mode: Enable verbose logging of raw LLM responses
            
        Returns:
            List of standardized response objects with completion text and metadata
        """
        try:
            # Submit batch request to the LLM service
            llm_responses = batch_completion(**self.construct_request_payload(prompt_sequences))
            
            # Log detailed response information if debug enabled
            if debug_mode:
                print("LLM raw response:", llm_responses)
            
            # Process successful responses into standardized format
            return [
                {
                    "status": True,
                    "content": response.choices[0].message.content,  # Extract completion text
                    "original_query": messages[0]["content"],  # Store original query
                    "usage": (
                        response.usage.to_dict() if hasattr(response, "usage") else None
                    ),  # Include token usage stats if available
                }
                for response, messages in zip(llm_responses, prompt_sequences)
            ]
        except Exception as e:
            # Handle errors with standardized error response format
            return [
                {
                    "status": False,
                    "error": str(e),
                    "original_query": messages[0]["content"],
                }
                for messages in prompt_sequences
            ]

# Usage example:
# llm_client = UpaiLLM("claude-3-sonnet-20240229", "your-access-token")
# completions = llm_client.process_batch_prompts([
#     [{"role": "user", "content": "Explain quantum computing"}],
#     [{"role": "user", "content": "Summarize reinforcement learning"}]
# ])
