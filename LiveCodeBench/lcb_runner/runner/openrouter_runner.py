import os
import json
from time import sleep

try:
    import openai
    from openai import OpenAI
except ImportError as e:
    pass

from lcb_runner.runner.base_runner import BaseRunner


class OpenRouterRunner(BaseRunner):
    """
    Runner for OpenRouter API compatible models.
    Supports custom endpoint, API key, and model name via command line arguments.
    """
    
    def __init__(self, args, model):
        super().__init__(args, model)
        
        # Get endpoint and API key from args or environment variables
        endpoint = args.custom_endpoint or os.getenv("OPENROUTER_ENDPOINT", "https://openrouter.ai/api/v1")
        api_key = args.custom_api_key or os.getenv("OPENROUTER_API_KEY")
        
        if not api_key:
            raise ValueError(
                "OpenRouter API key is required. "
                "Set it via --custom_api_key or OPENROUTER_API_KEY environment variable."
            )
        
        # Normalize endpoint: remove trailing /chat/completions if present
        # OpenAI SDK automatically adds /chat/completions to the base_url
        endpoint = endpoint.rstrip("/")
        if endpoint.endswith("/chat/completions"):
            endpoint = endpoint[:-16]  # Remove "/chat/completions"
            print(f"Warning: Removed /chat/completions from endpoint. Using: {endpoint}")
        if not endpoint.endswith("/v1"):
            # Ensure it ends with /v1 for OpenAI compatibility
            if not endpoint.endswith("/v1/"):
                endpoint = endpoint.rstrip("/") + "/v1"
        
        # Create client instance (not class variable to support different endpoints/keys)
        self.client = OpenAI(
            api_key=api_key,
            base_url=endpoint,
        )
        
        print(f"Initialized OpenRouter client with endpoint: {endpoint}")
        
        # Configure API call parameters
        model_name = args.custom_model_name or args.model
        # Set n=1 and loop to generate multiple candidates (like DeepSeekRunner)
        # Some APIs don't support returning multiple candidates in one call
        self.client_kwargs: dict[str | str] = {
            "model": model_name,  # Use custom model name if provided
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "top_p": args.top_p,
            "frequency_penalty": 0,
            "presence_penalty": 0,
            "n": 1,  # Always use n=1, then loop args.n times
            "timeout": args.openai_timeout,
        }
        
        print(f"Using model: {model_name}")
        print(f"API endpoint: {endpoint}")
        print(f"Will generate {args.n} candidates by making {args.n} API calls")

    def _run_single(self, prompt: list[dict[str, str]], n: int = 10) -> list[str]:
        """
        Execute API calls to OpenRouter to generate multiple candidates.
        Since OpenRouter may not support returning multiple candidates in one call,
        we loop args.n times with n=1 to generate multiple candidates.
        
        Args:
            prompt: List of message dictionaries (OpenAI format)
            n: Number of retries (internal use, not used in this implementation)
        
        Returns:
            List of generated text strings (length = args.n)
        """
        assert isinstance(prompt, list), "Prompt should be a list of messages"

        def __run_single_attempt(counter):
            """Single API call attempt with retry logic"""
            if counter == 0:
                print("Max retries reached. Returning empty response.")
                return None
                
            try:
                response = self.client.chat.completions.create(
                    messages=prompt,
                    **self.client_kwargs,
                )
                # Since n=1, we should only get one choice
                if response.choices and len(response.choices) > 0:
                    content = response.choices[0].message.content
                    if content is None:
                        print("Warning: API returned None content")
                        return ""
                    return content
                else:
                    print("Warning: API returned no choices")
                    return None
            except json.JSONDecodeError as e:
                # Handle JSON decode errors (API returned non-JSON response)
                print(f"JSON Decode Error: API returned invalid JSON response")
                print(f"Error details: {repr(e)}")
                print(f"Model: {self.client_kwargs.get('model')}")
                print(f"Endpoint: {self.client.base_url}")
                print("This might indicate:")
                print("  1. API endpoint is incorrect")
                print("  2. API returned an error page (HTML instead of JSON)")
                print("  3. Network issue or API server problem")
                if counter > 1:
                    print("Sleeping for 30 seconds before retry...")
                    sleep(30)
                    return __run_single_attempt(counter - 1)
                else:
                    # Return empty string instead of raising to continue processing
                    print("Returning empty response after max retries")
                    return ""
            except openai.NotFoundError as e:
                # 404 error - usually means model not found or wrong endpoint
                print(f"404 Not Found Error:")
                print(f"  Model: {self.client_kwargs.get('model')}")
                print(f"  Endpoint: {self.client.base_url}")
                print(f"  Error: {repr(e)}")
                print("\nPossible causes:")
                print("  1. Model name is incorrect. Check available models at https://openrouter.ai/models")
                print("  2. Endpoint URL is wrong. Should be: https://openrouter.ai/api/v1 (without /chat/completions)")
                print("  3. API key doesn't have access to this model")
                if counter > 1:
                    print("Sleeping for 30 seconds before retry...")
                    sleep(30)
                    return __run_single_attempt(counter - 1)
                else:
                    raise e
            except (
                openai.APIError,
                openai.RateLimitError,
                openai.InternalServerError,
                openai.OpenAIError,
                openai.APIStatusError,
                openai.APITimeoutError,
                openai.APIConnectionError,
            ) as e:
                print("Exception: ", repr(e))
                print(f"Model: {self.client_kwargs.get('model')}")
                print(f"Endpoint: {self.client.base_url}")
                print("Sleeping for 30 seconds...")
                print("Consider reducing the number of parallel processes.")
                sleep(30)
                return __run_single_attempt(counter - 1)
            except Exception as e:
                # Catch all other exceptions including JSONDecodeError from httpx
                error_type = type(e).__name__
                error_msg = str(e)
                
                # Check if it's a JSON decode error (might be wrapped in httpx or openai)
                is_json_error = (
                    "JSONDecodeError" in error_type or 
                    "Expecting value" in error_msg or
                    "JSON" in error_type or
                    isinstance(e, json.JSONDecodeError)
                )
                
                if is_json_error:
                    print(f"JSON Decode Error: API returned invalid JSON response")
                    print(f"Error type: {error_type}")
                    print(f"Error details: {error_msg[:200]}...")  # Truncate long messages
                    print(f"Model: {self.client_kwargs.get('model')}")
                    print(f"Endpoint: {self.client.base_url}")
                    print("This might indicate:")
                    print("  1. API endpoint is incorrect")
                    print("  2. API returned an error page (HTML instead of JSON)")
                    print("  3. Network issue or API server problem")
                    print("  4. API rate limit or quota exceeded")
                    print("  5. Response was truncated or corrupted")
                    if counter > 1:
                        print("Sleeping for 30 seconds before retry...")
                        sleep(30)
                        return __run_single_attempt(counter - 1)
                    else:
                        print("Returning empty response after max retries")
                        return ""
                else:
                    print(f"Unexpected error type: {error_type}")
                    print(f"Error: {error_msg[:200]}...")  # Truncate long messages
                    if counter > 1:
                        print("Sleeping for 30 seconds before retry...")
                        sleep(30)
                        return __run_single_attempt(counter - 1)
                    else:
                        print("Returning empty response after max retries")
                        return ""

        # Generate args.n candidates by making args.n separate API calls
        outputs = []
        for i in range(self.args.n):
            try:
                result = __run_single_attempt(10)
                if result is not None and result != "":
                    outputs.append(result)
                else:
                    # If we can't get a result, add empty string to maintain length
                    outputs.append("")
            except Exception as e:
                # If a single generation fails, add empty string and continue
                print(f"Warning: Failed to generate candidate {i+1}/{self.args.n}, using empty string")
                outputs.append("")
        
        # Ensure we have exactly args.n outputs
        while len(outputs) < self.args.n:
            outputs.append("")
        
        return outputs

