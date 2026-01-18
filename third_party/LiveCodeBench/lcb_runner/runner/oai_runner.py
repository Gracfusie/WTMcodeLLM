import os
from time import sleep
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import openai
    from openai import OpenAI
except ImportError as e:
    pass

from lcb_runner.lm_styles import LMStyle
from lcb_runner.runner.base_runner import BaseRunner


def _openai_support_n() -> bool:
    val = os.environ.get("OPENAI_SUPPORT_N", "1")
    return val.lower() not in ("false", "0", "no", "off")


class OpenAIRunner(BaseRunner):
    client = OpenAI(
        api_key=os.getenv("OPENAI_KEY"),
    )

    def __init__(self, args, model):
        super().__init__(args, model)
        self.n = args.n
        if model.model_style == LMStyle.OpenAIReasonPreview:
            self.client_kwargs: dict[str | str] = {
                "model": args.model,
                "max_completion_tokens": 25000,
            }
        elif model.model_style == LMStyle.OpenAIReason:
            assert (
                "__" in args.model
            ), f"Model {args.model} is not a valid OpenAI Reasoning model as we require reasoning effort in model name."
            model, reasoning_effort = args.model.split("__")
            self.client_kwargs: dict[str | str] = {
                "model": model,
                "reasoning_effort": reasoning_effort,
            }
        else:
            self.client_kwargs: dict[str | str] = {
                "model": args.model,
                "temperature": args.temperature,
                "max_tokens": args.max_tokens,
                "top_p": args.top_p,
                "frequency_penalty": 0,
                "presence_penalty": 0,
                "n": args.n if _openai_support_n() else 1,
                "timeout": args.openai_timeout,
                # "stop": args.stop, --> stop is only used for base models currently
            }
            if os.environ.get("OPENAI_EXTRA_BODY"):
                self.client_kwargs["extra_body"] = json.loads(os.environ.get("OPENAI_EXTRA_BODY"))

    def _run_single(self, prompt: list[dict[str, str]], retry: int = 10) -> list[str]:
        if _openai_support_n():
            return self._run_single_with_n(prompt, retry)
        
        results = []
        with ThreadPoolExecutor(max_workers=self.n) as executor:
            futures = [executor.submit(self._run_single_with_n, prompt, retry) for _ in range(self.n)]
            for future in as_completed(futures):
                results.extend(future.result())
        return results

    def _run_single_with_n(self, prompt: list[dict[str, str]], retry: int = 10) -> list[str]:
        assert isinstance(prompt, list)

        if retry == 0:
            print("Max retries reached. Returning empty response.")
            return []

        try:
            response = OpenAIRunner.client.chat.completions.create(
                messages=prompt,
                **self.client_kwargs,
            )
        except (
            openai.APIError,
            openai.RateLimitError,
            openai.InternalServerError,
            openai.OpenAIError,
            openai.APIStatusError,
            openai.APITimeoutError,
            openai.InternalServerError,
            openai.APIConnectionError,
            json.decoder.JSONDecodeError
        ) as e:
            print("Exception: ", repr(e))
            print("Sleeping for 30 seconds...")
            print("Consider reducing the number of parallel processes.")
            sleep(30)
            return self._run_single_with_n(prompt, retry=retry - 1)
        except Exception as e:
            print(f"Failed to run the model for {prompt}!")
            print("Exception: ", repr(e))
            import traceback; traceback.print_exc()
            raise e
        return [c.message.content for c in response.choices]
