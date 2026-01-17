from abc import ABC, abstractmethod
from typing import Sequence, Optional
import os
import sys
from pathlib import Path

from .watermark_params import WatermarkParams

# Lazy imports for proxy detector dependencies
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
except ImportError:  # pragma: no cover - optional runtime dep
    torch = None
    AutoTokenizer = None
    AutoModelForCausalLM = None

_ACW_DIR = Path(__file__).resolve().parent.parent / "logits_processors" / "utils"
if str(_ACW_DIR) not in sys.path:
    sys.path.insert(0, str(_ACW_DIR))
try:  # type: ignore
    from watermark import ProxyLogitsGuidedWatermarker, ProxyWatermarkDetector
except ImportError:  # pragma: no cover - optional runtime dep
    ProxyLogitsGuidedWatermarker = None
    ProxyWatermarkDetector = None


class BaseDetector(ABC):
    """
    Abstract base for watermark detectors.

    Given a text sample and its corresponding watermark parameters,
    computes a statistical z-score used to infer watermark presence.
    """

    @abstractmethod
    def detect(self, text: str, params: WatermarkParams) -> float:
        """
        Compute z-score for a single sample.
        """
        raise NotImplementedError

    def batch_detect(self, samples: Sequence[str], params: WatermarkParams) -> Sequence[float]:
        """Optional convenience: detect across multiple samples."""
        return [self.detect(s, params) for s in samples]


class DummyZScoreDetector(BaseDetector):
    """
    Placeholder detector implementation.

    This is a stub that returns a deterministic pseudo z-score based on
    simple text heuristics. Replace with a real hypothesis test.
    """

    def detect(self, text: str, params: WatermarkParams) -> float:
        # Example heuristic: scale by length and presence of punctuation
        length = max(len(text), 1)
        punct = sum(ch in ".,;:!?" for ch in text)
        base = (punct + 1) / (length ** 0.5)

        # Nudge based on params to show wiring is in place
        delta = params.delta or 0.0
        entropy = (params.entropy_threshold or 0.0) * 0.1
        window = (params.proxy_window_size or 0) * 0.001
        return base + delta + entropy + window


class ProxyDetector(BaseDetector):
    """
    Proxy-guided watermark detector mirroring interact_with_detector.py proxy logic.

    Loads a proxy LM and uses ProxyWatermarkDetector to produce a z-score.
    """

    def __init__(
        self,
        proxy_model: Optional[str] = None,
        entropy_threshold_default: Optional[float] = None,
        secret_key: Optional[int] = None,
        window_size: Optional[int] = None,
        z_threshold: float = 4.0,
        prefix_str: Optional[str] = None,
        suffix_str: Optional[str] = None,
        device_override: Optional[str] = None,
    ) -> None:
        if ProxyLogitsGuidedWatermarker is None or ProxyWatermarkDetector is None:
            raise ImportError("Proxy watermark modules not found; check logits_processors/utils/watermark.py")
        if AutoTokenizer is None or AutoModelForCausalLM is None:
            raise ImportError("transformers is required for ProxyDetector")

        self.device = device_override or ("cuda" if torch and torch.cuda.is_available() else "cpu")
        self.z_threshold = z_threshold
        self.entropy_default = entropy_threshold_default
        self.window_default = window_size

        self.proxy_model_name = proxy_model or os.environ.get("WATERMARK_PROXY_MODEL")
        if not self.proxy_model_name:
            raise ValueError("ProxyDetector requires proxy_model (env WATERMARK_PROXY_MODEL or arg)")

        # Resolve model path (use snapshot_download if available, else raw path)
        model_path = self.proxy_model_name
        try:  # pragma: no cover - optional dependency
            from modelscope import snapshot_download

            model_path = snapshot_download(self.proxy_model_name)
        except Exception:
            pass

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.proxy_model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="auto").to(self.device).eval()

        self.prefix_str = prefix_str or os.environ.get("WATERMARK_PROXY_TEMPLATE_PREFIX", "<|fim_prefix|>")
        self.suffix_str = suffix_str or os.environ.get("WATERMARK_PROXY_TEMPLATE_SUFFIX", "<|fim_suffix|>\n<|fim_middle|>")
        self.secret_key = secret_key if secret_key is not None else _safe_int_env("WATERMARK_SECRET_KEY")

        prefix_ids = self.tokenizer.encode(self.prefix_str, add_special_tokens=False)
        suffix_ids = self.tokenizer.encode(self.suffix_str, add_special_tokens=False)

        core_logic = ProxyLogitsGuidedWatermarker(
            entropy_threshold=self.entropy_default,
            vocab_size=len(self.tokenizer.get_vocab()),
            secret_key=self.secret_key,
        )

        self.detector = ProxyWatermarkDetector(
            watermarker=core_logic,
            proxy_model=self.proxy_model,
            tokenizer=self.tokenizer,
            device=self.device,
            prefix_ids=prefix_ids,
            suffix_ids=suffix_ids,
            window_size=self.window_default if self.window_default is not None else -1,
        )

    def detect(self, text: str, params: WatermarkParams) -> float:
        # Update detector thresholds per-call from params when provided
        if hasattr(self.detector.watermarker, "entropy_threshold"):
            self.detector.watermarker.entropy_threshold = (
                params.entropy_threshold
                if params.entropy_threshold is not None
                else self.entropy_default
            )
        if params.proxy_window_size is not None:
            # ProxyWatermarkDetector exposes window_size attr
            setattr(self.detector, "window_size", params.proxy_window_size)

        score_dict = self.detector.detect(text=text, z_threshold=self.z_threshold)
        # Expect z_score key; fallback to 0.0 if missing
        return float(score_dict.get("z_score", 0.0))


def _safe_int_env(key: str) -> Optional[int]:
    val = os.environ.get(key)
    if val is None:
        return None
    try:
        return int(val)
    except ValueError:
        return None
