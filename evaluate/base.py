from abc import ABC, abstractmethod
from typing import Sequence

from .watermark_params import WatermarkParams


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
