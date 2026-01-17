from dataclasses import dataclass
from typing import Optional, Dict, Any


@dataclass
class WatermarkParams:
    """
    Parameters required by the detector to compute a z-score for a sample.

    These correspond to runtime overrides carried in `extra_args` (see interact.py):
    - entropy_threshold: optional float override for watermark entropy threshold
    - delta: optional float override for watermark delta
    - proxy_window_size: optional int override for watermark proxy window size
    """

    entropy_threshold: Optional[float] = None
    delta: Optional[float] = None
    proxy_window_size: Optional[int] = None

    @staticmethod
    def from_extra_args(extra_args: Optional[Dict[str, Any]]) -> "WatermarkParams":
        if not extra_args:
            return WatermarkParams()
        return WatermarkParams(
            entropy_threshold=extra_args.get("watermark_entropy_threshold"),
            delta=extra_args.get("watermark_delta"),
            proxy_window_size=extra_args.get("watermark_proxy_window_size"),
        )
