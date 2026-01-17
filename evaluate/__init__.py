"""
Detector package skeleton for watermark robustness checks.

Exposes minimal interfaces and a CLI runner to:
- generate samples via the existing chat client
- optionally apply simple attack transforms
- compute a placeholder z-score via a detector interface
- save results to a chosen output directory (from extra_args when present)
"""

__all__ = [
    "WatermarkParams",
    "BaseDetector",
]
