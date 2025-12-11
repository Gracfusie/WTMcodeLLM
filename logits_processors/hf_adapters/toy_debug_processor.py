from __future__ import annotations

from typing import Callable, Optional, Sequence

import torch
from transformers import LogitsProcessor


class ToyDebugLogitsProcessor(LogitsProcessor):
    """极简调试用 HF LogitsProcessor。

    - 不改 logits。
    - 仅打印前缀和 top-1 预测，便于在 HF 流水线快速 smoke test。
    """

    def __init__(
        self,
        tokenizer: Optional[object] = None,
        print_first_n: int = 1,
        printer: Callable[[str], None] = print,
        name: str = "ToyDebug",
        log_file: Optional[str] = None,
    ) -> None:
        if tokenizer is None:
            raise ValueError("ToyDebugLogitsProcessor requires a tokenizer.")

        self.tokenizer = tokenizer
        self.print_first_n = print_first_n
        self.printer = printer
        self.name = name
        self.log_file = log_file

    def _decode(self, token_ids: Sequence[int]) -> list[str]:
        if self.tokenizer is None:
            return [str(tid) for tid in token_ids]
        decoded: list[str] = []
        for tid in token_ids:
            try:
                decoded.append(self.tokenizer.decode([tid]))
            except Exception:
                decoded.append(str(tid))
        return decoded

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
    ) -> torch.FloatTensor:
        batch = min(self.print_first_n, input_ids.shape[0])
        for b in range(batch):
            prefix_ids = input_ids[b].tolist()
            prefix_text = (
                self.tokenizer.decode(prefix_ids)
                if self.tokenizer is not None
                else str(prefix_ids)
            )

            top_val, top_id = torch.max(scores[b], dim=-1)
            top_id_int = int(top_id)
            top_token = self._decode([top_id_int])[0]
            log_line = (
                f"[{self.name}] batch={b} "
                f"prefix={prefix_text} "
                f"next='{top_token}' "
                f"(id={top_id_int}, logit={float(top_val):.3f})"
            )
            self.printer(log_line)
            if self.log_file:
                try:
                    with open(self.log_file, "a", encoding="utf-8") as f:
                        f.write(log_line + "\n")
                except Exception:
                    pass

        return scores
 