import os
import torch
from rich.console import Console
from vllm.sampling_params import SamplingParams
from config import EnvConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from modelscope import snapshot_download
from vllm.v1.sample.logits_processor import BatchUpdate, MoveDirectionality
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

def get_model_path(model_name: str) -> str:
    if os.environ.get('VLLM_USE_MODELSCOPE', False):
        return snapshot_download(model_name)
    else:
        return model_name

console = Console()

def get_common_tokens(tokenizer1, tokenizer2):
    """
    检查 tokenizer2 (proxy) 的词汇表是否是 tokenizer1 (main) 的子集。
    """
    def get_id_content_dict(tk):
        vocab = {v: k for k, v in tk.get_vocab().items()}
        vocab.update({k: v.content for k, v in tk.added_tokens_decoder.items()})
        return vocab  # {token_id: token_content}

    main_vocab = get_id_content_dict(tokenizer1)  # main model (larger one)
    proxy_vocab = get_id_content_dict(tokenizer2)  # proxy model

    # 检查 proxy 是否是 main 的子集
    proxy_token_ids = set(proxy_vocab.keys())
    main_token_ids = set(main_vocab.keys())
    
    # proxy 不应该有 main 没有的 token
    new_tokens = proxy_token_ids - main_token_ids
    if new_tokens:
        console.print(
            f"x Proxy 模型引入了 {len(new_tokens)} 个 main 模型没有的新 token: {list(new_tokens)[:10]}...",
            style="bold red",
        )
        raise ValueError("Proxy 模型不得引入 main 模型没有的新 token")
    
    # 检查共有部分是否完全相等
    for token_id in proxy_token_ids:
        if proxy_vocab[token_id] != main_vocab[token_id]:
            console.print(
                f"x Token ID {token_id} 在两个模型中的内容不一致:\n"
                f"  Main: {repr(main_vocab[token_id])}\n"
                f"  Proxy: {repr(proxy_vocab[token_id])}",
                style="bold red",
            )
            raise ValueError("Proxy 和 Main 模型在共有 token 上的内容必须完全相等")
    
    console.print(
        f"✓ Proxy 模型词汇表检查通过:\n"
        f"  Main 模型词汇量: {len(main_token_ids)}\n"
        f"  Proxy 模型词汇量: {len(proxy_token_ids)}\n"
        f"  覆盖率: {len(proxy_token_ids) / len(main_token_ids) * 100:.2f}%",
        style="bold green",
    )
    
    return sorted(list(proxy_token_ids))

    # ============ 旧的 IOU 计算逻辑 ============
    # def get_id_content_set(tk):
    #     vocab = {v: k for k, v in tk.get_vocab().items()}
    #     vocab.update({k: v.content for k, v in tk.added_tokens_decoder.items()})
    #     return set(vocab.items())
    #
    # set1 = get_id_content_set(tokenizer1)
    # set2 = get_id_content_set(tokenizer2)
    #
    # intersection = set1 & set2
    # union = set1 | set2
    #
    # iou = len(intersection) / len(union) if union else 0.0
    # console.print(
    #     f"Tokenizers IoU (如果太小，可能说明大小模型用的 tokenizer 不同): {iou:.4f} "
    #     f"(Common: {len(intersection)} / Union: {len(union)})",
    #     style="bold bright_red",
    # )
    # assert iou > 0.9, "IoU 过小，请确定大小模型是否同 Family"
    #
    # return sorted([item[0] for item in intersection])

@dataclass
class RequestState:
    kv_cache: Tuple[Tuple[torch.Tensor, torch.Tensor], ...]
    prompt_tok_ids: List[int]
    output_tok_ids: List[int]
    next_tok_logits: torch.Tensor = None
    watermarked_count: int = 0
    non_watermarked_count: int = 0
    entropy_threshold: Optional[float] = None
    delta: Optional[float] = None
    window_size: Optional[int] = None

class ProxyModelManager:

    def __init__(self, main_tokenizer: AutoTokenizer, device: torch.device, is_pin_memory: bool):
        self.device = device
        self.is_pin_memory = is_pin_memory

        # main_model_path = get_model_path(EnvConfig.main_model)
        proxy_model_path = get_model_path(EnvConfig.watermark_proxy_model)
        
        self.proxy_model = AutoModelForCausalLM.from_pretrained(
            proxy_model_path,
            device_map=device, 
            torch_dtype="auto",
        )
        self.proxy_model.eval()
        self.proxy_tokenizer = AutoTokenizer.from_pretrained(proxy_model_path, device_map=device)
        # self.main_tokenizer = AutoTokenizer.from_pretrained(main_model_path, device_map=device)
        self.main_tokenizer = main_tokenizer

        # 只能对 common_ids 计算 green / red list
        self.common_ids = get_common_tokens(self.main_tokenizer, self.proxy_tokenizer)

        self.req_info_by_idx: Dict[int, RequestState] = {}
        self.req_info_by_obj: Dict[int, RequestState] = {}

        # sanity check
        _apple_toks = self.proxy_tokenizer.encode("apple")
        assert isinstance(_apple_toks, list) and isinstance(_apple_toks[0], int) and len(_apple_toks) == 1, "apple 的 token id 应该是一个整数"

        self.prefix_tok_ids = self.proxy_tokenizer.encode(EnvConfig.watermark_proxy_template_prefix)
        self.suffix_tok_ids = self.proxy_tokenizer.encode(EnvConfig.watermark_proxy_template_suffix)

    @classmethod
    def validate_params(cls, params: SamplingParams):
        overrides = {}
        if params.extra_args:
            for key in ['watermark_entropy_threshold', 'watermark_delta', 'watermark_proxy_window_size']:
                if key in params.extra_args:
                    overrides[key] = params.extra_args[key]
        return overrides if overrides else None
    
    # 注意：BatchUpdate 的注释中有写 the `output_tok_ids` list (which is an element of each
    # tuple in `added`) is a reference to the request's running output tokens
    # list
    def update_state(self, batch_update: BatchUpdate | None):
        if batch_update:
            # Process added requests.
            for index, params, prompt_tok_ids, output_tok_ids in batch_update.added:
                assert prompt_tok_ids is not None, "只支持对话模式"
                
                assert params is not None
                overrides = self.validate_params(params)
                if (id(output_tok_ids) in self.req_info_by_obj):
                    request_state = self.req_info_by_obj[id(output_tok_ids)]
                    print ("加回一个曾经处理过的 request")
                else:
                    request_state = RequestState(
                        kv_cache=None,
                        prompt_tok_ids=prompt_tok_ids,
                        output_tok_ids=output_tok_ids,
                        entropy_threshold=overrides.get('watermark_entropy_threshold') if overrides else None,
                        delta=overrides.get('watermark_delta') if overrides else None,
                        window_size=overrides.get('watermark_proxy_window_size') if overrides else None,
                    )
                    if overrides:
                        console.print(f"🔥 收到带 OVERRIDE 的 Request: {overrides}", style="bold bright_magenta")
                    else:
                        print ("创建新 Request 对象")
                self.req_info_by_idx[index] = request_state
                self.req_info_by_obj[id(output_tok_ids)] = request_state

            if self.req_info_by_idx:
                # Process removed requests.
                for index in batch_update.removed:
                    req_state = self.req_info_by_idx.get(index)
                    if req_state and req_state.output_tok_ids:
                        last_token = req_state.output_tok_ids[-1]
                        eos_id = self.main_tokenizer.eos_token_id
                        if last_token == eos_id:
                            console.print(f"[Stats] Watermarked: {req_state.watermarked_count}, Non-watermarked: {req_state.non_watermarked_count}", style="bold yellow")
                    self.req_info_by_idx.pop(index, None)

                # Process moved requests, unidirectional move (a->b) and swap
                # (a<->b)
                for adx, bdx, direct in batch_update.moved:
                    a_val = self.req_info_by_idx.pop(adx, None)
                    b_val = self.req_info_by_idx.pop(bdx, None)
                    if a_val is not None:
                        self.req_info_by_idx[bdx] = a_val
                    if direct == MoveDirectionality.SWAP and b_val is not None:
                        self.req_info_by_idx[adx] = b_val

        # sanity_check idx 连续
        assert sorted(self.req_info_by_idx.keys()) == list(range(len(self.req_info_by_idx))), "idx 不连续"

        # 在这里进行无 batching 的 inference
        # 虽然很蠢但是如果不这么做就得改 vllm engine 了
        
        with torch.inference_mode():
            for req_state in self.req_info_by_idx.values():

                window_size = req_state.window_size if req_state.window_size is not None else EnvConfig.window_size
                if window_size > 0:
                    proxy_in_tok_ids = self.prefix_tok_ids + req_state.output_tok_ids[-window_size:] + self.suffix_tok_ids
                else:
                    proxy_in_tok_ids = self.prefix_tok_ids + req_state.output_tok_ids + self.suffix_tok_ids

                in_long_tensor = torch.tensor(proxy_in_tok_ids, device=self.device, dtype=torch.long).unsqueeze(0) # TODO

                if req_state.kv_cache is not None:
                    # 暂未实现
                    import pudb.remote; pudb.remote.set_trace()
                    # sanity check: 只应该比上次多两个 token
                    # assert len(proxy_in_tok_ids) == cached_tokens + 2

                if req_state.kv_cache is None:
                    outputs = self.proxy_model(
                        input_ids=in_long_tensor,
                        return_dict=True
                    )
                else:
                    # 暂时不实现 kv cache
                    assert (False)
                    assert (EnvConfig.window_size == -1), "window_size 不为 -1 时无法实现 kv cache"
                    # outputs = self.proxy_model(
                    #     input_ids=in_long_tensor,
                    #     past_key_values=req_state.kv_cache,
                    #     # position_ids
                    #     use_cache=True,
                    #     return_dict=True
                    # )
                
                req_state.next_tok_logits = outputs.logits[0, -1, :].clone()

