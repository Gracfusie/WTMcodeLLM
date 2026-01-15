import os
import json
from rich.console import Console

class EnvConfig:
    main_model: str = os.environ["MAIN_MODEL"]
    watermark_proxy_model: str = os.environ["WATERMARK_PROXY_MODEL"]
    watermark_proxy_template_prefix: str = os.environ["WATERMARK_PROXY_TEMPLATE_PREFIX"]
    watermark_proxy_template_suffix: str = os.environ["WATERMARK_PROXY_TEMPLATE_SUFFIX"]
    watermark_entropy_threshold: float = float(os.environ.get("WATERMARK_ENTROPY_THRESHOLD", "0.5"))
    watermark_delta: float = float(os.environ.get("WATERMARK_DELTA", "2.0"))
    watermark_secret_key: int = int(os.environ.get("WATERMARK_SECRET_KEY", "42"))
    window_size: int = int(os.environ.get("WATERMARK_PROXY_WINDOW_SIZE", "-1"))

console = Console()

console.print("模型相关环境配置:", style="bold bright_red")
console.print(f"主模型: {EnvConfig.main_model}", style="bold bright_red")
console.print(f"Proxy 模型: {EnvConfig.watermark_proxy_model}", style="bold bright_red")
console.print(f"Proxy 模式 Template 例: {json.dumps(EnvConfig.watermark_proxy_template_prefix + "#include <stdio.h>" + EnvConfig.watermark_proxy_template_suffix)}", style="bold bright_red")
console.print("-" * 100, style="bold bright_blue")
console.print("水印相关环境配置:", style="bold bright_red")
console.print(f"熵阈值: {EnvConfig.watermark_entropy_threshold}", style="bold bright_red")
console.print(f"Delta: {EnvConfig.watermark_delta}", style="bold bright_red")
console.print(f"密钥: {EnvConfig.watermark_secret_key}", style="bold bright_red")
console.print("-" * 100, style="bold bright_red")