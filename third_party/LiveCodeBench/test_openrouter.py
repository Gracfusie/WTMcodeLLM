#!/usr/bin/env python3
"""
快速测试 OpenRouter 集成的脚本
用于验证配置是否正确
"""

import os
import sys

def test_openrouter_config():
    """测试 OpenRouter 配置"""
    print("=" * 60)
    print("OpenRouter 配置测试")
    print("=" * 60)
    
    # 检查环境变量
    endpoint = os.getenv("OPENROUTER_ENDPOINT", "https://openrouter.ai/api/v1")
    api_key = os.getenv("OPENROUTER_API_KEY")
    
    print(f"\n1. Endpoint: {endpoint}")
    if api_key:
        print(f"2. API Key: {api_key[:10]}...{api_key[-4:] if len(api_key) > 14 else '***'}")
    else:
        print("2. API Key: ❌ 未设置 (请设置 OPENROUTER_API_KEY 环境变量)")
    
    # 检查必要的包
    print("\n3. 检查依赖包:")
    try:
        import openai
        print("   ✅ openai 包已安装")
    except ImportError:
        print("   ❌ openai 包未安装，请运行: pip install openai")
        return False
    
    # 检查 runner 文件
    print("\n4. 检查文件:")
    runner_path = "lcb_runner/runner/openrouter_runner.py"
    if os.path.exists(runner_path):
        print(f"   ✅ {runner_path} 存在")
    else:
        print(f"   ❌ {runner_path} 不存在")
        return False
    
    # 测试导入
    print("\n5. 测试导入:")
    try:
        from lcb_runner.runner.openrouter_runner import OpenRouterRunner
        print("   ✅ OpenRouterRunner 可以正常导入")
    except Exception as e:
        print(f"   ❌ 导入失败: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("配置检查完成！")
    print("=" * 60)
    
    if not api_key:
        print("\n⚠️  警告: 未设置 API key，请设置环境变量:")
        print("   export OPENROUTER_API_KEY='your-key-here'")
        print("\n或者使用命令行参数:")
        print("   --custom_api_key your-key-here")
    
    print("\n使用示例:")
    print("python -m lcb_runner.runner.main \\")
    print("    --model test-model \\")
    print("    --custom_endpoint https://openrouter.ai/api/v1 \\")
    print("    --custom_api_key your-key \\")
    print("    --custom_model_name openai/gpt-4 \\")
    print("    --scenario codegeneration \\")
    print("    --debug \\")
    print("    --evaluate")
    
    return True

if __name__ == "__main__":
    success = test_openrouter_config()
    sys.exit(0 if success else 1)

