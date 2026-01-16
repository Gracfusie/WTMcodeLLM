# OpenRouter 模型测评使用指南

本指南说明如何使用 LiveCodeBench 测评通过 OpenRouter 或其他自定义 API endpoint 访问的模型。

## 功能特点

✅ **无需修改代码**：通过命令行参数即可使用  
✅ **不影响现有功能**：完全向后兼容  
✅ **支持环境变量**：可通过环境变量配置  
✅ **灵活配置**：支持自定义 endpoint、API key 和模型名

## 快速开始

### 方式一：使用命令行参数（推荐）

```bash
python -m lcb_runner.runner.main \
    --model any-model-name \
    --custom_endpoint https://openrouter.ai/api/v1 \
    --custom_api_key your-api-key-here \
    --custom_model_name openai/gpt-4 \
    --scenario codegeneration \
    --evaluate \
    --eval_limit 10
```

**⚠️ 重要提示**：所有参数必须在同一行，或者使用反斜杠 `\` 续行。不要将参数分开执行。

### 方式二：使用环境变量

```bash
# 设置环境变量
# ⚠️ 注意：endpoint 不要包含 /chat/completions，SDK 会自动添加
export OPENROUTER_ENDPOINT="https://openrouter.ai/api/v1"
export OPENROUTER_API_KEY=""

# 运行测评
python -m lcb_runner.runner.main \
    --model any-model-name \
    --custom_model_name qwen/qwen3-coder-30b-a3b-instruct \
    --scenario codegeneration \
    --evaluate \
    --eval_limit 50
    
```

### 方式三：混合使用（命令行参数优先）

```bash
# 环境变量作为默认值，命令行参数会覆盖
export OPENROUTER_API_KEY="default-key"

python -m lcb_runner.runner.main \
    --model any-model-name \
    --custom_endpoint https://openrouter.ai/api/v1 \
    --custom_api_key override-key \
    --custom_model_name openai/gpt-4 \
    --scenario codegeneration \
    --evaluate
```

## 参数说明

| 参数 | 说明 | 必需 | 环境变量 |
|------|------|------|----------|
| `--custom_endpoint` | API endpoint URL | 否 | `OPENROUTER_ENDPOINT` |
| `--custom_api_key` | API key | 是* | `OPENROUTER_API_KEY` |
| `--custom_model_name` | 实际使用的模型名称 | 否 | 无 |
| `--eval_limit` | 限制评估的问题数量 | 否 | 无 |

*如果未通过命令行参数提供，必须设置环境变量 `OPENROUTER_API_KEY`

**注意**：
- `--eval_limit` 只在 `--evaluate` 时生效
- 如果不指定 `--eval_limit`，将评估所有问题
- 指定 `--eval_limit N` 时，只会生成和评估前 N 个问题

**注意**：
- 如果提供了 `--custom_endpoint`、`--custom_api_key` 或 `--custom_model_name` 中的任意一个，系统会自动使用 OpenRouter runner
- `--model` 参数仍然需要提供（用于输出文件命名），但实际使用的模型名由 `--custom_model_name` 决定
- 如果未提供 `--custom_model_name`，将使用 `--model` 的值

## OpenRouter 示例

### 测评 GPT-4

```bash
python -m lcb_runner.runner.main \
    --model gpt-4-openrouter \
    --custom_endpoint https://openrouter.ai/api/v1 \
    --custom_api_key sk-or-v1-xxxxx \
    --custom_model_name openai/gpt-4 \
    --scenario codegeneration \
    --n 10 \
    --temperature 0.2 \
    --evaluate
```

### 测评 Claude 3.5 Sonnet

```bash
python -m lcb_runner.runner.main \
    --model claude-3.5-sonnet-openrouter \
    --custom_endpoint https://openrouter.ai/api/v1 \
    --custom_api_key sk-or-v1-xxxxx \
    --custom_model_name anthropic/claude-3.5-sonnet \
    --scenario codegeneration \
    --evaluate
```

### 测评其他模型

```bash
# DeepSeek Coder
python -m lcb_runner.runner.main \
    --model deepseek-coder-openrouter \
    --custom_endpoint https://openrouter.ai/api/v1 \
    --custom_api_key sk-or-v1-xxxxx \
    --custom_model_name deepseek/deepseek-coder \
    --scenario codegeneration \
    --evaluate

# Qwen 2.5 Coder
python -m lcb_runner.runner.main \
    --model qwen2.5-coder-openrouter \
    --custom_endpoint https://openrouter.ai/api/v1 \
    --custom_api_key sk-or-v1-xxxxx \
    --custom_model_name qwen/qwen-2.5-coder-7b-instruct \
    --scenario codegeneration \
    --evaluate
```

## 其他 OpenAI 兼容 API

### 使用自定义 API endpoint

```bash
python -m lcb_runner.runner.main \
    --model my-custom-model \
    --custom_endpoint https://api.example.com/v1 \
    --custom_api_key your-api-key \
    --custom_model_name my-model-name \
    --scenario codegeneration \
    --evaluate
```

## 输出文件

结果会保存在 `output/` 目录下，文件名基于 `--model` 参数（或 `--custom_model_name` 的 sanitized 版本）：

```
output/
  └── {model_repr}/
      ├── codegeneration_10_0.2.json          # 生成结果
      ├── codegeneration_10_0.2_eval.json      # 评估摘要
      └── codegeneration_10_0.2_eval_all.json  # 详细评估结果
```

例如，使用 `--model gpt-4-openrouter` 和 `--custom_model_name openai/gpt-4`：
- 输出目录：`output/gpt-4-openrouter/`
- 实际使用的模型：`openai/gpt-4`

## 完整示例脚本

创建一个脚本 `run_openrouter_test.sh`：

```bash
#!/bin/bash

# 配置
ENDPOINT="https://openrouter.ai/api/v1"
API_KEY="sk-or-v1-your-key-here"
MODEL_NAME="openai/gpt-4"

# 运行测评
python -m lcb_runner.runner.main \
    --model gpt-4-openrouter \
    --custom_endpoint "$ENDPOINT" \
    --custom_api_key "$API_KEY" \
    --custom_model_name "$MODEL_NAME" \
    --scenario codegeneration \
    --n 10 \
    --temperature 0.2 \
    --evaluate \
    --release_version release_v2
```

运行：
```bash
chmod +x run_openrouter_test.sh
./run_openrouter_test.sh
```

## Windows PowerShell 示例

```powershell
# 设置环境变量
$env:OPENROUTER_ENDPOINT = "https://openrouter.ai/api/v1"
$env:OPENROUTER_API_KEY = "sk-or-v1-your-key-here"

# 运行测评
python -m lcb_runner.runner.main `
    --model gpt-4-openrouter `
    --custom_model_name openai/gpt-4 `
    --scenario codegeneration `
    --evaluate
```

## 常见问题

### Q: 如何查看可用的模型名称？

A: 访问 OpenRouter 的模型列表：https://openrouter.ai/models  
模型名称格式通常是：`provider/model-name`

### Q: API key 在哪里获取？

A: 
- OpenRouter: https://openrouter.ai/keys
- 其他服务：参考对应服务的文档

### Q: 支持哪些场景？

A: 支持所有场景：
- `codegeneration` - 代码生成
- `selfrepair` - 自我修复
- `testoutputprediction` - 测试输出预测
- `codeexecution` - 代码执行

### Q: 如何只生成不评估？

A: 去掉 `--evaluate` 参数：

```bash
python -m lcb_runner.runner.main \
    --model gpt-4-openrouter \
    --custom_endpoint https://openrouter.ai/api/v1 \
    --custom_api_key your-key \
    --custom_model_name openai/gpt-4 \
    --scenario codegeneration
```

### Q: 如何继续之前的测评？

A: 使用 `--continue_existing` 或 `--continue_existing_with_eval`：

```bash
python -m lcb_runner.runner.main \
    --model gpt-4-openrouter \
    --custom_endpoint https://openrouter.ai/api/v1 \
    --custom_api_key your-key \
    --custom_model_name openai/gpt-4 \
    --scenario codegeneration \
    --continue_existing \
    --evaluate
```

### Q: 如何调试？

A: 使用 `--debug` 参数，只处理前15个问题：

```bash
python -m lcb_runner.runner.main \
    --model gpt-4-openrouter \
    --custom_endpoint https://openrouter.ai/api/v1 \
    --custom_api_key your-key \
    --custom_model_name openai/gpt-4 \
    --scenario codegeneration \
    --debug \
    --evaluate
```

## 技术细节

- **API 兼容性**：使用 OpenAI 兼容的 API 接口
- **提示词格式**：自动使用 `OpenAIChat` 格式（system + user messages）
- **错误处理**：自动重试，支持超时和限流处理
- **多进程支持**：支持 `--multiprocess` 参数并行调用 API

## 注意事项

1. **API Key 安全**：不要将 API key 提交到代码仓库，使用环境变量或配置文件
2. **速率限制**：注意 API 的速率限制，可以调整 `--multiprocess` 参数
3. **成本控制**：OpenRouter 按使用量计费，注意控制生成数量（`--n` 参数）
4. **模型名称**：确保 `--custom_model_name` 与 API 提供商支持的模型名称完全匹配

## 故障排除

### 错误：404 Not Found

**常见原因**：

1. **Endpoint 路径错误**（最常见）
   - ❌ 错误：`https://openrouter.ai/api/v1/chat/completions`
   - ✅ 正确：`https://openrouter.ai/api/v1`
   - OpenAI SDK 会自动添加 `/chat/completions`，所以 endpoint 不应该包含这个路径

2. **模型名称不正确**
   - 检查模型名称是否在 OpenRouter 的模型列表中：https://openrouter.ai/models
   - 确保使用完整的模型名称，例如：`qwen/qwen3-coder-30b-a3b-instruct`

3. **API key 权限问题**
   - 确认 API key 有权限访问该模型
   - 某些模型可能需要特定的 API key 或订阅

**解决方案**：

```bash
# 1. 检查并修正 endpoint（移除 /chat/completions）
export OPENROUTER_ENDPOINT="https://openrouter.ai/api/v1"  # ✅ 正确

# 2. 验证模型名称
# 访问 https://openrouter.ai/models 查看可用模型

# 3. 测试 API key
curl https://openrouter.ai/api/v1/models \
  -H "Authorization: Bearer $OPENROUTER_API_KEY"
```

### 错误：API key is required

**解决方案**：确保提供了 API key
```bash
# 方式1：命令行参数
--custom_api_key your-key

# 方式2：环境变量
export OPENROUTER_API_KEY="your-key"
```

### 错误：Model not found

**解决方案**：检查模型名称是否正确，参考 OpenRouter 的模型列表

### 错误：Rate limit exceeded

**解决方案**：
1. 减少并行进程数：`--multiprocess 1`
2. 增加重试间隔（代码中已实现）
3. 检查 API 配额

### 错误：Timeout

**解决方案**：增加超时时间
```bash
--openai_timeout 180  # 默认90秒
```

## 更多信息

- OpenRouter 文档：https://openrouter.ai/docs
- LiveCodeBench 文档：参考 `README.md` 和 `测评流程文档.md`

