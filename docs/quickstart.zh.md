# 运行完整 AgenQA Pipeline

[English](./quickstart.md)

公开仓库现在包含完整的 AgenQA Director--Operator--Evaluator runtime、可携式 API 配置和一篇合成示例论文 PDF。Demo 直接调用研究代码中的 graph、role prompts、KnownTree 更新、Path-Fold、solver evaluation、revision policy、artifact persistence 和 playback，不是另写的玩具 pipeline。

## 1. 安装

在仓库根目录执行：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[test]"
```

## 2. 配置 API Key

```bash
export OPENAI_API_KEY="your-key"
```

Key 只从环境变量读取。`agenqa-demo` 刻意不提供 CLI key 参数，避免密钥出现在 shell history 或进程列表中。

可选覆盖：

```bash
export AGENQA_MODEL="gpt-5-mini"
export AGENQA_API_BASE="https://api.openai.com/v1"
export AGENQA_API_STYLE="responses"
```

## 3. 运行内置示例 PDF

```bash
agenqa-demo
```

内置输入为
[`examples/papers/layered_thermal_transport_demo.pdf`](../examples/papers/layered_thermal_transport_demo.pdf)，是一篇专为本公开仓库编写的两页合成科学论文。

默认 `max_steps=3`：初始 state 为 step 0，因此最多产生两个相互依赖的 QA steps。每个 step 会发起多次模型调用，因为它真实执行 init、Director、DraftChain、Format、StepCertBuilder、Path-Fold、medium/strong solver、judge 以及必要的 revise。

## 4. 运行其他论文

```bash
agenqa-demo --pdf /absolute/path/to/paper.pdf
```

常用覆盖：

```bash
agenqa-demo \
  --pdf /absolute/path/to/paper.pdf \
  --model gpt-5-mini \
  --max-steps 3 \
  --output outputs/my-paper
```

若 OpenAI-compatible gateway 只提供 `chat/completions`：

```bash
agenqa-demo \
  --api-style chat \
  --base-url https://gateway.example.com/v1 \
  --model provider-model-name
```

## 输出产物

每次运行都会在指定 output 下创建带时间戳的独立子目录。关键产物包括：

```text
state.json                         最终 graph state 与 KnownTree memory
run_config.json                    实际解析后的运行配置
run.log                            执行日志
run_playback_*.md                  人类可读的轨迹回放
00_Prompts_Snapshot/               本次实际使用的 role prompts
round_*/                           Director 与 operator 轮次
  */edge_kqa.json                  局部验证视图
  */path_kqa.json                  solver-facing folded 视图
  */answer_contract_report.json    输出 contract 检查
  */subruns_raw/                   渲染 prompt 与角色解析结果
  */solve/                         medium/strong solver 与 judge 产物
```

## 直接使用完整 CLI

Quickstart wrapper 等价于：

```bash
agenqa --config config/agent_openai.yaml agent-run \
  --paper-path examples/papers/layered_thermal_transport_demo.pdf \
  --output outputs/manual \
  --max-steps 3 \
  --max-rounds 6 \
  --lang en
```

## PDF 说明

默认 portable config 使用 `pypdf` 读取 PDF 文本层。没有可用文本层的扫描 PDF 需要在 `init.source.pdf_extract` 中另行配置 OCR/layout parser；默认安装不会下载这些重型模型。
