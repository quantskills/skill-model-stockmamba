---
name: model-stockmamba
description: Use when training a standalone StockMamba cross-sectional stock model on a supplied factor sequence and market context panel, then exporting portable stock prediction signals.
quantSkills:
  schema_version: 2.0.0
  organization: quantskills
  organization_url: https://github.com/quantskills
  repository: skill-model-stockmamba
  repository_url: https://github.com/quantskills/skill-model-stockmamba
  project_type: skill
  license: GPL-3.0-only
  maintainer: X-Tech-group
  collection: research-models
  catalog: {category: "06", subcategory: 06.statistical-ml-models}
  workflow: {primary_stage: modeling, workflow_stages: [modeling]}
  tags: [model-stockmamba, factor-model, state-space, attention, signal-export]
  requires: []
  summary_zh: 使用 StockMamba 风格状态空间与注意力模型，将多因子序列合成为股票横截面预测信号。
  summary_en: Trains a StockMamba-style state-space and attention model on factor sequences and exports stock signals.
  status: active
  validation_level: runnable
  maintainer_type: community
  platforms: [cursor, claude-code, codex, hermes, openclaw]
  interface:
    mode: structured
    envelope: {name: quantskills-envelope, version: 1.0.0}
    inputs:
      - {profile: factor-panel, version_range: ">=1.0.0 <2.0.0", required: true}
    outputs:
      - {profile: factor-signal, version: 1.0.0}
      - {profile: model-run-manifest, version: 1.0.0}
    adapters: [factor-model-training, factor-signal-export]
---

```json qsh-form
{"version":1,"task":{"placeholder":"提供因子序列、市场上下文、标签、时间切分与输出目录","required":true},"fields":[{"key":"run_mode","label":"运行模式","type":"select","options":[{"value":"train","label":"训练"},{"value":"predict","label":"导出信号"}]}],"prompt_template":"{{task}}\\n使用 StockMamba 执行 {{run_mode}}，先校验输入和时间切分，再检查输出清单。"}
```

# StockMamba 多因子模型

本 Skill 提供 StockMamba 风格的独立社区实现：市场状态扫描、因子门控、时间注意力、跨股票注意力、时间蒸馏和排序感知损失。用户提供自己的因子序列、市场上下文和标签。

推理阶段不读取标签，序列窗口只能使用当前日期及之前的数据。输出为 date、ticker、prediction 信号。模型不包含因子挖掘、组合优化或具体市场数据。

本项目是非官方社区实现，不声称复现论文数值结果，仅供量化研究和教育用途。
