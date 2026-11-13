# StockMamba 多因子模型

简体中文 | [English](README.en.md)

本项目提供 StockMamba 风格的独立社区实现，用于多因子股票收益预测和横截面信号导出。实现包含选择性状态空间市场扫描、市场状态因子门控、时间注意力、跨股票注意力、时间蒸馏和排序感知损失。

> 本项目是非官方社区实现。它不包含作者官方代码、私有数据或内部实验结果，也不承诺论文数值可直接复现。

## 适用场景

适合以下任务：

- 使用日频或其他规则频率的多因子序列预测未来收益；
- 将股票因子序列与市场上下文一起输入时序模型；
- 在验证集上选择 checkpoint，并输出可供其他回测工具使用的股票信号；
- 对不同历史窗口、市场特征和损失权重进行可审计实验。

本仓库不负责因子挖掘、组合优化、交易执行或实时行情接入。

## 方法概览

```text
因子序列 + 市场上下文
          |
          v
市场状态扫描器 -> 因子门控
          |
          v
时间注意力 -> 跨股票注意力 -> 时间蒸馏
          |
          v
股票横截面预测分数
```

模型结构、损失函数和论文对应关系见 [论文映射](references/paper_mapping.md) 和 [配置说明](references/configuration.md)。

## 安装

```bash
python -m pip install -e .
```

建议使用 Python 3.10 或更高版本。训练和推理依赖 PyTorch、Pandas、PyArrow、NumPy 和 PyYAML。

## 快速开始

生成合成样例并运行一轮训练、推理：

```bash
python examples/make_sample.py
python scripts/validate_inputs.py \\
  --input examples/generated/train.parquet \\
  --config examples/train.yaml
python scripts/train.py \\
  --config examples/train.yaml \\
  --input examples/generated/train.parquet \\
  --output-dir outputs/demo
python scripts/infer.py \\
  --checkpoint outputs/demo/checkpoint.pt \\
  --input examples/generated/test.parquet \\
  --output outputs/demo/signals.parquet
```

## 输入格式

训练面板为 CSV 或 Parquet，至少包含：

| 字段 | 说明 |
| --- | --- |
| `date` | 交易日期或有序时间标识 |
| `ticker` | 股票标识 |
| `feature_*` | 股票因子列，可有多个 |
| `market_*` | 市场上下文列，可有多个 |
| `label` | 训练标签，仅训练阶段需要 |

约束：

- 同一 `date,ticker` 只能有一行；
- 数值列必须是有限值；
- 模型会按日期构造历史窗口；
- 每个窗口只能使用当前日期及之前的数据；
- 输入因子和市场上下文必须在使用时点可获得。

完整字段说明见 [输入格式](references/input_schema.md)。

## 配置

核心配置包括：

- `model.lookback`：历史窗口长度；
- `model.d_model`：隐藏表示维度；
- `model.temporal_heads`、`model.stock_heads`：注意力头数；
- `model.beta`：因子门控温度；
- `loss.alpha`、`loss.gamma`：排序感知损失参数；
- `training.epochs`、`training.learning_rate`、`training.grad_clip`；
- `split.train_dates`、`split.validation_dates`：时间切分。

不要根据测试集结果反向修改配置。详细说明见 [配置说明](references/configuration.md)。

## 输出

训练目录包含：

- `checkpoint.pt`：验证集指标最优模型；
- `history.json`：训练损失和验证集指标；
- `selection_manifest.json`：checkpoint 选择记录。

推理输出为：

```text
date | ticker | prediction
```

输出格式见 [输出格式](references/output_schema.md)。

## 验证

```bash
python -m compileall -q src scripts
python -m pytest -q
node scripts/validate-qsh-form.mjs SKILL.md
```

仓库中的测试覆盖模型输出维度、损失函数和输入 schema。正式研究还应额外检查时间切分、标签成熟度、特征快照和回测执行口径。

## 研究边界

本项目只生成股票横截面预测信号，不提供收益保证，也不构成投资建议。使用者需要自行确认数据来源、点时可用性、标签定义、交易成本、股票池和回测规则。

## 许可证

GPL-3.0-only，见 [LICENSE](LICENSE)。
