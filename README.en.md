# StockMamba Multi-Factor Model

[简体中文](README.md)

This repository provides an independent community implementation of a StockMamba-style model for cross-sectional stock return prediction and portable signal export.

The implementation combines a selective state-space market scanner, market-conditioned factor gates, temporal attention, cross-stock attention, temporal distillation, and a rank-aware loss.

> This is an unofficial community implementation. It does not contain the paper authors' code, private data, or internal experiment outputs, and it does not claim numerical reproduction of the paper.

## Scope

The project supports:

- Training on user-supplied multi-factor sequences and market context;
- Chronological train/validation splits and checkpoint selection;
- Configurable lookback windows, model dimensions, attention heads, and loss weights;
- Inference that exports `date,ticker,prediction` signals.

It does not provide factor mining, portfolio optimization, execution, or live market data access.

## Installation

```bash
python -m pip install -e .
```

Python 3.10 or newer is recommended.

## Quick start

```bash
python examples/make_sample.py
python scripts/validate_inputs.py --input examples/generated/train.parquet --config examples/train.yaml
python scripts/train.py --config examples/train.yaml --input examples/generated/train.parquet --output-dir outputs/demo
python scripts/infer.py --checkpoint outputs/demo/checkpoint.pt --input examples/generated/test.parquet --output outputs/demo/signals.parquet
```

## Data contract

Training data must contain `date`, `ticker`, one or more `feature_*` columns, one or more `market_*` columns, and `label`. Inference data omits `label`. Keys must be unique and numeric values must be finite.

The model constructs historical windows by date. Future features and labels must never be included in a window available at the prediction time.

See [input schema](references/input_schema.md), [configuration](references/configuration.md), and [output schema](references/output_schema.md).

## Validation

```bash
python -m compileall -q src scripts
python -m pytest -q
node scripts/validate-qsh-form.mjs SKILL.md
```

## License

GPL-3.0-only. See [LICENSE](LICENSE).
