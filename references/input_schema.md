# 输入格式

CSV 或 Parquet 面板至少包含 `date`、`ticker`、`feature_*`、`market_*`；训练阶段额外包含 `label`。同一 `date,ticker` 只能出现一行，数值列必须有限。序列构造按日期对齐，并只使用当前日期及之前的窗口。
