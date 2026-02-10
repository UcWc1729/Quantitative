# 获取数据后如何进行预测与可视化

数据已通过爬虫或 TuShare 写入 AlphaLab 后，按以下步骤完成**预测**和**可视化**。

## 框架自带的算法（可直接选用）

| 类型 | 选项 | 说明 |
|------|------|------|
| **因子数据集** | `alpha158` | Qlib 风格 158 个基础因子（K 线形态、时序等） |
| | `alpha101` | WorldQuant Alpha101 因子集 |
| **预测模型** | `lasso` | Lasso 回归，特征稀疏、可解释 |
| | `lgb` | LightGBM 梯度提升树，需安装 lightgbm |
| | `mlp` | 多层感知机，需安装 torch |

在 `run_predict_and_visualize.py` 中通过 **DATASET_NAME**（`alpha158` / `alpha101`）和 **MODEL_NAME**（`lasso` / `lgb` / `mlp`）切换即可。

---

## 一、整体流程

```
已写入 lab 的日线数据（daily/*.parquet）
    ↓
lab.load_bar_df() 加载为 DataFrame
    ↓
Alpha158 构建特征与标签 → prepare_data() → process_data()
    ↓
模型 fit(dataset) → predict(dataset, Segment.TEST)
    ↓
得到预测值 → 组成 signal 表 (datetime, vt_symbol, signal)
    ↓
可视化：Alphalens 因子分析 + 自定义图表
```

---

## 二、预测步骤（代码顺序）

### 1. 环境与参数

```python
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import polars as pl
from pathlib import Path
from functools import partial

from vnpy.trader.constant import Interval
from vnpy.alpha import AlphaLab, Segment
from vnpy.alpha.dataset import (
    AlphaDataset,
    process_drop_na,
    process_cs_norm,
    process_fill_na,
)
from vnpy.alpha.dataset.datasets.alpha_158 import Alpha158
from vnpy.alpha.model.models.lasso_model import LassoModel
from vnpy.alpha import AlphaModel
import numpy as np
```

- **实验室路径**：与爬虫/下载脚本里 `LAB_PATH` 一致（如 `"./lab/astock_crawl"` 或 `"./lab/csi300_tushare"`）。
- **时间区间**：`start`、`end` 要在你已有数据范围内，并留出 `extended_days` 做因子计算。

### 2. 加载 K 线数据

若**没有**保存指数成分（未调用过 `save_component_data`），可从 lab 的日线目录列出所有股票：

```python
lab = AlphaLab("./lab/astock_crawl")  # 与下载时一致
start, end = "2020-01-01", "2024-12-31"
extended_days = 60
interval = Interval.DAILY

# 从已有 parquet 列出所有标的
component_symbols = [f.stem for f in Path(lab.daily_path).glob("*.parquet")]
if not component_symbols:
    raise SystemExit("lab 下没有日线数据，请先运行爬虫或 TuShare 下载脚本")

df = lab.load_bar_df(component_symbols, interval, start, end, extended_days)
```

若有指数成分，可改为：

```python
index_symbol = "000300.SSE"
component_symbols = lab.load_component_symbols(index_symbol, start, end)
```

### 3. 构建数据集（Alpha158 + 预处理）

```python
train_period = ("2020-01-01", "2022-12-31")
valid_period = ("2023-01-01", "2023-06-30")
test_period = ("2023-07-01", "2024-12-31")

dataset = Alpha158(
    df,
    train_period=train_period,
    valid_period=valid_period,
    test_period=test_period,
)

dataset.add_processor("learn", partial(process_drop_na, names=["label"]))
dataset.add_processor("learn", partial(process_cs_norm, names=["label"], method="zscore"))
dataset.add_processor("infer", partial(process_fill_na, fill_value=0))

# 无成分过滤时用 None；有成分时用 lab.load_component_filters(index_symbol, start, end)
dataset.prepare_data(filters=None, max_workers=4)
dataset.process_data()
```

### 4. 训练与预测

```python
model: AlphaModel = LassoModel(alpha=0.0005, max_iter=1000)
model.fit(dataset)

pre: np.ndarray = model.predict(dataset, Segment.TEST)
df_test = dataset.fetch_infer(Segment.TEST)
df_test = df_test.with_columns(pl.Series("signal", pre))
signal = df_test.select(["datetime", "vt_symbol", "signal"])
```

### 5. 保存模型与信号（可选）

```python
name = "my_lasso"
lab.save_dataset(name, dataset)   # 可选
lab.save_model(name, model)
lab.save_signal(name, signal)
```

---

## 三、可视化方式

### 1. Alphalens 因子/信号分析（推荐）

用框架自带的**信号表现分析**，会生成分位数收益、IC、多空收益等图表：

```python
dataset.show_signal_performance(signal)
```

- 会弹出多张 **matplotlib** 图（分位数统计、收益分析、IC 等）。
- 在 Jupyter 里直接显示；在脚本里运行时会开窗口，若需保存可在调用前设置：

```python
import matplotlib
matplotlib.use("Agg")  # 不弹窗，只保存
import matplotlib.pyplot as plt
# 再调用 show_signal_performance；若 alphalens 内部 plt.show()，可改为先保存再 show
```

### 2. 单因子表现（可选）

若想看某个特征（如 `"kmid"`）的预测能力，不跑模型也可：

```python
dataset.show_feature_performance("kmid")
```

同样会生成 Alphalens 的完整 tear sheet。

### 3. 自定义简单图表

用 **matplotlib** 或 **plotly** 做简单可视化，例如信号分布、按日聚合信号：

```python
import matplotlib.pyplot as plt

# 信号分布直方图
fig, ax = plt.subplots(1, 1, figsize=(8, 4))
ax.hist(signal["signal"], bins=80, edgecolor="black", alpha=0.7)
ax.set_xlabel("signal")
ax.set_ylabel("count")
ax.set_title("Prediction signal distribution (test set)")
plt.tight_layout()
plt.savefig("signal_dist.png", dpi=150)
plt.show()

# 按日统计：当日平均信号
daily = signal.group_by("datetime").agg(pl.col("signal").mean().alias("mean_signal"))
daily = daily.sort("datetime")
fig, ax = plt.subplots(1, 1, figsize=(10, 3))
ax.plot(daily["datetime"], daily["mean_signal"], linewidth=0.8)
ax.set_xlabel("date")
ax.set_ylabel("mean signal")
ax.set_title("Daily average prediction signal")
plt.tight_layout()
plt.savefig("signal_by_date.png", dpi=150)
plt.show()
```

---

## 四、回测与资金曲线（可选）

若要把预测信号当成「多空权重」做回测，可用 vnpy.alpha 的**回测引擎**，得到资金曲线、回撤等，再画图：

```python
from vnpy.alpha.strategy import BacktestingEngine
from vnpy.alpha.strategy.strategies.equity_demo_strategy import EquityDemoStrategy

# 需先为各合约 lab.add_contract_setting(...)
engine = BacktestingEngine(lab, ...)
setting = {}
engine.add_strategy(EquityDemoStrategy, setting, signal)
engine.run_backtest()
engine.show_result()  # 会调 plotly 画资金曲线、回撤等
```

具体参数与合约配置见 `research_workflow_lasso.ipynb` 后半部分。

---

## 五、一键脚本

目录下提供了 **`run_predict_and_visualize.py`**，在已写入 lab 的前提下可：

1. 从 `LAB_PATH` 的 daily 下自动识别股票列表；
2. 加载数据 → Alpha158 → 训练 Lasso → 预测；
3. 调用 `show_signal_performance(signal)` 做 Alphalens 可视化；
4. 画信号分布、按日均值等简单图并保存；
5. 保存模型与信号到 lab。

用法：先确保 `LAB_PATH` 与爬虫/下载脚本一致，再执行：

```bash
python run_predict_and_visualize.py
```

---

## 六、小结

| 步骤       | 作用 |
|------------|------|
| `load_bar_df` | 从 lab 读入 K 线 |
| `Alpha158` + `prepare_data` + `process_data` | 特征与标签、预处理 |
| `model.fit` / `model.predict` | 训练与预测 |
| `signal = df_test.select(["datetime","vt_symbol","signal"])` | 得到信号表 |
| `show_signal_performance(signal)` | 因子/信号分析图（Alphalens） |
| 自定义 matplotlib/plotly | 信号分布、时间序列等 |
| `BacktestingEngine` + `show_result` | 回测与资金曲线 |

数据准备好后，按上述顺序即可完成**预测 + 可视化**。
