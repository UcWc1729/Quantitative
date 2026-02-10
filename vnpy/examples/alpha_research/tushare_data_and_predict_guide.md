# 用 TuShare 获取数据并完成预测的详细流程

本文说明：**仅用 TuShare 拉取 A 股数据 → 写入 AlphaLab → 构建因子数据集 → 训练模型 → 得到预测**的完整步骤（不涉及交易）。

---

## 一、环境与依赖

```bash
# 安装 vnpy（含 alpha 可选依赖）
pip install "vnpy[alpha]"

# TuShare（需在 https://tushare.pro 注册并获取 token）
pip install tushare
```

在 [TuShare Pro](https://tushare.pro) 注册后，在个人中心获取 **token**。日线接口一般需一定积分（如 120 分起），具体见官网说明。

---

## 二、整体流程概览

```
TuShare 拉取行情
    ↓
转成 BarData → lab.save_bar_data() 写入 lab/daily/
    ↓
（可选）指数成分 → lab.save_component_data()
    ↓
lab.load_bar_df() 读出 → 得到 pl.DataFrame (datetime, open, high, low, close, volume, turnover, vwap, vt_symbol)
    ↓
Alpha158(df, train/valid/test_period) + 处理器
    ↓
dataset.prepare_data() → dataset.process_data()
    ↓
LassoModel / LgbModel / MlpModel：fit(dataset) → predict(dataset, segment)
    ↓
得到预测值，可存为 signal 或自行分析
```

---

## 三、步骤 1：用 TuShare 拉数据并写入 AlphaLab

### 3.1 代码与数据格式约定

- TuShare 的 `ts_code` 格式：`000001.SZ`（深交所）、`600000.SH`（上交所）。
- VeighNa 的 `vt_symbol` 格式：`000001.SZSE`、`600000.SSE`。
- 转换规则：`SH` → `SSE`，`SZ` → `SZSE`；且 BarData 需要 `symbol`（仅代码，如 `000001`）和 `exchange`（`Exchange.SSE` / `Exchange.SZSE`）。

AlphaLab 按 **日线** 存到 `lab_path/daily/{vt_symbol}.parquet`，每列：`datetime`, `open`, `high`, `low`, `close`, `volume`, `turnover`, `open_interest`。  
TuShare 日线一般有：`trade_date`, `open`, `high`, `low`, `close`, `vol`, `amount`。需把 `vol`→`volume`，`amount`→`turnover`，无持仓则 `open_interest=0`。

### 3.2 示例：批量下载股票日线并保存

```python
from datetime import datetime
from pathlib import Path

import tushare as ts
from tqdm import tqdm

from vnpy.trader.constant import Exchange, Interval
from vnpy.trader.object import BarData
from vnpy.alpha import AlphaLab

# 1）初始化 TuShare（替换为你的 token）
ts.set_token("你的tushare_token")
pro = ts.pro_api()

# 2）创建实验室目录
lab = AlphaLab("./lab/csi300_tushare")

# 3）ts_code -> (symbol, exchange)
def ts_code_to_vn(ts_code: str):
    code, market = ts_code.split(".")
    if market == "SH":
        return code, Exchange.SSE
    return code, Exchange.SZSE

# 4）获取要下载的股票列表（示例：沪深300成分，或自定列表）
# 方法A：从 TuShare 指数成分接口取（需权限）
# df_idx = pro.index_weight(index_code="399300.SZ", start_date="20200101", end_date="20241231")
# symbols = df_idx["con_code"].unique().tolist()

# 方法B：自定义股票列表（ts_code 格式）
symbols = ["000001.SZ", "600000.SH"]  # 示例

start_date = "20200101"
end_date = "20241231"

for ts_code in tqdm(symbols):
    try:
        df = pro.daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
    except Exception as e:
        print(f"skip {ts_code}: {e}")
        continue
    if df is None or df.empty:
        continue

    symbol, exchange = ts_code_to_vn(ts_code)
    bars = []
    for _, row in df.iterrows():
        dt = datetime.strptime(str(row["trade_date"]), "%Y%m%d")
        bar = BarData(
            symbol=symbol,
            exchange=exchange,
            datetime=dt,
            interval=Interval.DAILY,
            open_price=float(row["open"]),
            high_price=float(row["high"]),
            low_price=float(row["low"]),
            close_price=float(row["close"]),
            volume=float(row.get("vol", row.get("volume", 0))),
            turnover=float(row.get("amount", row.get("turnover", 0))),
            open_interest=0,
            gateway_name="TUSHARE",
        )
        bars.append(bar)
    if bars:
        lab.save_bar_data(bars)
```

若使用 **指数成分** 并希望与官方 RQ/XT 示例一致地做截面回测，需要把「每个交易日 → 成分股列表」存成 `lab.save_component_data(index_symbol, index_components)`，其中 `index_components` 的 key 为 `"YYYY-MM-DD"`，value 为 `vt_symbol` 列表（如 `["000001.SZSE", "600000.SSE"]`）。可从 TuShare 的指数成分接口按日取成分，再转成 `vt_symbol` 后按日填入。

---

## 四、步骤 2：准备成分股与区间（若做截面策略）

若不做指数成分过滤，可跳过成分股，直接用「已保存的股票列表」作为 `component_symbols`。

```python
# 若已用 save_component_data 保存了指数成分
index_symbol = "000300.SSE"
start, end = "2020-01-01", "2024-12-31"
component_symbols = lab.load_component_symbols(index_symbol, start, end)
```

若没有成分数据，可从 `lab.daily_path` 下已有 parquet 文件列出所有 `vt_symbol`，作为 `component_symbols`：

```python
component_symbols = [f.stem for f in Path(lab.daily_path).glob("*.parquet")]
```

---

## 五、步骤 3：加载 K 线为 DataFrame

```python
import polars as pl
from vnpy.trader.constant import Interval

interval = Interval.DAILY
start = "2020-01-01"
end = "2024-12-31"
extended_days = 60   # 前延后延，用于计算因子

df: pl.DataFrame = lab.load_bar_df(
    component_symbols, interval, start, end, extended_days
)
```

`df` 将包含：`datetime`, `open`, `high`, `low`, `close`, `volume`, `turnover`, `open_interest`, `vwap`, `vt_symbol`。  
若某只股票没有对应 parquet，会被跳过并在日志中报错。

---

## 六、步骤 4：构建数据集（Alpha158 + 处理器）

```python
from functools import partial
from vnpy.alpha.dataset import (
    AlphaDataset,
    process_drop_na,
    process_cs_norm,
    process_fill_na,
)
from vnpy.alpha.dataset.datasets.alpha_158 import Alpha158

train_period = ("2020-01-01", "2022-12-31")
valid_period = ("2023-01-01", "2023-06-30")
test_period = ("2023-07-01", "2024-12-31")

dataset: AlphaDataset = Alpha158(
    df,
    train_period=train_period,
    valid_period=valid_period,
    test_period=test_period,
)

# 预处理：训练时丢弃 label 缺失、截面标准化 label；推理时缺失填 0
dataset.add_processor("learn", partial(process_drop_na, names=["label"]))
dataset.add_processor("learn", partial(process_cs_norm, names=["label"], method="zscore"))
dataset.add_processor("infer", partial(process_fill_na, fill_value=0))
```

若使用指数成分过滤（与 RQ/XT 示例一致）：

```python
filters = lab.load_component_filters(index_symbol, start, end)
dataset.prepare_data(filters=filters, max_workers=4)
```

若不使用成分过滤：

```python
dataset.prepare_data(filters=None, max_workers=4)
```

然后执行：

```python
dataset.process_data()
```

---

## 七、步骤 5：训练模型并预测

```python
from vnpy.alpha import AlphaModel, Segment
from vnpy.alpha.model.models.lasso_model import LassoModel

model: AlphaModel = LassoModel(alpha=0.0005, max_iter=1000)
model.fit(dataset)

# 在测试段上预测
pred: np.ndarray = model.predict(dataset, Segment.TEST)
```

若需要和 `datetime / vt_symbol` 对齐，可从 `dataset.fetch_infer(Segment.TEST)` 取同样顺序的索引：

```python
import numpy as np
df_test = dataset.fetch_infer(Segment.TEST)
df_test = df_test.with_columns(pl.Series("signal", pred))
# 或保存
lab.save_signal("tushare_lasso_test", df_test)
```

至此已完成「TuShare 数据 → 预测」全流程，无需任何交易或网关。

---

## 八、可选：用 datafeed 对接 vnpy_tushare

若已安装 **vnpy_tushare**，可在 `vt_setting.json` 中配置：

```json
{
  "datafeed.name": "tushare",
  "datafeed.username": "你的token或用户名",
  "datafeed.password": ""
}
```

之后在代码中通过 `get_datafeed()` 获取 datafeed，用 `HistoryRequest` + `query_bar_history` 拉 K 线，再 `lab.save_bar_data(bars)`，与官方 RQ/XT 示例用法一致。数据格式与上面一致，只是数据来源从「直接调 TuShare」改为「通过 VeighNa datafeed」。

---

## 九、流程小结

| 步骤 | 内容 |
|------|------|
| 1 | TuShare 拉日线 → 转 BarData → `lab.save_bar_data(bars)`（及可选成分 `lab.save_component_data`） |
| 2 | 确定 `component_symbols`（成分或目录下 parquet 列表） |
| 3 | `lab.load_bar_df(...)` 得到 `df` |
| 4 | `Alpha158(df, train/valid/test) + 处理器` → `prepare_data()` → `process_data()` |
| 5 | `LassoModel`（或 LGB/MLP）`fit(dataset)` → `predict(dataset, Segment.TEST)` → 得到预测 |

按上述顺序执行即可实现「仅用 TuShare 获取数据并完成预测」，不涉及期货与交易。
