"""
A 股日线爬虫（满足 VeighNa alpha 预测需求）。

数据来源：东方财富网公开 K 线接口（无需注册、无需 token，可直接爬取）。
说明：同花顺官方数据需在 quantapi.10jqka.com.cn 申请 API；本脚本使用东方财富接口，
     数据格式与 alpha 预测流程完全兼容，可直接 lab.load_bar_df → Alpha158 → 预测。

运行前：pip install requests "vnpy[alpha]"
可修改：LAB_PATH、STOCK_LIST、START_DATE、END_DATE、REQUEST_DELAY。
"""
from __future__ import annotations

import time
from datetime import datetime
from urllib.parse import urlencode

import requests
from tqdm import tqdm

from vnpy.trader.constant import Exchange, Interval
from vnpy.trader.object import BarData
from vnpy.alpha import AlphaLab, logger

# 实验室目录（与后续 load_bar_df 一致）
LAB_PATH = "./lab/astock_crawl"
# 下载区间（东方财富接口按“结束日期 + 条数”拉取，这里用 end 表示截止日）
START_DATE = "2020-01-01"
END_DATE = "2025-12-31"
# 请求间隔（秒），避免请求过快
REQUEST_DELAY = 0.2

# 示例股票列表：沪市 1.xxxxxx，深市 0.xxxxxx（与东方财富 secid 一致）
# 也可只写 ts_code，脚本会转成 secid
STOCK_LIST = [
    "000001.SZ",
    "000002.SZ",
    "600000.SH",
    "600580.SH",
]

# 东方财富 K 线接口（日线）
KLINE_URL = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
# 请求头，模拟浏览器
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Referer": "https://quote.eastmoney.com/",
}


def ts_code_to_secid(ts_code: str) -> str:
    """TuShare/通用代码 000001.SZ / 600000.SH -> 东方财富 secid 0.000001 / 1.600000"""
    code, market = ts_code.upper().split(".")
    if market == "SH":
        return f"1.{code}"
    return f"0.{code}"


def ts_code_to_vn(ts_code: str) -> tuple[str, Exchange]:
    """ts_code -> (symbol, exchange) for BarData."""
    code, market = ts_code.upper().split(".")
    if market == "SH":
        return code, Exchange.SSE
    return code, Exchange.SZSE


def fetch_kline_eastmoney(
    secid: str,
    end_date: str = "20500101",
    lmt: int = 10000,
    fqt: int = 0,
) -> list[dict] | None:
    """
    从东方财富拉取日 K 线。
    secid: 0.000001（深）或 1.600000（沪）
    end_date: 结束日期 YYYYMMDD，通常写大一点即可
    lmt: 最多返回条数
    fqt: 0 不复权 1 前复权 2 后复权
    返回: [{"date","open","close","high","low","volume","turnover"}, ...]，失败返回 None
    """
    params = {
        "fields1": "f1,f3",
        "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
        "klt": 101,
        "fqt": fqt,
        "secid": secid,
        "end": end_date,
        "lmt": lmt,
        "iscca": 0,
    }
    url = f"{KLINE_URL}?{urlencode(params)}"
    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
        r.raise_for_status()
        j = r.json()
    except Exception as e:
        logger.warning(f"请求失败 secid={secid} err={e}")
        return None

    data = j.get("data")
    if not data:
        return None
    klines = data.get("klines")
    if not klines:
        return None

    # klines 每条为字符串: "2024-01-02,10.5,10.8,11,10.4,1000000,10500000,..."
    # 对应 f51,f52,f53,f54,f55,f56,f57,... 即 日期,开,收,高,低,量,额,...
    rows = []
    for s in klines:
        parts = s.split(",")
        if len(parts) < 7:
            continue
        rows.append({
            "date": parts[0],
            "open": float(parts[1]),
            "close": float(parts[2]),
            "high": float(parts[3]),
            "low": float(parts[4]),
            "volume": float(parts[5]),
            "turnover": float(parts[6]),
        })
    return rows


def filter_rows_by_date(
    rows: list[dict],
    start_date: str,
    end_date: str,
) -> list[dict]:
    """只保留 [start_date, end_date] 内的数据。日期格式 YYYY-MM-DD。"""
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    out = []
    for r in rows:
        dt = datetime.strptime(r["date"], "%Y-%m-%d")
        if start <= dt <= end:
            out.append(r)
    return out


def main() -> None:
    lab = AlphaLab(LAB_PATH)
    end_ymd = END_DATE.replace("-", "")

    for ts_code in tqdm(STOCK_LIST, desc="爬取日线"):
        secid = ts_code_to_secid(ts_code)
        rows = fetch_kline_eastmoney(secid=secid, end_date=end_ymd, lmt=10000)
        if not rows:
            logger.warning(f"无数据: {ts_code} ({secid})")
            continue

        rows = filter_rows_by_date(rows, START_DATE, END_DATE)
        if not rows:
            continue

        symbol, exchange = ts_code_to_vn(ts_code)
        bars: list[BarData] = []
        for r in rows:
            dt = datetime.strptime(r["date"], "%Y-%m-%d")
            bar = BarData(
                symbol=symbol,
                exchange=exchange,
                datetime=dt,
                interval=Interval.DAILY,
                open_price=r["open"],
                high_price=r["high"],
                low_price=r["low"],
                close_price=r["close"],
                volume=r["volume"],
                turnover=r["turnover"],
                open_interest=0,
                gateway_name="EASTMONEY",
            )
            bars.append(bar)
        lab.save_bar_data(bars)
        time.sleep(REQUEST_DELAY)

    logger.info(f"数据已写入 {LAB_PATH}，日线文件在 {lab.daily_path}")


if __name__ == "__main__":
    main()
