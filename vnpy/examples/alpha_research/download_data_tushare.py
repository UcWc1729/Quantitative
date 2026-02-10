"""
使用 TuShare 下载 A 股日线并写入 AlphaLab，供 alpha 预测流程使用。

运行前：
1. pip install tushare "vnpy[alpha]"
2. 在 https://tushare.pro 注册并获取 token，替换下方 TUSHARE_TOKEN
3. 可选：修改 STOCK_LIST、START_DATE、END_DATE、LAB_PATH
"""
from datetime import datetime
from pathlib import Path

from tqdm import tqdm

from vnpy.trader.constant import Exchange, Interval
from vnpy.trader.object import BarData
from vnpy.alpha import AlphaLab, logger

# 在 https://tushare.pro 获取 token 后替换
TUSHARE_TOKEN = "你的tushare_token"
# 实验室目录（与后续 load_bar_df 一致）
LAB_PATH = "./lab/csi300_tushare"
# 下载区间
START_DATE = "20200101"
END_DATE = "20241231"
# 示例：少量股票（可改为从指数成分接口获取更多）
STOCK_LIST = [
    "000001.SZ", "000002.SZ", "600000.SH", "600519.SH",
]


def ts_code_to_vn(ts_code: str) -> tuple[str, Exchange]:
    """TuShare ts_code -> (symbol, exchange)."""
    code, market = ts_code.split(".")
    if market == "SH":
        return code, Exchange.SSE
    return code, Exchange.SZSE


def main() -> None:
    try:
        import tushare as ts
        ts.set_token(TUSHARE_TOKEN)
        pro = ts.pro_api()
    except Exception as e:
        logger.error(f"TuShare 初始化失败: {e}，请安装 tushare 并设置正确 token")
        return

    lab = AlphaLab(LAB_PATH)

    for ts_code in tqdm(STOCK_LIST, desc="下载日线"):
        try:
            df = pro.daily(ts_code=ts_code, start_date=START_DATE, end_date=END_DATE)
        except Exception as e:
            logger.warning(f"跳过 {ts_code}: {e}")
            continue
        if df is None or df.empty:
            continue

        symbol, exchange = ts_code_to_vn(ts_code)
        bars: list[BarData] = []
        for _, row in df.iterrows():
            dt = datetime.strptime(str(row["trade_date"]), "%Y%m%d")
            vol = float(row.get("vol", row.get("volume", 0)))
            amount = float(row.get("amount", row.get("turnover", 0)))
            bar = BarData(
                symbol=symbol,
                exchange=exchange,
                datetime=dt,
                interval=Interval.DAILY,
                open_price=float(row["open"]),
                high_price=float(row["high"]),
                low_price=float(row["low"]),
                close_price=float(row["close"]),
                volume=vol,
                turnover=amount,
                open_interest=0,
                gateway_name="TUSHARE",
            )
            bars.append(bar)
        if bars:
            lab.save_bar_data(bars)

    logger.info(f"数据已写入 {LAB_PATH}，日线文件在 {lab.daily_path}")


if __name__ == "__main__":
    main()
