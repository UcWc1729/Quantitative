"""
获取数据后的预测与可视化一键脚本。

前提：已用爬虫或 TuShare 把 A 股日线写入 AlphaLab（lab_path/daily/*.parquet）。
本脚本使用框架自带的因子数据集与模型，完成：加载数据 → 特征 → 训练 → 预测 → 可视化。

框架自带算法：
- 因子数据集：alpha158（Qlib 158 因子）、alpha101（WorldQuant 101 因子）
- 预测模型：lasso、lgb（LightGBM）、mlp（多层感知机）

运行：pip install "vnpy[alpha]" 后执行 python run_predict_and_visualize.py
可修改下方 LAB_PATH、时间区间、DATASET_NAME、MODEL_NAME 及模型参数。
"""
from __future__ import annotations

import os
import subprocess
import sys
import warnings
from pathlib import Path
from functools import partial

import numpy as np
import polars as pl

warnings.filterwarnings("ignore", category=FutureWarning)

from vnpy.trader.constant import Interval
from vnpy.alpha import AlphaLab, Segment, AlphaModel
from vnpy.alpha.dataset import AlphaDataset, process_drop_na, process_cs_norm, process_fill_na
from vnpy.alpha.dataset.datasets.alpha_158 import Alpha158
from vnpy.alpha.dataset.datasets.alpha_101 import Alpha101
from vnpy.alpha.model.models.lasso_model import LassoModel
from vnpy.alpha.model.models.lgb_model import LgbModel
from vnpy.alpha.model.models.mlp_model import MlpModel
from vnpy.alpha import logger

# 与爬虫/下载脚本里的 LAB_PATH 一致
LAB_PATH = "./lab/astock_crawl"
# 时间区间（需在已有数据范围内）
START = "2020-01-01"
END = "2024-12-31"
EXTENDED_DAYS = 60
TRAIN_PERIOD = ("2020-01-01", "2022-12-31")
VALID_PERIOD = ("2023-01-01", "2023-06-30")
TEST_PERIOD = ("2023-07-01", "2024-12-31")
# 框架自带因子数据集：alpha158 | alpha101
DATASET_NAME = "alpha158"
# 框架自带模型：lasso | lgb | mlp
MODEL_NAME = "lasso"
# 任务名，用于保存 model/signal
TASK_NAME = "predict_demo"
# 简单图保存目录（当前目录下）
PLOT_DIR = Path("./predict_plots")
# 只画一只股票的预测：填 vt_symbol 如 "600000.SSE"；填 None 则画全部
PLOT_SINGLE_STOCK: str | None = "600580.SSE"
# 保存图后是否用系统默认程序打开（弹出显示）
OPEN_PLOTS_AFTER_SAVE = True


def main() -> None:
    lab = AlphaLab(LAB_PATH)
    daily_path = Path(lab.daily_path)
    component_symbols = [f.stem for f in daily_path.glob("*.parquet")]
    if not component_symbols:
        logger.error(f"未找到日线数据，请先在 {LAB_PATH} 下运行爬虫或下载脚本")
        return

    logger.info(f"共 {len(component_symbols)} 只标的，加载 K 线...")
    df = lab.load_bar_df(
        component_symbols,
        Interval.DAILY,
        START,
        END,
        EXTENDED_DAYS,
    )
    if df is None or df.is_empty():
        logger.error("load_bar_df 为空")
        return

    # 使用框架自带的因子数据集
    dataset_cls = Alpha158 if DATASET_NAME.lower() == "alpha158" else Alpha101
    logger.info(f"构建因子数据集: {DATASET_NAME}...")
    dataset: AlphaDataset = dataset_cls(
        df,
        train_period=TRAIN_PERIOD,
        valid_period=VALID_PERIOD,
        test_period=TEST_PERIOD,
    )
    dataset.add_processor("learn", partial(process_drop_na, names=["label"]))
    dataset.add_processor("learn", partial(process_cs_norm, names=["label"], method="zscore"))
    dataset.add_processor("infer", partial(process_fill_na, fill_value=0))
    dataset.prepare_data(filters=None, max_workers=4)
    dataset.process_data()

    # 使用框架自带的模型
    model: AlphaModel
    if MODEL_NAME.lower() == "lasso":
        model = LassoModel(alpha=0.0005, max_iter=1000)
    elif MODEL_NAME.lower() == "lgb":
        model = LgbModel(
            learning_rate=0.1,
            num_leaves=31,
            num_boost_round=1000,
            early_stopping_rounds=50,
        )
    elif MODEL_NAME.lower() == "mlp":
        df_train = dataset.fetch_learn(Segment.TRAIN)
        input_size = len(df_train.columns) - 3  # 去掉 datetime, vt_symbol, label
        model = MlpModel(
            input_size=input_size,
            hidden_sizes=(256, 128),
            lr=0.001,
            n_epochs=300,
            batch_size=2000,
            early_stop_rounds=50,
        )
    else:
        logger.error(f"未知模型: {MODEL_NAME}，请使用 lasso | lgb | mlp")
        return

    logger.info(f"训练 {MODEL_NAME} 并预测...")
    model.fit(dataset)
    pre: np.ndarray = model.predict(dataset, Segment.TEST)
    df_test = dataset.fetch_infer(Segment.TEST)
    df_test = df_test.with_columns(pl.Series("signal", pre))
    signal = df_test.select(["datetime", "vt_symbol", "signal"])

    lab.save_model(TASK_NAME, model)
    lab.save_signal(TASK_NAME, signal)
    logger.info(f"已保存模型与信号: {TASK_NAME}")

    # 若只显示一只股票：先筛出用于画图与 Alphalens 的信号
    plot_signal = signal
    if PLOT_SINGLE_STOCK is not None:
        uniq = signal["vt_symbol"].unique().to_list()
        one_stock = signal.filter(pl.col("vt_symbol") == PLOT_SINGLE_STOCK)
        if one_stock.is_empty():
            first = uniq[0] if uniq else None
            if first is not None:
                plot_signal = signal.filter(pl.col("vt_symbol") == first)
                logger.warning(f"未找到 {PLOT_SINGLE_STOCK}，改为画第一只: {first}。可用: {uniq[:10]}")
            else:
                logger.warning("无可用标的，仍用全量 signal")
        else:
            plot_signal = one_stock
            logger.info(f"仅绘制单只股票: {PLOT_SINGLE_STOCK}")

    # Alphalens 信号分析（需截面多只股票，单只时会报错故跳过）
    n_stocks = plot_signal["vt_symbol"].n_unique()
    if n_stocks >= 2:
        logger.info("生成 Alphalens 信号分析图...")
        dataset.show_signal_performance(plot_signal)
    else:
        logger.info("当前仅单只股票，跳过 Alphalens 分析（仅保存下方两张自定义图）")

    # 取测试集真实收益（label），与 signal 对齐后用于画图
    df_raw_test = dataset.fetch_raw(Segment.TEST).select(["datetime", "vt_symbol", "label"])
    plot_with_label = plot_signal.join(df_raw_test, on=["datetime", "vt_symbol"], how="inner").drop_nulls("label")
    has_label = not plot_with_label.is_empty()

    # 简单自定义图并保存
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams["font.sans-serif"] = ["SimHei"]
        # 字体全部调小一半（原约 11/13/12/10）
        plt.rcParams["font.size"] = 6
        plt.rcParams["axes.titlesize"] = 6
        plt.rcParams["axes.labelsize"] = 6
        plt.rcParams["xtick.labelsize"] = 5
        plt.rcParams["ytick.labelsize"] = 5
        # 防止文字挤在一起：留足边距、刻度与图略分开
        plt.rcParams["axes.titlepad"] = 10
        plt.rcParams["axes.labelpad"] = 6
        plt.rcParams["xtick.major.pad"] = 4
        plt.rcParams["ytick.major.pad"] = 4
    except ImportError:
        logger.warning("未安装 matplotlib，跳过自定义图保存")
        return

    def _open_plot(path: Path) -> None:
        if not OPEN_PLOTS_AFTER_SAVE or not path.exists():
            return
        path_abs = str(path.resolve())
        try:
            if sys.platform == "win32":
                os.startfile(path_abs)
            elif sys.platform == "darwin":
                subprocess.run(["open", path_abs], check=False, timeout=2)
            else:
                subprocess.run(["xdg-open", path_abs], check=False, timeout=2)
        except OSError:
            try:
                if sys.platform == "win32":
                    subprocess.run(["cmd", "/c", "start", "", path_abs], check=False, timeout=2)
                else:
                    raise
            except Exception as e:
                logger.warning(f"无法自动打开图片，请手动打开: {path_abs} ({e})")
        except Exception as e:
            logger.warning(f"无法自动打开图片，请手动打开: {path_abs} ({e})")

    # 分布图：预测值 + 真实收益（有 label 时画两个子图）
    if has_label:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        ax1.hist(plot_with_label["signal"], bins=50, edgecolor="black", alpha=0.7, color="C0")
        ax1.set_xlabel("signal (预测)")
        ax1.set_ylabel("count")
        ax2.hist(plot_with_label["label"], bins=50, edgecolor="black", alpha=0.7, color="C1")
        ax2.set_xlabel("label (真实收益)")
        ax2.set_ylabel("count")
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(8, 4))
        ax1.hist(plot_signal["signal"], bins=80, edgecolor="black", alpha=0.7)
        ax1.set_xlabel("signal")
        ax1.set_ylabel("count")
    plt.tight_layout(pad=1.2)
    path_dist = PLOT_DIR / "signal_dist.png"
    fig.savefig(path_dist, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"已保存 {path_dist}")
    _open_plot(path_dist)

    # 按日：预测 vs 真实收益（有 label 时画两条线）
    daily = plot_signal.group_by("datetime").agg(pl.col("signal").mean().alias("mean_signal")).sort("datetime")
    fig, ax = plt.subplots(1, 1, figsize=(12, 4))
    ax.plot(daily["datetime"], daily["mean_signal"], linewidth=0.8, label="预测 (signal)")
    if has_label:
        daily_label = plot_with_label.group_by("datetime").agg(pl.col("label").mean().alias("mean_label")).sort("datetime")
        ax.plot(daily_label["datetime"], daily_label["mean_label"], linewidth=0.8, label="真实收益 (label)")
        ax.legend(loc="best", fontsize=5)
    ax.set_xlabel("date")
    ax.set_ylabel("mean" + (" (预测 / 真实)" if has_label else " signal"))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    plt.tight_layout(pad=1.2)
    fig.subplots_adjust(bottom=0.18)
    path_date = PLOT_DIR / "signal_by_date.png"
    fig.savefig(path_date, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"已保存 {path_date}")
    _open_plot(path_date)


if __name__ == "__main__":
    main()
