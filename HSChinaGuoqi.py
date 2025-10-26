# -*- coding: utf-8 -*-

import akshare as ak
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import mplcursors
import matplotlib.dates as mdates
from matplotlib.dates import num2date
import warnings, os, time, concurrent.futures, traceback

warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 🟡==================== 全局变量（一行改） ====================🟡
START_YEAR = 2020            # <--- 只改这里即可
START_DATE = f"{START_YEAR}-01-01"
# 🟡============================================================🟡
STOCKS = {
    # 金融（银行、保险、券商）
    "建设银行": "00939",
    "工商银行": "01398",
    "中国银行": "03988",
    "农业银行": "01288",
    "招商银行": "03968",
    "交通银行": "03328",
    "邮储银行": "01658",
    "中国平安": "02318",
    "中国人寿": "02628",
    "中国财险": "02328",
    "中信股份": "00267",

    # 互联网与科技平台
    "腾讯控股": "00700",
    "阿里巴巴-SW": "09988",
    "美团-W": "03690",
    "小米集团-W": "01810",
    "京东集团-SW": "09618",
    "百度集团-SW": "09888",
    "网易-S": "09999",
    "快手-W": "01024",
    "联想集团": "00992",

    # 能源与资源
    "中国海洋石油": "00883",
    "中国石油股份": "00857",
    "中国石油化工股份": "00386",
    "中国神华": "01088",
    "紫金矿业": "02899",

    # 消费与制造
    "比亚迪股份": "01211",
    "吉利汽车": "00175",
    "安踏体育": "02020",
    "蒙牛乳业": "02319",
    "华润啤酒": "00291",
    "农夫山泉": "09633",
    "海底捞": "06862",
    "泡泡玛特": "09992",  # 2025年9月新纳入[5](@ref)[7](@ref)
    "舜宇光学科技": "02382",
    "申洲国际": "02313",

    # 地产与基建
    "中国海外发展": "00688",
    "华润置地": "01109",
    "新奥能源": "02688",

    # 电信与科技服务
    "中国移动": "00941",
    "中国联通": "00762",
    "中芯国际": "00981",
    "石药集团": "01093",
    "海尔智家": "06690",
    "中通快递-W": "02057",
    "京东健康": "06618",
    "新东方-S": "09901",
    "携程集团-S": "09961"
}

ANALYSIS_DAYS = 900
MAX_THREADS   = 4
BUY_THRESHOLD = 30
SELL_THRESHOLD = 70
CACHE_DIR = rf"D:\hstech_cache_{START_YEAR}"   # 🟡 按年份隔离缓存
os.makedirs(CACHE_DIR, exist_ok=True)

# --------------------------------------------------
# ---------- 1. fetch_one 返回前 ----------
# -------------- 1. fetch_one 改成“日期驱动回补” --------------
def fetch_one(code: str, days: int):
    cache = os.path.join(CACHE_DIR, f"{code}.pkl")
    hit_cache = False
    df = None

    # 读缓存
    if os.path.exists(cache):
        try:
            df = pd.read_pickle(cache)
        except Exception:
            df = None

    # 要不要补新数据
    last_cached = df.index[-1] if df is not None and len(df) else None
    need_update = (last_cached is None or
                   last_cached.date() < pd.Timestamp("today").date())

    # 缓存够长且无需更新 → 直接返回
    if df is not None and len(df) >= days + 60 and not need_update:
        hit_cache = True
        return df.iloc[-days:], hit_cache

    # 需要补的数据区间
    start_date_needed = (last_cached + pd.Timedelta(days=1)).strftime(r"%Y-%m-%d") \
                        if last_cached else START_DATE
    end = pd.Timestamp("today").strftime(r"%Y-%m-%d")
    new_df = None

    # 1. 东财
    try:
        time.sleep(np.random.uniform(0.8, 1.5))
        new_df = ak.stock_hk_hist(symbol=code, period="daily",
                                  start_date=start_date_needed,
                                  end_date=end, adjust="qfq")
        if new_df is not None and not new_df.empty:
            print(f"[东财] {code} 新数据 {len(new_df)} 根")
    except Exception as e:
        print(f"[东财] {code} 失败：{str(e)[:60]}")

    # 2. 新浪降级
    if (new_df is None or new_df.empty) and need_update:
        try:
            time.sleep(np.random.uniform(0.8, 1.5))
            new_df = ak.stock_hk_daily(symbol=code, adjust="qfq")
            if new_df is not None and not new_df.empty:
                new_df["date"] = pd.to_datetime(new_df["date"])
                new_df = new_df[new_df["date"] >= start_date_needed]
                print(f"[新浪] {code} 新数据 {len(new_df)} 根")
        except Exception as e:
            print(f"[新浪] {code} 失败：{str(e)[:60]}")

    # 合并/整理
    if new_df is not None and not new_df.empty:
        if "日期" in new_df.columns:
            new_df = new_df.rename(columns={"日期": "date", "收盘": "close"})
        elif "date" not in new_df.columns:
            new_df = new_df.reset_index().rename(columns={"index": "date"})
        if "close" not in new_df.columns:
            new_df = new_df.rename(columns={"收盘": "close"})

        new_df = new_df[["date", "close"]].copy()
        new_df["date"] = pd.to_datetime(new_df["date"])
        new_df = new_df.set_index("date").sort_index()

        df = new_df if df is None else pd.concat([df, new_new]).loc[~pd.concat([df, new_df]).index.duplicated()]

    if df is None or len(df) < 60:
        return None, False

    df["ma60"] = df["close"].rolling(60).mean()
    df["above"] = df["close"] > df["ma60"]
    df.to_pickle(cache)
    return df.iloc[-days:], hit_cache
# ------------------ 2. load_all 解包返回值 ------------------
def load_all(stocks: dict, days: int):
    codes = list(stocks.values())
    with concurrent.futures.ThreadPoolExecutor(MAX_THREADS) as pool:
        fut2code = {pool.submit(fetch_one, c, days): c for c in codes}
        data = {}
        for f in concurrent.futures.as_completed(fut2code):
            c, (df, hit) = fut2code[f], f.result()      # ⬅️ 解包 hit
            if df is not None:
                data[c] = df
                print(f"{c}  {'[缓存]' if hit else '[新获取]'}")
        print(f"本次有效股票 {len(data)}/{len(codes)}")
        return data
# --------------------------------------------------
# 计算比例 & 等权指数
# --------------------------------------------------
# ---------- 2. calc_ratio_and_eqindex 里 ----------
def calc_ratio_and_eqindex(stock_data: dict):
    # ⬅️ 这里千万别写成 df.index()
    all_dates = sorted({d for df in stock_data.values() for d in df.index})
    print('最早日期:', all_dates[0], '  最晚日期:', all_dates[-1])
    ratio, eq_idx = [], []
    for day in all_dates:
        abv, closes = 0, []
        for df in stock_data.values():
            if day in df.index:
                closes.append(df.loc[day, 'close'])
                abv += df.loc[day, 'above']
        if closes:               # 至少有一只股票有数据
            ratio.append(abv / len(closes))
            eq_idx.append(np.mean(closes))
        else:
            ratio.append(np.nan)
            eq_idx.append(np.nan)

    # 返回两条 Series，索引都是 DatetimeIndex
    dt_idx = pd.to_datetime(all_dates)
    return (pd.Series(ratio, index=dt_idx, name='above_ratio'),
            pd.Series(eq_idx, index=dt_idx, name='eq_price'))
# 画图（交互 + PNG）
# --------------------------------------------------
def plot_result(ratio: pd.Series, eq: pd.Series):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    # 1. 等权指数
    l1, = ax1.plot(eq.index, eq, label="恒生国企等权指数", color="crimson", lw=1.8)
    ax1.set_ylabel("等权指数（点）"); ax1.grid(alpha=0.3); ax1.legend(loc="upper left")

    # 2. 比例
    l2, = ax2.plot(ratio.index, ratio * 100, label="站上60日线比例", color="steelblue", lw=1.8)
    ax2.axhline(BUY_THRESHOLD, ls="--", color="green", alpha=0.8, label=f"{BUY_THRESHOLD}%")
    ax2.axhline(SELL_THRESHOLD, ls="--", color="red", alpha=0.8, label=f"{SELL_THRESHOLD}%")
    ax2.set_ylabel("比例（%）"); ax2.set_ylim(0, 100); ax2.grid(alpha=0.3); ax2.legend(loc="upper right")

    # 彩色区间
    ax2.fill_between(ratio.index, 0, BUY_THRESHOLD,
                     where=ratio * 100 < BUY_THRESHOLD, color="green", alpha=0.15)
    ax2.fill_between(ratio.index, BUY_THRESHOLD, SELL_THRESHOLD,
                     where=(ratio * 100 >= BUY_THRESHOLD) & (ratio * 100 <= SELL_THRESHOLD),
                     color="yellow", alpha=0.15)
    ax2.fill_between(ratio.index, SELL_THRESHOLD, 100,
                     where=ratio * 100 > SELL_THRESHOLD, color="red", alpha=0.15)

    # 交互光标
    for line in (l1, l2):
        mplcursors.cursor(line, hover=True).connect(
            "add", lambda sel, src=line: sel.annotation.set_text(
                f"{num2date(sel.target[0]).strftime('%Y-%m-%d')}\n{sel.target[1]:.2f}"))

    plt.title(f"恒生国企成分股 {START_YEAR}~：等权指数 vs 站上60日线比例")
    png = f"HSTech_{START_YEAR}_{datetime.today():%Y%m%d}.png"
    plt.tight_layout(); plt.savefig(png, dpi=300); print("已保存", png); plt.show()

# --------------------------------------------------
# 主流程
# --------------------------------------------------
if __name__ == "__main__":
    print(f"开始拉取 {START_YEAR} 年以来港股恒生国企成分股数据...")
    data = load_all(STOCKS, ANALYSIS_DAYS)
    if len(data) < 3:
        print("有效股票不足，退出"); exit()
    ratio, eq = calc_ratio_and_eqindex(data)
    print("最新比例 {:.1f}%".format(ratio.iloc[-1] * 100))
    plot_result(ratio, eq)