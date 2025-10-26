# -*- coding: utf-8 -*-
"""
恒生科技成分股
1. 站上60日线比例
2. 等权价格指数
3. 30%以下加仓 / 70%以上减仓
-------------------------------------------------------------------------------
东财 → 新浪自动降级；起始年份可自定义；缓存/并发/交互图保留；复制即跑。
"""
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
    # 互联网与平台
    "腾讯控股": "00700",
    "阿里巴巴-SW": "09988",
    "美团-W": "03690",
    "京东集团-SW": "09618",
    "快手-W": "01024",
    "百度集团-SW": "09888",
    "哔哩哔哩-W": "09626",
    "网易-S": "09999",
    "腾讯音乐-SW": "01698",
    
    # 新能源汽车与智能汽车
    "比亚迪股份": "01211",
    "理想汽车-W": "02015",
    "小鹏汽车-W": "09868",
    "蔚来-SW": "09866",
    
    # 半导体与硬件
    "中芯国际": "00981",
    "华虹半导体": "01347",
    "ASMPacific": "00522",
    "比亚迪电子": "00285",
    "舜宇光学科技": "02382",
    "瑞声科技": "02018",
    "小米集团-W": "01810",
    "联想集团": "00992",
    
    # 软件、AI及云计算
    "金山软件": "03888",
    "金蝶国际": "00268",
    "商汤-W": "00020",
    
    # 大健康与金融科技
    "京东健康": "06618",
    "阿里健康": "00241",
    "平安好医生": "01833",
    
    # 其他新经济
    "海尔智家": "06690",
    "同程旅行": "00780",
    "万国数据-SW": "09698"
}

ANALYSIS_DAYS = 900
MAX_THREADS   = 4
BUY_THRESHOLD = 30
SELL_THRESHOLD = 70
CACHE_DIR = rf"D:\hstech_cache_{START_YEAR}"   # 🟡 按年份隔离缓存
os.makedirs(CACHE_DIR, exist_ok=True)

# --------------------------------------------------
# ---------- 1. fetch_one 返回前 ----------
def fetch_one(code: str, days: int):
    cache = os.path.join(CACHE_DIR, f"{code}.pkl")
    if os.path.exists(cache):
        try:
            df = pd.read_pickle(cache)
            if len(df) >= days + 60:
                return df.iloc[-days:]
        except Exception:
            pass

    end = datetime.today().strftime("%Y-%m-%d")
    df = None

    # 1. 东财
    try:
        time.sleep(np.random.uniform(0.8, 1.5))
        df = ak.stock_hk_hist(symbol=code, period="daily",
                              start_date=START_DATE, end_date=end, adjust="qfq")
        if df is not None and not df.empty:
            print(f"[东财] {code} 成功")
    except Exception as e:
        print(f"[东财] {code} 失败：{str(e)[:60]}")

    # 2. 新浪
    if df is None or df.empty:
        try:
            time.sleep(np.random.uniform(0.8, 1.5))
            df = ak.stock_hk_daily(symbol=code, adjust="qfq")
            if df is not None and not df.empty:
                df["date"] = pd.to_datetime(df["date"])
                df = df[df["date"] >= START_DATE]
                print(f"[新浪] {code} 成功")
        except Exception as e:
            print(f"[新浪] {code} 失败：{str(e)[:60]}")

    # 彻底没拉到数据
    if df is None or df.empty or len(df) < 60:
        print(f"↓↓ 彻底 skip {code}")
        return None

    # 统一列名 → 只保留这两列 → 日期当索引
    if "日期" in df.columns:
        df = df.rename(columns={"日期": "date", "收盘": "close"})
    elif "date" not in df.columns:
        df = df.reset_index().rename(columns={"index": "date"})
    if "close" not in df.columns:
        df = df.rename(columns={"收盘": "close"})

    df = df[["date", "close"]].copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    df["ma60"] = df["close"].rolling(60).mean()
    df["above"] = df["close"] > df["ma60"]
    df.to_pickle(cache)
    return df.iloc[-days:]
# --------------------------------------------------
def load_all(stocks: dict, days: int):
    codes = list(stocks.values())
    with concurrent.futures.ThreadPoolExecutor(MAX_THREADS) as pool:
        fut2code = {pool.submit(fetch_one, c, days): c for c in codes}
        data = {}
        for f in concurrent.futures.as_completed(fut2code):
            c, df = fut2code[f], f.result()
            if df is not None:
                data[c] = df
        print(f"本次有效股票 {len(data)}/{len(codes)}")
        return data

# --------------------------------------------------
# 计算比例 & 等权指数
# --------------------------------------------------
# ---------- 2. calc_ratio_and_eqindex 里 ----------
def calc_ratio_and_eqindex(stock_data: dict):
    # 所有真实日期
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
    l1, = ax1.plot(eq.index, eq, label="恒生科技等权指数", color="crimson", lw=1.8)
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

    plt.title(f"恒生科技成分股 {START_YEAR}~：等权指数 vs 站上60日线比例")
    png = f"HSTech_{START_YEAR}_{datetime.today():%Y%m%d}.png"
    plt.tight_layout(); plt.savefig(png, dpi=300); print("已保存", png); plt.show()

# --------------------------------------------------
# 主流程
# --------------------------------------------------
if __name__ == "__main__":
    print(f"开始拉取 {START_YEAR} 年以来港股恒生科技成分股数据...")
    data = load_all(STOCKS, ANALYSIS_DAYS)
    if len(data) < 3:
        print("有效股票不足，退出"); exit()
    ratio, eq = calc_ratio_and_eqindex(data)
    print("最新比例 {:.1f}%".format(ratio.iloc[-1] * 100))
    plot_result(ratio, eq)