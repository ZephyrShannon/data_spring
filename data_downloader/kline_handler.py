import datetime
import gzip
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 列名
COLUMNS = ['Timestamp', 'Volume', 'Close', 'High', 'Low', 'Open']

# 30秒K线 → 5分钟 = 10根K线
LOOKAHEAD_BARS = 10

def build_filepath(base: str, biz: str, data_type: str, market: str, dt: datetime) -> str:
    year = dt.strftime("%Y")
    month = dt.strftime("%m")
    day = dt.strftime("%d")
    hour = dt.strftime("%H")

    path = f"{biz}/{data_type}/{year}{month}/"

    filename = f"{market}-{year}{month}"

    if data_type == "candlesticks_30s":
        filename += f"{day}.csv.gz"
    elif data_type.startswith("candlesticks_"):
        filename += f".csv.gz"
    elif data_type == "deals" and biz == "spot":
        filename += ".csv.gz"
    elif data_type in ["trades", "mark_prices", "funding_applies", "funding_updates"]:
        filename += f".csv.gz"
    else:
        filename += f"{day}{hour}.csv.gz"
    return "/".join([base, path + filename])

def test_load():
    data_dir = "/Users/zephyr/codes/alpha_spring/data_spring/data"
    dt_curr = datetime.datetime(year=2023,month=3,day=1, hour=8)
    market = "BTC_USDT"
    market = "ETH_USDT"
    df, freq = load_monthly_klines(data_dir, dt_curr, market)

    df_input = df
    has_header = False,
    column_order = None
    earn_5m = generate_trading_labels(df, dt_curr, None, True, time_period_min=5)
    # plot_market_data(earn_5m, "earn_5m.html")
    plot_hourly_summary(earn=earn_5m, freq_str="5m", output_file="earn_5m_summary.html")

    earn_15m = generate_trading_labels(df, dt_curr, None, True, time_period_min=15)
    plot_market_data(earn_15m, "earn_15m.html")
    plot_hourly_summary(earn_15m, "15m", "earn_15m_summary.html")

    earn_30m = generate_trading_labels(df, dt_curr, None, True, time_period_min=30)
    plot_hourly_summary(earn_30m, "30m", "earn_30m_summary.html")

    earn_60m = generate_trading_labels(df, dt_curr, None, True, time_period_min=60)
    plot_market_data(earn_60m, "earn_60m.html")

    earn_180m = generate_trading_labels(df, dt_curr, None, True, time_period_min=180)
    plot_market_data(earn_180m, "earn_180m.html")


def generate_labels_and_plot_summary(data_dir:str, biz:str, market:str, m: datetime.datetime):
    df, freq = load_monthly_klines(data_dir, m, market)
    if df is not None:
        print(f"Data loaded for: {m.date()}")
    file_path = f"{data_dir}/{biz}/lables/{m.year}{m.month}/{market}-{m.year}{m.month}"
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    all_earns = list()
    remove_col = ['Timestamp', 'Volume', 'Close', 'High', 'Low', 'Open', 'entry_long', 'entry_short']

    for freq in [5,15,30,60,180]:
        earns = generate_trading_labels(df, m, None, True, time_period_min=freq)
        summary_file = f"{file_path}_{freq}m_sum.html"
        plot_hourly_summary(earn=earns, freq_str=f"{freq}m", output_file=summary_file)
        print(f"Plot file: {summary_file} saved")
        all_earns.append(earns)
        earns = earns.set_index("Timestamp")
        all_data_cols = [col for col in earns.columns if col not in remove_col]
        all_earns.append(earns[all_data_cols])
    merged = pd.concat(all_earns,axis=1)
    label_file = file_path+".csv"
    merged.to_csv(label_file)
    print(f"Label file: {label_file} saved")


def load_month(data_dir:str, market:str, m: datetime.datetime) -> (pd.DataFrame, int):
    if m is not None:
        file_path = build_filepath(base=data_dir, biz='spot', data_type = "candlesticks_30s", market=market, dt=m)
        freq = 30
        if not Path(file_path).exists():
            print(f"⚠️ 文件不存在: {file_path}")
            freq = 60
            file_path = build_filepath(base=data_dir, biz='spot', data_type = "candlesticks_1m", market=market, dt=m)
            if not Path(file_path).exists():
                return None, None

        with gzip.open(file_path, 'rt') as f:
            df = pd.read_csv(f, header=None, names=COLUMNS)
        return df, freq


def load_monthly_klines(data_dir: str, dt_curr: datetime.datetime, market: str) -> (pd.DataFrame,int):
    if dt_curr.month == 12:
        dt_next = datetime.datetime(year=dt_curr.year + 1, month=1, day=1)
    else:
        dt_next = datetime.datetime(year=dt_curr.year, month =dt_curr.month + 1, day=1)

    df_next, freq_next = load_month(data_dir, market, dt_next)
    if df_next is None:
        return None, None
    df_curr, freq_s = load_month(data_dir, market, dt_curr)
    if df_curr is None:
        return None, None

    if freq_next > freq_s:
        df_curr = resample_df(df_curr, freq_next)
        freq = freq_next
    elif freq_s > freq_next:
        df_next = resample_df(df_next, freq_s)
        freq = freq_s
    else:
        freq = freq_s
    df = pd.concat([df_curr, df_next], ignore_index=True)
    return df, freq


COLUMNS = ['Timestamp', 'Volume', 'Close', 'High', 'Low', 'Open']


def detect_kline_interval(df: pd.DataFrame) -> int:
    interval = int(df.iloc[1]['Timestamp'] - df.iloc[0]['Timestamp'])
    if interval not in [30, 60]:
        raise ValueError(f"Unsupported interval: {interval}s, only 30s or 60s")
    return interval

def generate_trading_labels(
    df_input: pd.DataFrame,
    start_dt: datetime.datetime,
    end_dt: datetime.datetime = None,
    has_header: bool = False,
    column_order: list = None,
    time_period_min: int = 5,
    grade_percent:float = 0.01,
) -> pd.DataFrame:
    """
    向量化生成多空交易标签
    """
    if column_order is None:
        column_order = COLUMNS

    df = df_input.copy()
    start_ts = int(start_dt.timestamp())
    if end_dt is None:
        if start_dt.month == 12:
            end_ts = int(datetime.datetime(year=start_dt.year + 1, month=1, day=1).timestamp())
        else:
            end_ts = int(datetime.datetime(year=start_dt.year, month=start_dt.month + 1, day=1).timestamp())

    # 1. 处理列名
    if not has_header:
        df.columns = column_order

    df['Timestamp'] = df['Timestamp'].astype(int)
    df = df.sort_values('Timestamp').reset_index(drop=True)

    # 2. 检测周期 & 计算前瞻K线数
    interval = detect_kline_interval(df)
    LOOKAHEAD_BARS = (time_period_min * 60) // interval  # 5分钟需要的K线数
    print(f"✅ 检测周期: {interval}s, 5分钟 = {LOOKAHEAD_BARS} 根K线")

    if len(df) < LOOKAHEAD_BARS + 1:
        raise ValueError(f"数据太短，至少需要 {LOOKAHEAD_BARS + 1} 行")

    # 3. 提前 shift 获取下一根K线的 High/Low 作为开仓价
    df['entry_long'] = df['High'].shift(-1)   # 下一根 High
    df['entry_short'] = df['Low'].shift(-1)   # 下一根 Low

    # 4. 滚动窗口：未来5分钟内的 High.max 和 Low.min
    # 注意：rolling 默认是向后看，我们用 shift 反向
    df[f'max_close_long_{time_period_min}m'] = (
        df['Low']
        .rolling(window=LOOKAHEAD_BARS, min_periods=1).max()
        .shift(-LOOKAHEAD_BARS - 1)   #
    )

    df[f'max_close_short_{time_period_min}m'] = (
        df['High']
        .rolling(window=LOOKAHEAD_BARS, min_periods=1).min()
        .shift(-LOOKAHEAD_BARS - 1)  #
    )

    df[f'min_close_short_{time_period_min}m'] = (
        df['High']
        .rolling(window=LOOKAHEAD_BARS, min_periods=1).max()
        .shift(-LOOKAHEAD_BARS - 1)
    )

    df[f'min_close_long_{time_period_min}m'] = (
        df['Low']
        .rolling(window=LOOKAHEAD_BARS, min_periods=1).min()
        .shift(-LOOKAHEAD_BARS - 1)
    )

    df[f"long_earn_{time_period_min}m"] = (df[f'max_close_long_{time_period_min}m'] - df['entry_long']) / df['entry_long']
    df[f"short_earn_{time_period_min}m"] = (df['entry_short'] - df[f'max_close_short_{time_period_min}m']) / df['entry_short']
    df[f'long_lose_{time_period_min}m'] = (df[f'min_close_long_{time_period_min}m'] - df['entry_long']) / df['entry_long']
    df[f'short_lose_{time_period_min}m'] = (df['entry_short'] - df[f'min_close_short_{time_period_min}m']) / df['entry_short']
    df[f'long_grade_{time_period_min}m'] = df[f"long_earn_{time_period_min}m"] / grade_percent
    df.loc[df[f'long_lose_{time_period_min}m'] < -0.005, f'long_grade_{time_period_min}m'] = -1
    df[f'short_grade_{time_period_min}m'] = df[f"short_earn_{time_period_min}m"] / grade_percent
    df.loc[df[f'short_lose_{time_period_min}m'] < -0.005, f'short_grade_{time_period_min}m'] = -1

    conditions = [
        (df[f'long_grade_{time_period_min}m'] > df[f'short_grade_{time_period_min}m']) & (df[f'long_grade_{time_period_min}m'] > 0),  # 条件1：做多优势且为正
        (df[f'long_grade_{time_period_min}m'] < df[f'short_grade_{time_period_min}m']) & (df[f'short_grade_{time_period_min}m'] > 0)  # 条件2：做空优势且为正
    ]
    choices = [
        df[f'long_grade_{time_period_min}m'],  # 满足条件1时取 long_earn
        -df[f'short_grade_{time_period_min}m']  # 满足条件2时取 -short_earn
    ]

    df[f'ls_choice_{time_period_min}m'] = np.select(conditions, choices, default=0)

    # 2. 计算波动性
    df[f'volat_{time_period_min}m'] = df[f'long_grade_{time_period_min}m'] + df[f'short_earn_{time_period_min}m']

    # 3. 找出 volat 最大和最小的 20%
    volat_min_20 = df[f'volat_{time_period_min}m'].quantile(0.20)
    volat_max_80 = df[f'volat_{time_period_min}m'].quantile(0.80)
    # 判断是否在极端 20% 区域
    extreme_mask = (df[f'volat_{time_period_min}m'] <= volat_min_20) | (df[f'volat_{time_period_min}m'] >= volat_max_80)
    # 4. 将极端波动性的行的 long_short_choice 设为 0
    df.loc[extreme_mask, f'ls_choice_{time_period_min}m'] = 0
    return df[(df["Timestamp"] > start_ts) & (df["Timestamp"] <= end_ts)]


def resample_df(df: pd.DataFrame, freq_sec=30):
    # 假设df是你的DataFrame，并且'Timestamp'列是datetime类型
    # 首先确保'Timestamp'列为datetime类型
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])

    # 将'Timestamp'设置为DataFrame的索引
    df.set_index('Timestamp', inplace=True)

    # 假设你想要重采样'Close'列的数据，你可以根据需要调整
    # 进行重采样，频率为30秒('30S')，并选择合适的聚合方式或者插值方式
    # 由于是从长周期到短周期，这里提供一种线性插值的方法

    # 重新采样数据，这里以线性插值为例
    df_resampled = df.resample(f'{freq_sec}S').asfreq()

    # 对重采样后的数据进行插值，此处采用线性插值
    df_interpolated = df_resampled.interpolate(method='linear')

    # 如果需要将时间戳再次变为一列而不是索引，可以reset_index
    df_interpolated.reset_index(inplace=True)
    return df_interpolated

def find_kline(base_dir:str, dt: datetime.datetime, market: str):
    k30s = build_filepath(base_dir, "spot", "candlesticks_30s", market)
    if Path(k30s).exists():
      return k30s, 30
    k1m = build_filepath(base_dir, "spot", "candlesticks_1m", market)
    if Path(k1m).exists():
        return k1m, 60
    return None, None




def plot_market_data(df, freq_str:str, output_file='chart.html'):
    """
    根据提供的市场数据 DataFrame 绘制双子图并输出为 HTML 文件。

    参数:
    - df: 包含市场数据的 DataFrame，Timestamp 列为 Unix 秒级时间戳
    - output_file: 输出的 HTML 文件路径，默认为 'chart.html'
    """

    # === 1. 处理时间列：假设 Timestamp 是 Unix 时间戳（单位：秒）
    df = df.copy()  # 避免修改原始数据
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='s')

    # 排序（确保时间顺序正确）
    df = df.sort_values('Timestamp').reset_index(drop=True)

    # === 2. 创建两个子图：上图价格，下图百分比指标
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("Price Series", "Strategy Metrics (%)"),
        shared_xaxes=True,  # 共享 X 轴（时间），便于缩放同步
        vertical_spacing=0.1,  # 子图间距
        specs=[
            [{"secondary_y": False}],  # 上图：仅主Y轴
            [{"secondary_y": False}]  # 下图：也只用一个Y轴（统一百分比）
        ]
    )

    # === 3. 上图：价格数据（左 Y 轴）
    price_lines = [
        'High', 'Low', 'entry_long', 'entry_short',
        f'max_close_long_{freq_str}', f'max_close_short_{freq_str}',
        f'min_close_short_{freq_str}', f'min_close_long_{freq_str}'
    ]

    for col in price_lines:
        if col in df.columns and df[col].notna().any():
            fig.add_trace(
                go.Scatter(
                    x=df['Timestamp'],
                    y=df[col],
                    mode='lines',
                    name=col,
                    line=dict(width=1),
                    hovertemplate=f"{col}: %{{y:.4f}}<br>%{{x}}<extra></extra>"
                ),
                row=1, col=1
            )

    # === 4. 下图：百分比指标（乘以100后显示为%）
    percent_cols = [
        f'long_earn_{freq_str}', f'short_earn_{freq_str}',
        f'long_lose_{freq_str}', f'short_lose_{freq_str}',
        f'long_grade_{freq_str}', f'short_grade_{freq_str}',
        f'ls_choice_{freq_str}', f'volat_{freq_str}'
    ]

    for col in percent_cols:
        if col in df.columns and df[col].notna().any():
            fig.add_trace(
                go.Scatter(
                    x=df['Timestamp'],
                    y=df[col] * 100,  # 转换为百分比
                    mode='lines',
                    name=col,
                    line=dict(width=1, dash='dot' if 'grade' in col or 'choice' in col else 'solid'),
                    hovertemplate=f"{col}: %{{y:.2f}}%<br>%{{x}}<extra></extra>"
                ),
                row=2, col=1
            )

    # === 5. 更新布局 ===
    fig.update_layout(
        title="Market Data and Strategy Performance",
        xaxis_title="Timestamp",
        yaxis_title="Price",
        yaxis2_title="Percentage (%)",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.3,
            xanchor="center",
            x=0.5,
            font=dict(size=10)
        ),
        hovermode="x unified",  # 悬停时统一显示同时间点所有值
        height=800,  # 更高以便显示两个子图
        template="plotly_white"
    )

    # 设置下图 Y 轴为百分比格式
    fig.update_yaxes(
        tickformat=".1f%",
        title_text="Value (%)",
        row=2, col=1
    )

    # （可选）设置上图 Y 轴精度
    fig.update_yaxes(
        tickformat=".4f",
        row=1, col=1
    )

    # === 6. 输出为 HTML 文件 ===
    fig.write_html(output_file)
    print(f"📊 图表已保存至: {output_file}")

# --- 示例调用 ---
# df = pd.read_csv('your_data.csv')  # 假设 Timestamp 是秒级时间戳
# plot_market_data(df, 'market_analysis.html')



def plot_hourly_summary(earn, freq_str, output_file, resample_rule ='h'):
    """
    生成双子图 HTML 报告，数据按1小时重采样，策略指标乘以100（除volat_1h外），并设置第二张图Y轴在左侧。

    参数:
    - df: DataFrame，包含 'Timestamp'（Unix 秒时间戳）和其他列
    - output_file: 输出的 HTML 文件路径
    """
    # === 1. 处理时间列
    df = earn.copy()
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='s')
    df.set_index('Timestamp', inplace=True)

    # === 2. 数据降频：按1小时重采样
    price_data = pd.DataFrame()
    price_data['High'] = df['High'].resample(resample_rule).max()
    price_data['Low'] = df['Low'].resample(resample_rule).min()

    strategy_data = pd.DataFrame()
    earn_col = f'long_earn_{freq_str}'
    strategy_data[earn_col] = df[earn_col].resample(resample_rule).max() * 100  # 转为百分比

    earn_col = f'short_earn_{freq_str}'
    strategy_data[earn_col] = df[earn_col].resample(resample_rule).max() * 100

    lose_col = f'long_lose_{freq_str}'
    strategy_data[lose_col] = df[lose_col].resample(resample_rule).min() * 100

    lose_col = f'short_lose_{freq_str}'
    strategy_data[lose_col] = df[lose_col].resample(resample_rule).min() * 100

    volat_col = f'volat_{freq_str}'
    strategy_data[volat_col] = df[volat_col].resample(resample_rule).max()  # 不乘100

    choice_col = f'ls_choice_{freq_str}'
    per80_lose = strategy_data[(strategy_data[f'long_lose_{freq_str}'] < -0.8) & (strategy_data[f'short_lose_{freq_str}'] < -0.8)].shape[0]
    posi_80_lose = strategy_data[
        (strategy_data[f'long_lose_{freq_str}'] < -0.8) | (strategy_data[f'short_lose_{freq_str}'] < -0.8)].shape[0]
    posi_per80_earn = strategy_data[
        (strategy_data[f'long_earn_{freq_str}'] > 0.8) | (strategy_data[f'short_earn_{freq_str}'] > 0.8)].shape[0]
    if choice_col in df.columns:
        strategy_data[f'{choice_col}_min'] = df[choice_col].resample(resample_rule).min()
        strategy_data[f'{choice_col}_max'] = df[choice_col].resample(resample_rule).max()
        strategy_data[f'{choice_col}_mean'] = df[choice_col].resample(resample_rule).mean()

    # 重置索引以便绘图
    price_data.reset_index(inplace=True)
    strategy_data.reset_index(inplace=True)

    # === 3. 创建双子图
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("Price Range (High/Low)", "Strategy Metrics (per hour)"),
        shared_xaxes=True,
        vertical_spacing=0.1,
        specs=[[{"secondary_y": False}], [{"secondary_y": False}]]
    )

    # --- 图1: High / Low ---
    fig.add_trace(
        go.Scatter(
            x=price_data['Timestamp'],
            y=price_data['High'],
            mode='lines',
            name='High',
            line=dict(width=1.5, color='green'),
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=price_data['Timestamp'],
            y=price_data['Low'],
            mode='lines',
            name='Low',
            line=dict(width=1.5, color='red')
        ),
        row=1, col=1
    )

    # --- 图2: 策略指标 ---
    for col in strategy_data.columns:
        if col == 'Timestamp':
            continue
        is_percent = 'volat_' not in col  # 除了 volat 都是百分比
        color = 'blue' if 'earn' in col else 'red' if 'lose' in col else 'orange' if 'choice' in col else 'purple'
        dash = 'solid' if 'mean' not in col else 'dot'

        fig.add_trace(
            go.Scatter(
                x=strategy_data['Timestamp'],
                y=strategy_data[col],
                mode='lines',
                name=col,
                line=dict(width=1, color=color, dash=dash),
            ),
            row=2, col=1
        )

    # === 新增：在第二张图中添加 y=0.8 和 y=-0.8 的参考虚线
    fig.add_hline(
        y=0.8,
        line_dash="dash",
        line_color="gray",
        annotation_text="80% threshold",
        annotation_position="top right",
        annotation_font_size=10,
        annotation_bgcolor="white",
        row=2, col=1  # 指定添加到第二个子图
    )

    fig.add_hline(
        y=-0.8,
        line_dash="dash",
        line_color="gray",
        annotation_text="-80% threshold",
        annotation_position="bottom right",
        annotation_font_size=10,
        annotation_bgcolor="white",
        row=2, col=1  # 指定添加到第二个子图
    )

    # === 4. 布局设置
    fig.update_layout(
        title=f"Market Data Summary(hour), possible_80%_lose_times:{posi_80_lose}, all_80%_lose: {per80_lose}, possible_80%_earn:{posi_per80_earn}",
        yaxis=dict(title="Price", tickformat=".4f"),
        yaxis2=dict(
            title="Value (%)",
            side="left",
            tickformat=".2f%" if any(['volat_' not in col for col in strategy_data.columns]) else ""
        ),
        xaxis2_title="Timestamp",
        hovermode="x unified",
        height=700,
        template="plotly_white",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5,
            font=dict(size=10)
        )
    )

    # === 5. 保存为 HTML
    fig.write_html(output_file)
    print(f"双子图已保存至: {output_file}")

def main():
    import sys
    start_dt = sys.argv[1]
    end_dt = sys.argv[2]
    biz = sys.argv[3]
    market = sys.argv[4]
    data_dir = sys.argv[5]
    start_dt = datetime.datetime.strptime(start_dt, "%Y-%m-%d")
    end_dt = datetime.datetime.strptime(end_dt, "%Y-%m-%d")
    while start_dt < end_dt:
        generate_labels_and_plot_summary(data_dir=data_dir, biz=biz, market=market, m=start_dt)
        if start_dt.month == 12:
            start_dt = datetime.datetime(year=start_dt.year, month=start_dt.month, day=1)
        else:
            start_dt = datetime.datetime(year=start_dt.year+1, month=1, day=1)


if __name__ == "__main__":
    main()
    pass
