import pandas as pd
import datetime
import numpy as np
import gzip
import os
from pathlib import Path

COLUMNS = ['timestamp', 'volume', 'close', 'high', 'low', 'open']

def get_next_month(dt_curr: datetime.datetime):
    if dt_curr.month == 12:
        return datetime.datetime(year=dt_curr.year + 1, month=1, day=1)
    else:
        return datetime.datetime(year=dt_curr.year, month =dt_curr.month + 1, day=1)

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
        filename += ".csv.gz"
    else:
        path += f"{year}{month}{day}/"
        filename += f"{day}{hour}.csv.gz"
    return "/".join([base, path + filename])


def test_load_month():
    data_dir = "/Users/zephyr/codes/alpha_spring/data_spring/data"
    market = "BTC_USDT"
    biz = "spot"
    start_date = "2024-01-01"
    end_date = "2024-01-02"
    m = datetime.datetime(year=2025,month=10,day=1)

def load_kline_month(data_dir:str, market:str, m: datetime.datetime, interval_mins=1) -> (pd.DataFrame, int):
    if m is not None:
        file_path = build_filepath(base=data_dir, biz='spot', data_type="candlesticks_1m", market=market, dt=m)
        if Path(file_path).exists():
            with gzip.open(file_path, 'rt') as f:
                df = pd.read_csv(f, header=None, names=COLUMNS)
            return generate_m_minute_klines(df, interval_mins)

        start_day = datetime.datetime(year=m.year, month=m.month,day=1)
        one_day_time = datetime.timedelta(days=1)
        end_day = get_next_month(start_day)
        all_data_in_month = list()
        while start_day < end_day:
            file_path = build_filepath(base=data_dir, biz='spot', data_type = "candlesticks_30s", market=market, dt=start_day)
            if not Path(file_path).exists():
                return None
            with gzip.open(file_path, 'rt') as f:
                df = pd.read_csv(f, header=None, names=COLUMNS)
                # print(f"candlesticks_30s {start_day}:{df.shape[0]}")
                all_data_in_month.append(df)
            start_day += one_day_time
        if all_data_in_month:
            df_30s = pd.concat(all_data_in_month)
            df1m = generate_m_minute_klines(df_30s, interval_mins)
            return df1m

def load_kline_daily(data_dir:str, market:str, m: datetime.datetime):
    file_path = build_filepath(base=data_dir, biz='spot', data_type="candlesticks_30s", market=market, dt=m)
    if Path(file_path).exists():
        return None
    with gzip.open(file_path, 'rt') as f:
        df = pd.read_csv(f, header=None, names=COLUMNS)
        # print(f"candlesticks_30s {start_day}:{df.shape[0]}")
        return generate_m_minute_klines(df, 1)

def generate_m_minute_klines(df, m_interval):
    """
    将1分钟K线数据聚合为M分钟K线数据。

    参数:
    df_1min (pd.DataFrame): 包含1分钟K线数据的DataFrame。
                               必须包含列: 'timestamp', 'open', 'high', 'low', 'close', 'volume'。
                               'timestamp' 列应为 datetime 类型。
    m_interval (int): 目标K线的时间间隔（分钟）。

    返回:
    pd.DataFrame: 包含M分钟K线数据的DataFrame，列名相同。
                  'timestamp' 列为周期结束时间。
    """

    # 确保 'timestamp' 列是 datetime 类型
    if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')

    # 将 'timestamp' 设置为索引，这对于 resample 至关重要
    df_work = df.set_index('timestamp')

    # --- 聚合逻辑 ---
    # 使用 pd.Grouper 或直接字符串 'Xmin' 进行重采样
    # 'Xmin' 是 Pandas 识别的频率字符串，X 是分钟数
    resampled_data = df_work.resample(f'{m_interval}min').agg({
        'open': 'first',  # 开盘价：周期内的第一个 open 值
        'high': 'max',  # 最高价：周期内的 max high 值
        'low': 'min',  # 最低价：周期内的 min low 值
        'close': 'last',  # 收盘价：周期内的最后一个 close 值
        'volume': 'sum'  # 成交量：周期内 volume 的总和
    })

    # 删除任何可能因聚合产生的空行（例如，在一个完整的 M 分钟周期开始之前或结束之后的部分数据）
    resampled_data['volume'] = resampled_data['volume'].fillna(0.)
    ohlc_columns = ['open', 'high', 'low', 'close']
    resampled_data[ohlc_columns] = resampled_data[ohlc_columns].ffill()

    # 将索引（时间戳）移回列
    resampled_df = resampled_data.reset_index()

    # 可选：重命名列以匹配原始名称（如果 reset_index 后列名不是 'timestamp'）
    # 这里通常不需要，因为 reset_index 默认会保留原索引名（如果有的话）
    # 如果 timestamp 列名变了，可以手动指定: resampled_df.rename(columns={'index': 'timestamp'}, inplace=True)
    resampled_df['timestamp'] = resampled_df['timestamp'].astype('datetime64[s]').astype(np.int64)
    return resampled_df
