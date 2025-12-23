import datetime
import os.path
import typing_extensions
from typing import Tuple
import numpy as np
import pandas as pd
import gzip
from data_downloader.file_checker import build_filepath
import calendar
from data_downloader.data_tools import load_kline_month, get_next_month
from factor_tools import add_low_freq_factors, add_mid_freq_factors, add_hf_factors
from pathlib import Path


def get_last_month(dt_curr: datetime.datetime):
    if dt_curr.month == 1:
        return datetime.datetime(year=dt_curr.year - 1, month=12, day=1)
    else:
        return datetime.datetime(year=dt_curr.year, month=dt_curr.month - 1, day=1)


def get_data_hour(dt: datetime.datetime):
    if dt.minute == 0 and dt.second == 0:
        return dt - datetime.timedelta(hours=1)
    return dt


def is_same_hour(dt_cache: datetime.datetime, dt_curr: datetime.datetime):
    return ((dt_cache is not None) and (dt_cache.year == dt_curr.year)
            and (dt_cache.month == dt_curr.month)
            and (dt_cache.day == dt_curr.day)
            and (dt_cache.hour == dt_curr.hour))


def format_to_hours(dt: datetime.datetime) -> str:
    return f'{dt.year}-{dt.month:02d}-{dt.day:02d}:{dt.hour:02d}'


def generate_m_minute_klines(df_1min, m_interval):
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
    if not pd.api.types.is_datetime64_any_dtype(df_1min['timestamp']):
        df_1min['timestamp'] = pd.to_datetime(df_1min['timestamp'], unit='s')

    # 将 'timestamp' 设置为索引，这对于 resample 至关重要
    df_work = df_1min.set_index('timestamp')

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
    resampled_data.dropna(inplace=True)

    # 将索引（时间戳）移回列
    resampled_df = resampled_data.reset_index()

    # 可选：重命名列以匹配原始名称（如果 reset_index 后列名不是 'timestamp'）
    # 这里通常不需要，因为 reset_index 默认会保留原索引名（如果有的话）
    # 如果 timestamp 列名变了，可以手动指定: resampled_df.rename(columns={'index': 'timestamp'}, inplace=True)
    resampled_df['timestamp'] = resampled_df['timestamp'].astype('datetime64[s]').astype(np.int64)
    return resampled_df


def is_last_day_of_month(date_obj):
    """
    检查给定的 datetime.date 或 datetime.datetime 对象是否是该月的最后一天。

    Args:
        date_obj (datetime.date or datetime.datetime): 要检查的日期对象。

    Returns:
        bool: 如果是最后一天则返回 True，否则返回 False。
    """
    # 确保输入是 date 或 datetime 对象
    if not isinstance(date_obj, (datetime.date, datetime.datetime)):
        raise TypeError("输入必须是 datetime.date 或 datetime.datetime 对象")

    year = date_obj.year
    month = date_obj.month

    # 获取该月的总天数
    _, last_day_of_month = calendar.monthrange(year, month)

    # 比较日期的 'day' 部分是否等于该月的最后一天
    return date_obj.day == last_day_of_month


# 假设函数已定义
def get_hf_data(base_dir: str, data_type: str, dt: datetime.datetime, market_type: str) -> pd.DataFrame:
    """
    读取某小时的1s数据
    data_type: "feature" 或 "target"
    dt: 精确到小时（minute=0, second=0）
    返回 (3600, feature_dim)
    """
    file_path = build_filepath(base_dir, "spot", data_type, market_type, dt)
    df = pd.read_csv(file_path, sep=',', compression='gzip')
    return df


def utest():
    data_dir = "data"
    data_dir = "/Users/zephyr/codes/alpha_spring/data_spring/data"
    market = "BTC_USDT"
    start_time = datetime.datetime(year=2023, month=3, day=1, tzinfo=datetime.timezone.utc)
    end_time = datetime.datetime(year=2025, month=11, day=1, tzinfo=datetime.timezone.utc)
    biz = 'spot'
    seq_len_seconds = 3600
    td = TimeSeriesDataset(data_dir, market, start_time, end_time, seq_len_seconds)
    self = td
    idx = 1
    a = td[0]
    td[1]
    for i in range(3600):
        print(f"get {i}")
        d = td[i]


KLINE_COLUMNS = ['timestamp', 'volume', 'close', 'high', 'low', 'open']


def load_5m_kline(data_dir: str, market: str, m: datetime.datetime) -> pd.DataFrame:
    file_path = build_filepath(base=data_dir, biz='spot', data_type="candlesticks_5m", market=market, dt=m)
    if os.path.exists(file_path):
        with gzip.open(file_path, 'rt') as f:
            df = pd.read_csv(f, header=None, names=KLINE_COLUMNS)
            # df.set_index('timestamp', inplace=True)
            return df
    else:
        return load_kline_month(data_dir, market, m, 5)


import pandas as pd


def fill_missing_1min_klines(df):
    """
    补全 1 分钟 K 线中的缺失时间点

    输入:
        df: DataFrame，包含列 ['timestamp', 'open', 'high', 'low', 'close', 'volume']
              timestamp 为 int 类型，单位：秒

    输出:
        补全后的 DataFrame，按 timestamp 升序排列
    """
    # 1. 转换 timestamp 为 datetime 并设为索引
    df = df.copy()
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
    df = df.set_index('datetime').sort_index()

    # 2. 生成完整的 1 分钟时间范围
    full_range = pd.date_range(
        start=df.index.min(),
        end=df.index.max(),
        freq='1min'  # 1 minute
    )

    # 3. 重采样到完整时间网格
    df_full = df.reindex(full_range)

    # 4. 填充缺失的价格字段：用前一个 close 向前填充
    # 先确保 'close' 能被 forward fill
    df_full['close'] = df_full['close'].ffill()

    # 用 close 填充 open, high, low（因为无交易，价格不变）
    df_full['open'] = df_full['open'].fillna(df_full['close'])
    df_full['high'] = df_full['high'].fillna(df_full['close'])
    df_full['low'] = df_full['low'].fillna(df_full['close'])

    # 5. volume 缺失处填 0
    df_full['volume'] = df_full['volume'].fillna(0)

    # 6. 重新生成 timestamp 列（秒级 int）
    df_full['timestamp'] = df_full.index.astype('int64') // 10 ** 9

    # 7. 恢复原始列顺序（可选）
    df_full = df_full[['timestamp', 'volume', 'close', 'high', 'low', 'open']].reset_index(drop=True)

    return df_full


def load_monthly(data_dir: str, market: str, data_type: str, m: datetime.datetime) -> pd.DataFrame:
    file_path = build_filepath(base=data_dir, biz='spot', data_type=data_type, market=market, dt=m)
    if os.path.exists(file_path):
        with gzip.open(file_path, 'rt') as f:
            df = pd.read_csv(f, header=None, names=KLINE_COLUMNS)
            # df.set_index('timestamp', inplace=True)
            if data_type == "candlesticks_1m":
                return fill_missing_1min_klines(df)
            return df
    return load_kline_month(data_dir, market, m)




def load_hourly_data(data_dir: str, market: str, mt: datetime.datetime, data_type: str) -> pd.DataFrame:
    file_path = build_filepath(base=data_dir, biz='spot', data_type=data_type, market=market, dt=mt)
    if file_path.endswith(".gz"):
        with gzip.open(file_path, 'rt') as f:
            df = pd.read_csv(f)
    else:
        df = pd.read_csv(file_path)
    if "Timestamp" in df.columns:
        df.rename({"Timestamp": 'timestamp'}, axis=1, inplace=True)
    if 'Unnamed: 0' in df.columns:
        df.drop('Unnamed: 0', axis=1, inplace=True)

    df['timestamp'] = df['timestamp'].astype(int)
    df.set_index("timestamp", inplace=True)
    return df


def cut_data(df: pd.DataFrame, start_dt: datetime.datetime, nr: int):
    return df.loc[df.index <= int(start_dt.timestamp())][-nr:]


class HourlyCache:
    def __init__(self, data_dir: str, market: str, data_type: str, need_1hour=True):
        self.cache_data_now = None
        self.cache_data_last = None
        self.cache_data_merged = None
        self.cache_date_now: [datetime.datetime | None] = None
        self.cache_date_last: [datetime.datetime | None] = None
        self.data_dir = data_dir
        self.market = market
        self.data_type = data_type
        self.need_1hour = need_1hour

    def load_hourly_data(self, dt) -> pd.DataFrame:
        try:
            return load_hourly_data(self.data_dir, self.market, dt, self.data_type)
        except Exception as e:
            return None

    def exists(self, dt_curr: datetime.datetime):
        file_path = build_filepath(base=self.data_dir, biz='spot', data_type=self.data_type,
                                   market=self.market, dt=dt_curr)
        return os.path.exists(file_path)

    def get_hf_data(self, dt_curr: datetime.datetime):
        data_hour = get_data_hour(dt_curr)
        if is_same_hour(self.cache_date_now, data_hour):
            # print(f"[0]Return cache of date: {self.cache_date_now}")
            return self.cache_data_merged
        # 需要load新的了

        cache_data_last = self.cache_data_now
        cache_date_last = self.cache_date_now
        # 全新load
        self.cache_data_now = self.load_hourly_data(data_hour)
        # print(f"[1]Load data of hour:{data_hour}")

        if self.cache_data_now is None:
            return None
        self.cache_date_now = data_hour

        if self.need_1hour:
            last_hour = data_hour - datetime.timedelta(hours=1)
            if not is_same_hour(cache_date_last, last_hour):
                # print(f"[2]Reset data of {last_hour}")
                self.cache_data_last = self.load_hourly_data(last_hour)
                if self.cache_data_last is not None:
                    self.cache_date_last = last_hour
                else:
                    self.cache_date_last = None
            else:
                # print(f"[2]Reuse data of {last_hour}")
                self.cache_date_last = cache_date_last
                self.cache_data_last = cache_data_last
            # 不需要last
        else:
            self.cache_data_last = None

        if self.cache_data_last is not None:
            self.cache_data_merged = pd.concat([self.cache_data_last, self.cache_data_now])
        else:
            self.cache_data_merged = self.cache_data_now.copy()
        self.cache_data_merged = self.cache_data_merged.ffill()
        if 'high' in self.cache_data_now.columns:
            # print("ffill high and low")
            first_line_index = self.cache_data_now.index[0]
            first_high = self.cache_data_now.loc[first_line_index, 'high']
            if pd.isna(first_high):
                self.cache_data_now.loc[first_line_index, 'high'] = self.cache_data_merged.loc[first_line_index]['high']
                self.cache_data_now.loc[first_line_index, 'low'] = self.cache_data_merged.loc[first_line_index]['low']
        return self.cache_data_merged


class MonthlyDataCache:
    def __init__(self, data_dir: str, market: str, data_type: str):
        self.cache_data_now = None
        self.cache_data_last = None
        self.cache_data_merged = None
        self.cache_date_now: [datetime.datetime | None] = None
        self.cache_date_last: [datetime.datetime | None] = None
        self.data_dir = data_dir
        self.market = market
        self.data_type = data_type

    def load_month(self, dt) -> pd.DataFrame:
        try:
            if self.data_type == "candlesticks_5m":
                return load_5m_kline(self.data_dir, self.market, dt)
            return load_monthly(self.data_dir, self.market, self.data_type, dt)
        except Exception as e:
            return None

    def get_klines(self, dt_curr: datetime.datetime):
        if (self.cache_date_now is not None) and (self.cache_date_now.year == dt_curr.year) and (
                self.cache_date_now.month == dt_curr.month):
            return self.cache_data_merged
        # 需要load新的了

        self.cache_data_last = self.cache_data_now
        self.cache_date_last = self.cache_date_now
        # 全新load
        self.cache_data_now = self.load_month(dt_curr)
        self.cache_date_now = dt_curr

        if self.cache_data_last is None:
            last_month = get_last_month(dt_curr)
            self.cache_data_last = self.load_month(last_month)
            if self.cache_data_last is not None:
                self.cache_date_last = last_month
        if self.cache_data_last is not None:
            cache_data_merged = pd.concat([self.cache_data_last, self.cache_data_now])
        else:
            cache_data_merged = self.cache_data_now.copy()

        cache_data_merged.set_index('timestamp', inplace=True)
        self.cache_data_merged = cache_data_merged
        return self.cache_data_merged


def get_all_file_list(data_dir: str, biz: str, data_type: str, market: str, start_time: datetime.datetime,
                      end_time: datetime.datetime, interval=60):
    file_mergable = list()
    time_list = list()
    one_hour = datetime.timedelta(hours=1)
    while start_time < end_time:
        file_path = build_filepath(base=data_dir, biz=biz, data_type=data_type,
                                   market=market, dt=start_time)
        if os.path.exists(file_path):
            time_list.append(start_time)
        else:
            if len(time_list) != 0:
                if len(time_list) > 1:
                    print(f"Add new segment:[{format_to_hours(time_list[0])}-{format_to_hours(time_list[-1])}]")
                    file_mergable.append(TimeSeriesDataset(data_dir, market, time_list, interval, 3600))
                    time_list.clear()
                else:
                    print(f"drop one hour{format_to_hours(time_list[0])}")
        start_time = start_time + one_hour

    if len(time_list) != 0:
        if len(time_list) > 1:
            print(f"Add new segment:[{format_to_hours(time_list[0])}-{format_to_hours(time_list[-1])}]")
            file_mergable.append(TimeSeriesDataset(data_dir, market, time_list, interval, 3600))
            time_list.clear()
        else:
            print(f"drop one hour{format_to_hours(time_list[0])}")
    return file_mergable


def create_kline_data():
    start_time = datetime.datetime(year=2022, month=8, day=1, tzinfo=datetime.timezone.utc)
    end_time = datetime.datetime(year=2025, month=11, day=1, tzinfo=datetime.timezone.utc)
    market = "BTC_USDT"
    data_type = "candlesticks_1h"
    data_dir = "/Users/zephyr/codes/alpha_spring/data_spring/data"
    k1h = MonthlyDataCache(data_dir, market, data_type)
    data_type = "candlesticks_5m"
    k5m = MonthlyDataCache(data_dir, market, data_type)
    data_type = "candlesticks_1m"
    k1m = MonthlyDataCache(data_dir, market, data_type)
    from pathlib import Path
    price_factor = (1/50000)
    while start_time < end_time:
        next_date = get_next_month(start_time)
        data_1h = k1h.get_klines(start_time)
        factor_1h = add_low_freq_factors(data_1h, price_factor)

        factor_1h_monthly = factor_1h.loc[
            (factor_1h.index >= start_time.timestamp()) & (factor_1h.index < next_date.timestamp())]
        factor_1h_monthly.reset_index(inplace=True)
        k1h_filepath = build_filepath(data_dir, "spot", "factor_k1h", market, start_time)
        row_num = (next_date - start_time).days * 24
        assert row_num == factor_1h_monthly.shape[0]
        k1h_dir = Path(k1h_filepath).parent
        if not k1h_dir.exists():
            k1h_dir.mkdir(parents=True, exist_ok=False)
        factor_1h_monthly.to_csv(k1h_filepath, index=False, float_format='%.6f')
        start_time = next_date

    start_time = datetime.datetime(year=2023, month=3, day=1, tzinfo=datetime.timezone.utc)
    while start_time < end_time:
        next_date = get_next_month(start_time)
        data_5m_low = k5m.get_klines(start_time).copy()
        data_5m_low = add_low_freq_factors(data_5m_low, price_factor)

        factor_5m_low_monthly = data_5m_low.loc[
            (data_5m_low.index >= start_time.timestamp()) & (data_5m_low.index < next_date.timestamp())]

        row_num = (next_date - start_time).days * 24 * 12
        if row_num != factor_5m_low_monthly.shape[0]:
            print(f"{row_num} != {factor_5m_low_monthly.shape[0]} @ {start_time}")
        assert row_num == factor_5m_low_monthly.shape[0]

        k5m_filepath = build_filepath(data_dir, "spot", "factor_k5m_low", market, start_time)
        k5m_dir = Path(k5m_filepath).parent
        if not k5m_dir.exists():
            k5m_dir.mkdir(parents=True, exist_ok=False)
        factor_5m_low_monthly.reset_index().to_csv(k5m_filepath, index=False, float_format='%.6f')

        data_5m = k5m.get_klines(start_time).copy()
        factor_5m = add_mid_freq_factors(data_5m, price_factor, 5)
        price_cols = ['ma20_mid', 'ema12_mid', 'ema26_mid', 'b_up_mid', "b_lo_mid"]
        other_cols = ['bar_mid', 'b_wd_mid', 'rsi_mid','wr_mid', 'stoc_K_mid', 'stoc_D_mid']
        for name in price_cols:
            factor_5m[name] = factor_5m[name] * price_factor
        factor_5m['bar_mid'] *= (1/300)

        factor_5m_monthly = factor_5m.loc[
            (factor_5m.index >= start_time.timestamp()) & (factor_5m.index < next_date.timestamp())]
        assert factor_5m_monthly.shape[0] == row_num
        k5m_filepath = build_filepath(data_dir, "spot", "factor_k5m", market, start_time)
        k5m_dir = Path(k5m_filepath).parent
        if not k5m_dir.exists():
            k5m_dir.mkdir(parents=True, exist_ok=False)
        factor_5m_monthly.reset_index().to_csv(k5m_filepath, index=False, float_format='%.6f')

        data_1m = k1m.get_klines(start_time).copy()
        factor_1m = add_mid_freq_factors(data_1m, price_factor, 1)
        factor_1m_monthly = factor_1m.loc[
            (factor_1m.index >= start_time.timestamp()) & (factor_1m.index < next_date.timestamp())]
        k1m_filepath = build_filepath(data_dir, "spot", "factor_k1m", market, start_time)
        row_num = (next_date - start_time).days * 24 * 60
        assert factor_1m_monthly.shape[0] == row_num
        k1m_dir = Path(k1m_filepath).parent
        if not k1m_dir.exists():
            k1m_dir.mkdir(parents=True, exist_ok=False)
        factor_1m_monthly.reset_index().to_csv(k1m_filepath, index=False, float_format='%.6f')
        start_time = next_date

def find_missing_date(df: pd.DataFrame):
    dt_series = pd.to_datetime(df.index, unit='s')
    df['datetime'] = dt_series
    # 4. 生成完整的 1 分钟时间范围
    start = df['datetime'].min()
    end = df['datetime'].max()
    full_range = pd.date_range(start=start, end=end, freq='1min')  # '1T' = 1 minute

    # 5. 找出缺失的时间点
    present = df['datetime']
    missing = full_range.difference(present)

    print(f"共缺失 {len(missing)} 行")
    print("缺失的时间点（前10个）:")
    print(missing[:10])

create_kline_data()


def main():
    data_dir = "data"
    market = "BTC_USDT"
    start_time = datetime.datetime(year=2023, month=3, day=1, tzinfo=datetime.timezone.utc)
    end_time = datetime.datetime(year=2025, month=11, day=1, tzinfo=datetime.timezone.utc)
    biz = 'spot'
    data_type = "labels"
    all_list = get_all_file_list(data_dir, biz, data_type, market, start_time, end_time, 60)
    ss = SegmentSets(all_list)
    total_len = ss.__len__()
    print(f"Found {total_len} length")
    start_dt = datetime.datetime.now()
    for i in range(total_len):
        a, b = ss[i]
        print(f"Found data of len{a.shape}")
    end_dt = datetime.datetime.now()
    time_cost = end_dt - start_dt
    print(f"Cost time: {time_cost}")
