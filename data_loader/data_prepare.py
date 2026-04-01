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
    df_full['timestamp'] = df_full.index.astype('int64') #// 10 ** 9

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


def classize_labels():
    start_time = datetime.datetime(year=2023, month=3, day=1, tzinfo=datetime.timezone.utc)
    end_time = datetime.datetime(year=2025, month=11, day=1, tzinfo=datetime.timezone.utc)
    market = "BTC_USDT"
    data_type = "labels"
    data_dir = "/Users/zephyr/codes/alpha_spring/data_spring/data"

    max_all = list()
    min_all = list()
    while start_time < end_time:
        label_filepath = build_filepath(data_dir, "spot", data_type, market, start_time)
        if os.path.exists(label_filepath):
            df = pd.read_csv(label_filepath).set_index('timestamp')
            min = df.quantile(0.025)
            max = df.quantile(0.975)
            max_all.append(max)
            min_all.append(min)
        start_time += datetime.timedelta(hours=1)
    max_all = pd.DataFrame(max_all)
    min_all = pd.DataFrame(min_all)
    max_all = max_all.quantile(0.975)
    min_all = min_all.quantile(0.025)
    return max_all, min_all


import pandas as pd


def classify_dataframe_to_5_bins(df: pd.DataFrame,
                                 max_series: pd.Series) -> pd.DataFrame:
    """
    将 DataFrame 中的每一列根据给定的 min 和 max 转换为 5 个等宽类别（0~4）。

    Parameters:
    ----------
    df : pd.DataFrame
        输入的数值型 DataFrame。
    min_series : pd.Series
        每列对应的最小值，索引应与 df.columns 对齐。
    max_series : pd.Series
        每列对应的最大值，索引应与 df.columns 对齐。

    Returns:
    -------
    pd.DataFrame
        与输入 df 同 shape 的整数 DataFrame，值为 0,1,2,3,4。
        超出 [min, max] 范围的值会被 clip 到边界（即 <min → 0, >max → 4）。
    """
    # 确保 min_series 和 max_series 的索引与 df.columns 一致

    max_series = max_series.reindex(df.columns)

    # 初始化结果 DataFrame
    result = pd.DataFrame(index=df.index, columns=df.columns, dtype='int64')

    for col in df.columns:
        col_min = min_series[col]
        col_max = max_series[col]

        if col_min >= col_max:
            raise ValueError(f"Column '{col}': min ({col_min}) >= max ({col_max})")

        # 计算每个值在 [min, max] 区间中的位置（归一化到 [0, 1)）
        normalized = (df[col] - col_min) / (col_max - col_min)

        # 将超出范围的值限制在 [0, 1]
        normalized = np.clip(normalized, 0.0, 1.0)

        # 映射到 0~4 的整数（5 个 bin）
        # 注意：使用 floor( x * 5 )，但 1.0 会变成 5，所以先乘再取 min
        bins = (normalized * 5).astype(int)
        bins = np.clip(bins, 0, 4)  # 确保最大为 4

        result[col] = bins

    return result


import numpy as np
import pandas as pd


def estimate_quantile_bins_for_5_classes(
        min_val: float,
        max_val: float,
        total_samples: int,
        histogram_counts: np.ndarray,  # shape=(200,), 第一阶段统计的各桶频数
) -> np.ndarray:
    """
    基于 200 桶直方图，估算 5 等分（quintile）的边界值。

    返回 6 个边界值，对应 5 个区间：
        [bin_edges[0], bin_edges[1]) → 类别 0
        [bin_edges[1], bin_edges[2]) → 类别 1
        ...
        [bin_edges[4], bin_edges[5]] → 类别 4

    Parameters:
    ----------
    min_val : float
        全局最小值
    max_val : float
        全局最大值
    total_samples : int
        总样本数（用于计算目标分位位置）
    histogram_counts : np.ndarray of shape (200,)
        每个桶的频数（通过遍历所有文件累加得到）

    Returns:
    -------
    bin_edges : np.ndarray of shape (6,)
        5 分类的 6 个边界值（包含 min 和 max）
    """
    if min_val >= max_val:
        raise ValueError("min_val 必须 < max_val")
    if len(histogram_counts) != 200:
        raise ValueError("histogram_counts 必须长度为 200")

    n_bins = 200
    bin_width = (max_val - min_val) / n_bins
    bin_edges_full = np.linspace(min_val, max_val, n_bins + 1)  # shape (201,)

    # 计算累计频数
    cumsum_counts = np.cumsum(histogram_counts)
    total = total_samples

    # 目标分位点：20%, 40%, 60%, 80%
    target_quantiles = [0.2, 0.4, 0.6, 0.8]
    target_positions = [int(q * total) for q in target_quantiles]

    estimated_edges = [min_val]  # 起始边界

    for target_pos in target_positions:
        # 找到第一个累计频数 >= target_pos 的桶索引
        bin_idx = np.searchsorted(cumsum_counts, target_pos, side='left')

        if bin_idx >= n_bins:
            edge_val = max_val
        else:
            # 线性插值估算分位点在桶内的精确位置
            cum_before = cumsum_counts[bin_idx - 1] if bin_idx > 0 else 0
            count_in_bin = histogram_counts[bin_idx]

            if count_in_bin == 0:
                # 如果桶为空，取桶右边界
                edge_val = bin_edges_full[bin_idx + 1]
            else:
                # 插值比例
                ratio = (target_pos - cum_before) / count_in_bin
                edge_val = bin_edges_full[bin_idx] + ratio * bin_width

            # 边界保护
            edge_val = np.clip(edge_val, min_val, max_val)
        estimated_edges.append(edge_val)
    estimated_edges.append(max_val)
    return np.array(estimated_edges)


import pandas as pd
import numpy as np


def classify_signal_by_proportion(
        choice_df: pd.DataFrame,
        min_max_values: dict,
        neutral_ratio: float = 0.05,  # 中性区占单侧范围的比例（如 5%）
        strong_threshold: float = 0.35,  # 强信号起始比例（从 0 开始算，如 35%）
        return_counts: bool = True
) -> tuple[pd.DataFrame, dict[str, pd.Series]]:
    """
    按比例对信号进行 5 分类（强卖、弱卖、中性、弱买、强买），
    并可选返回每列各类别的样本数量。
    """
    if neutral_ratio >= strong_threshold:
        raise ValueError("neutral_ratio 必须 < strong_threshold，否则弱信号区间为空")

    labels_df = pd.DataFrame(index=choice_df.index, columns=choice_df.columns, dtype='int64')
    counts = {
        'strong_sell': pd.Series(0, index=choice_df.columns, dtype=int),
        'weak_sell': pd.Series(0, index=choice_df.columns, dtype=int),
        'neutral': pd.Series(0, index=choice_df.columns, dtype=int),
        'weak_buy': pd.Series(0, index=choice_df.columns, dtype=int),
        'strong_buy': pd.Series(0, index=choice_df.columns, dtype=int),
    }

    for col in choice_df.columns:
        x = choice_df[col].copy()
        col_min, col_max = min_max_values[col]

        if col_min >= 0:
            # 全为非负：无卖出信号
            col_min = min(col_min, -col_max) if col_max > 0 else -1e-8
        if col_max <= 0:
            # 全为非正：无买入信号
            col_max = max(col_max, -col_min) if col_min < 0 else 1e-8

        # 单侧范围（取绝对值）
        pos_range = col_max  # 正向最大值
        neg_range = -col_min  # 负向最大值（正值）

        # 计算阈值
        neutral_pos = neutral_ratio * pos_range
        neutral_neg = neutral_ratio * neg_range

        strong_pos = strong_threshold * pos_range
        strong_neg = strong_threshold * neg_range

        labels = np.full(len(x), 2, dtype=np.int64)  # 默认中性

        # --- 买入区域 ---
        buy_mask = x > neutral_pos
        if buy_mask.any():
            weak_buy_mask = (x > neutral_pos) & (x <= strong_pos)
            strong_buy_mask = x > strong_pos
            labels[weak_buy_mask] = 3
            labels[strong_buy_mask] = 4

        # --- 卖出区域 ---
        sell_mask = x < -neutral_neg
        if sell_mask.any():
            weak_sell_mask = (x < -neutral_neg) & (x >= -strong_neg)
            strong_sell_mask = x < -strong_neg
            labels[weak_sell_mask] = 1
            labels[strong_sell_mask] = 0

        labels_df[col] = labels

        # 统计数量
        unique, counts_ = np.unique(labels, return_counts=True)
        count_dict = dict(zip(unique, counts_))
        counts['strong_sell'][col] = count_dict.get(0, 0)
        counts['weak_sell'][col] = count_dict.get(1, 0)
        counts['neutral'][col] = count_dict.get(2, 0)
        counts['weak_buy'][col] = count_dict.get(3, 0)
        counts['strong_buy'][col] = count_dict.get(4, 0)

    if return_counts:
        return labels_df, counts
    else:
        return labels_df


# ==============================
# 主流程：你需要填充文件遍历部分
# ==============================

def compute_volatility_5_class_bins_from_files():
    """
    主函数：遍历所有文件，统计 200 桶直方图，然后估算 5 分类边界。
    Parameters:
    ----------
    min_val, max_val : float
        已知的全局最小/最大波动性
    total_samples : int
        所有文件中波动性值的总数（可预先统计）
    file_list_or_iterator : iterable
        文件路径列表或生成器（由你提供）

    Returns:
    -------
    quintile_edges : np.ndarray of shape (6,)
        5 分类的边界值
    """
    n_hist_bins = 200
    # --- 第一阶段：遍历所有文件，累加直方图 ---
    start_time = datetime.datetime(year=2023, month=3, day=1, tzinfo=datetime.timezone.utc)
    end_time = datetime.datetime(year=2025, month=11, day=1, tzinfo=datetime.timezone.utc)
    market = "BTC_USDT"
    data_type = "labels"
    data_dir = "/Users/zephyr/codes/alpha_spring/data_spring/data"

    volatility_columns = {"volat_5m":0.148492,"volat_15m":0.225434,
                          "volat_30m":0.407082,"volat_60m":0.725834,
                          "volat_180m":1.782520}

    all_bins = dict()
    for col,max_val in volatility_columns.items():
        bins = np.linspace(0, max_val, n_hist_bins + 1)
        hist_counts = np.zeros(n_hist_bins, dtype=np.int64)
        all_bins[col] = (bins,hist_counts)
    total_samples = 0
    processed_hours = 0
    while start_time < end_time:
        label_filepath = build_filepath(data_dir, "spot", data_type, market, start_time)
        if os.path.exists(label_filepath):
            df = pd.read_csv(label_filepath).set_index('timestamp')
            total_samples += 3600
            processed_hours += 1
            if processed_hours % 240 == 0:
                print(f"Processed {processed_hours}!")
            # 示例伪代码：
            # volatility_series = load_volatility_from_file(file_path)  # shape (n,)
            # 必须确保 volatility_series 是 1D 数值数组，且无 NaN

            # 将当前文件的波动性值分配到 200 个桶中
            # 注意：使用 np.digitize，bins 是 201 个边界

            for col_name, (bins,hist_counts) in all_bins.items():
                volatility_series = df[col_name].values  # ←←← 你在这里填入从 file_path 读取的数据
                # 处理边界：max_val 会落入最后一个桶
                digitized = np.digitize(volatility_series, bins, right=False)  # 返回 1~201
                # digitized == 0 → < min_val（应极少）
                # digitized == 201 → == max_val（我们归入第 200 桶）
                digitized = np.clip(digitized, 1, n_hist_bins)
                # 转为 0-based index
                indices = digitized - 1
                # 累加计数
                unique_indices, counts = np.unique(indices, return_counts=True)
                hist_counts[unique_indices] += counts
        start_time += datetime.timedelta(hours=1)

    all_cols = dict()
    for col_name, (bins,hist_counts) in all_bins.items():
         # --- 第二阶段：估算 5 分位边界 ---
        quintile_edges = estimate_quantile_bins_for_5_classes(
            min_val=0,
            max_val=volatility_columns[col_name],
            total_samples=total_samples,
            histogram_counts=hist_counts
        )
        all_cols[col_name] = quintile_edges
    return all_cols

'''
ls_choice_5m      0.603769
volat_5m          0.148492
ls_choice_15m     1.014868
volat_15m         0.225434
ls_choice_30m     1.360917
volat_30m         0.407082
ls_choice_60m     1.745529
volat_60m         0.725834
ls_choice_180m    2.463534
volat_180m        1.782520
Name: 0.975, dtype: float64
>>> b
ls_choice_5m     -0.910540
volat_5m          0.000002
ls_choice_15m    -1.301925
volat_15m         0.000027
ls_choice_30m    -1.639917
volat_30m         0.000072
ls_choice_60m    -2.056453
volat_60m         0.000131
ls_choice_180m   -3.028236
volat_180m        0.000223
'''

min_max_values = {"ls_choice_5m": (-0.910540, 0.603769),
                  "ls_choice_15m": (-1.301925, 1.014868),
                  "ls_choice_30m": (-1.639917, 1.360917),
                  "ls_choice_60m": (-2.056453, 1.745529),
                  "ls_choice_180m": (-3.028236, 2.463534)}

import numpy as np
from numpy import ndarray
cuts = {'volat_5m': np.array([0., 0.00113779, 0.00457846, 0.01006888, 0.02020491, 0.148492]),
        'volat_15m': np.array([0., 0.00300668, 0.01064053, 0.02218996, 0.04123717, 0.225434]),
        'volat_30m': np.array([0., 0.00570804, 0.02069834, 0.04286237, 0.0774941,0.407082]),
        'volat_60m': np.array([0., 0.01173768, 0.04403809, 0.08707927, 0.15255994, 0.725834]),
        'volat_180m': np.array([0., 0.04227571, 0.14395993, 0.2652156, 0.44690631, 1.78252])}

def classify_all_volatile_labels():
    start_time = datetime.datetime(year=2024, month=1, day=1, tzinfo=datetime.timezone.utc)
    end_time = datetime.datetime(year=2025, month=11, day=1, tzinfo=datetime.timezone.utc)
    market = "BTC_USDT"
    data_type = "labels"
    data_dir = "/Users/zephyr/codes/alpha_spring/data_spring/data"
    choice_counts = dict()
    last_time = start_time
    while start_time < end_time:
        label_filepath = build_filepath(data_dir, "spot", data_type, market, start_time)
        if os.path.exists(label_filepath):
            df = pd.read_csv(label_filepath).set_index('timestamp')
            break
            classified_labels = dict()
            for col, cut_bins in cuts.items():
                digitized = np.digitize(df[col].values, cut_bins, right=False)  # 返回 1~201
                # digitized == 0 → < min_val（应极少）
                # digitized == 201 → == max_val（我们归入第 200 桶）
                digitized = np.clip(digitized, 1, 5)
                digitized = digitized - 1
                classified_labels[col] = digitized
            choice_columns = ["ls_choice_5m", "ls_choice_15m", "ls_choice_30m", "ls_choice_60m", "ls_choice_180m"]
            choice_df = df[choice_columns]
            neutral_ratio = 0.05
            strong_threshold: float = 0.45
            return_counts: bool = True
            choices, counts = classify_signal_by_proportion(choice_df, min_max_values, neutral_ratio, strong_threshold, return_counts)
            #
            for choice, count in counts.items():
                existed_count = choice_counts.get(choice)
                if existed_count is None:
                    choice_counts[choice] = count
                else:
                    choice_counts[choice] = existed_count + count

            choice_columns = ["ls_choice_5m", "ls_choice_15m", "ls_choice_30m", "ls_choice_60m", "ls_choice_180m"]
            choice_df = df[choice_columns]
            neutral_ratio = 0.05
            strong_threshold: float = 0.35
            return_counts: bool = True
            choices, counts = classify_signal_by_proportion(choice_df, min_max_values, neutral_ratio, strong_threshold,
                                                            return_counts)

            vol_choice_columns = list(set(df.columns) - set(choice_columns)) # ["ls_choice_5m", "ls_choice_15m", "ls_choice_30m", "ls_choice_60m", "ls_choice_180m"]
            choice_df = df[vol_choice_columns]
            neutral_ratio = 0.10
            strong_threshold: float = 0.45
            return_counts: bool = True
            choices, counts = classify_signal_by_proportion(choice_df, min_max_values, neutral_ratio, strong_threshold,
                                                            return_counts)

            for choice, count in counts.items():
                existed_count = choice_counts.get(choice)
                if existed_count is None:
                    choice_counts[choice] = count
                else:
                    choice_counts[choice] = existed_count + count

            class_lables = choices.assign(**classified_labels)

            label_filepath = build_filepath(data_dir, "spot", "class_labels", market, start_time)
            parent = Path(label_filepath).parent
            if not parent.exists():
                os.makedirs(parent, exist_ok=True)
            #class_lables.to_csv(label_filepath)
        start_time += datetime.timedelta(hours=1)
        if start_time.month != last_time.month:
            print(f"{last_time.year}-{last_time.month}: {choice_counts}")
    return choice_counts


def create_kline_data():
    # (2023, 8, 13, 13, 11) miss 15 minn
    # 2023-08-14 07 nan
    # 2025-06-29 05 nan
    data_start = datetime.datetime(year=2025, month=6, day=1, tzinfo=datetime.timezone.utc)
    start_time = data_start
    end_time = datetime.datetime(year=2025, month=7, day=1, tzinfo=datetime.timezone.utc)
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
    while start_time < start_time:
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

    start_time = data_start
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

create_kline_data()

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


import numpy as np
from typing import Tuple, Optional
import warnings

warnings.filterwarnings('ignore')


def calculate_future_metrics(
        df: pd.DataFrame,
        n_minutes: int,
        long_ratio: float,
        short_ratio: float,
        future_price_col: str = 'nm_price',
        fill_na: bool = True,
        min_periods: int = None
) -> pd.DataFrame:
    """
    计算未来n分钟的收益和风险指标

    参数:
    ----------
    df : pd.DataFrame
        1秒频率的DataFrame，必须包含列:
        - total_sell_vol, total_sell_amount, total_buy_vol, total_buy_amount
        - high, low
        - nm_price (或指定的未来价格列)
    n_minutes : int
        未来时间窗口长度（分钟）
    future_price_col : str
        未来价格列名，用于计算收益
    fill_na : bool
        是否填充NaN值
    min_periods : int
        滑动窗口最小观测数，None表示需要全部n*60个观测

    返回:
    ----------
    pd.DataFrame
        添加了以下列的原始DataFrame:
        - price: 当前价格
        - max_high_nmin: 未来n分钟的最高价的最高值
        - min_high_nmin: 未来n分钟的最高价的最低值
        - max_low_nmin: 未来n分钟的最低价的最低值
        - min_low_nmin: 未来n分钟的最低价的最低值
        - future_return: 未来n分钟收益率
        - risk_rate: 风险率
        - long_signal: 做多信号 (如果未来收益率>0)
        - short_signal: 做空信号 (如果未来收益率<0)
    """

    # 创建副本，避免修改原始数据
    df = df.copy()
    # 参数验证
    required_cols = ['total_sell_vol', 'total_sell_amount',
                     'total_buy_vol', 'total_buy_amount', 'high', 'low']

    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"DataFrame缺少必要列: {missing_cols}")

    # 1. 计算当前价格
    df['price'] = (df['total_sell_amount'] + df['total_buy_amount']) / \
                  (df['total_sell_vol'] + df['total_buy_vol'])

    # 2. 处理NaN值
    if fill_na:
        # 前向填充，然后后向填充
        df['price'] = df['price'].ffill().bfill()
        df['high'] = df['high'].ffill().bfill()
        df['low'] = df['low'].ffill().bfill()
        df[future_price_col] = df['price'].shift(-1)

    # 3. 计算未来n分钟的窗口大小（1秒频率 → n*60秒）
    window_size = n_minutes * 60

    # 设置最小观测数
    if min_periods is None:
        min_periods = window_size  # 默认需要完整窗口

    # 4. 计算未来n分钟的高低价统计
    # 注意：使用.shift(-1)因为下一行开仓
    # max_high_nmin: 未来n分钟的最高价的最高值
    df['max_high_nmin'] = df['high'].shift(-1).rolling(
        window=window_size, min_periods=min_periods
    ).max()

    reversed_max = df['high'].shift(-1)[::-1].rolling(window=window_size, min_periods=min_periods).max()[::-1]
    df['max_high_nmin'] = reversed_max


    # min_high_nmin: 未来n分钟的最高价的最低值
    df['min_high_nmin'] = df['high'].shift(-1)[::-1].rolling(
        window=window_size, min_periods=min_periods
    ).min()[::-1]

    # max_low_nmin: 未来n分钟的最低价的最低值
    df['max_low_nmin'] = df['low'].shift(-1)[::-1].rolling(
        window=window_size, min_periods=min_periods
    ).max()[::-1]

    # min_low_nmin: 未来n分钟的最低价的最低值
    df['min_low_nmin'] = df['low'].shift(-1)[::-1].rolling(
        window=window_size, min_periods=min_periods
    ).min()[::-1]

    # 5. 计算未来n分钟后的价格（用于计算收益）
    # 获取未来n分钟后的价格（窗口结束时的价格）
    df['future_price_nmin'] = df[future_price_col].shift(-window_size)

    # 6. 计算收益率
    # 使用下一行的开仓价和n分钟后的清仓价
    df['future_return'] = (df['future_price_nmin'] - df['price'].shift(-1)) / df['price'].shift(-1)

    # 7. 计算风险率
    df['risk_rate'] = np.nan

    # 当收益率为正时（做多），风险率 = (开仓价 - 期间最低价) / 开仓价
    long_mask = df['future_return'] >= 0
    if long_mask.any():
        # 开仓价
        entry_price = df['price'].shift(-1)
        # 期间最低价是low的最小值
        min_low = df['min_low_nmin']
        df.loc[long_mask, 'risk_rate'] = (entry_price[long_mask] - min_low[long_mask]) / entry_price[long_mask]

    # 当收益率为负时（做空），风险率 = (期间最高价 - 开仓价) / 开仓价
    short_mask = df['future_return'] < 0
    if short_mask.any():
        entry_price = df['price'].shift(-1)
        max_high = df['max_high_nmin']
        df.loc[short_mask, 'risk_rate'] = (max_high[short_mask] - entry_price[short_mask]) / entry_price[short_mask]

    # 8. 生成交易信号
    df['long_signal'] = (df['future_return'] >= long_ratio).astype(int)
    df['short_signal'] = (df['future_return'] <= short_ratio).astype(int)

    # 9. 添加风险收益比
    df['risk_return_ratio'] = abs(df['risk_rate'] / (df['future_return'] + 1e-10))
    df['long_signal'][df['risk_return_ratio'] > 0.6] = 0
    df['short_signal'][df['risk_return_ratio'] > 0.6] = 0
    # 10. 清理临时列
    df.drop(['future_price_nmin'], axis=1, inplace=True)

    return df


def calculate_metrics_for_multiple_windows(
        df: pd.DataFrame,
        windows: list = [1, 3, 5, 10, 15, 30],
        future_price_col: str = 'nm_price',
        open_ratio:float = 0.01
) -> pd.DataFrame:
    """
    为多个时间窗口计算未来指标

    参数:
    ----------
    df : pd.DataFrame
        原始数据
    windows : list
        时间窗口列表（分钟）
    future_price_col : str
        未来价格列名

    返回:
    ----------
    pd.DataFrame
        包含所有窗口指标的DataFrame
    """
    result_df = pd.DataFrame(index=df.index) # df.copy()

    for n_min in windows:
        #print(f"计算 {n_min} 分钟窗口指标...")
        valve = open_ratio * n_min / 30
        temp_df = calculate_future_metrics(
            df=df,
            n_minutes=n_min,
            long_ratio=valve,
            short_ratio=-valve,
            future_price_col=future_price_col,
            fill_na=True,
            min_periods=None
        )

        suffix = f"_{n_min}min"
        new_cols = ['long_signal','short_signal']
        for col in new_cols:
            new_col_name = f"{col}{suffix}"
            result_df[new_col_name] = temp_df[col]

    return result_df


def get_ticks(base_dir: str,  dt: datetime.datetime, market_type: str) -> Optional[pd.DataFrame]:
    """
    读取某小时的1s数据
    data_type: "feature" 或 "target"
    dt: 精确到小时（minute=0, second=0）
    返回 (3600, feature_dim)
    """
    file_path = build_filepath(base_dir, "spot", "ticks", market_type, dt)
    if os.path.exists(file_path):
        df = pd.read_csv(file_path, sep=',', compression='gzip')
        df['timestamp'] = df['timestamp'].astype(np.int64)
        df = df.set_index("timestamp")
        return df
    else:
        return None

def calculate_lables(dt: datetime.datetime, base_dir, market_type, valve):
    hf = get_ticks(base_dir, dt, market_type)
    if hf is None:
        return None
    next_hour = dt + datetime.timedelta(hours=1)
    next_hf = get_ticks(base_dir, next_hour, market_type)
    if next_hf is None:
        return None
    all_data = pd.concat([hf, next_hf], axis=0)
    df = all_data
    all_labels = calculate_metrics_for_multiple_windows(df, open_ratio=valve)
    return all_labels.loc[dt.timestamp(): next_hour.timestamp()].dropna().copy()

def test_cal_labels(base_dir = 'data', ratio=0.006, label_prefix="ls1"):
    dt_start = datetime.datetime(year=2023, month=3, day=1, tzinfo=datetime.timezone.utc)
    dt_end = datetime.datetime(year=2025, month=11, day=1, tzinfo=datetime.timezone.utc)
    #dt_start = datetime.datetime(year=2024, month=1, day=1, hour=19, tzinfo=datetime.timezone.utc)
    total_sum = None
    batch_sum = None
    total_num = 0
    market_type = "BTC_USDT"
    all_sums = dict()
    label_name = f"{label_prefix}_labels"
    while dt_start < dt_end:
        labels = calculate_lables(dt_start, base_dir, market_type, ratio)
        file_path = build_filepath(base_dir, "spot", label_name, market_type, dt_start)

        if labels is not None:
            p_path = Path(file_path).parent
            if not p_path.exists():
                #print(f"Create dir: {p_path}")
                os.makedirs(p_path)
            labels.to_csv(file_path)

            total_num += 1
            sum_data = labels.sum()
            all_sums[dt_start] = sum_data.copy()
            if total_sum is None:
                total_sum = sum_data
            else:
                total_sum += sum_data
            if batch_sum is None:
                batch_sum = sum_data
            else:
                batch_sum += sum_data
            if total_num % 168 == 0:
                print(f"{dt_start} [{total_num}]: {total_sum / total_num}")
                print(f"{batch_sum/168}")
                batch_sum = None
        dt_start += datetime.timedelta(hours=1)
    sum_df = pd.DataFrame(all_sums).T
    sum_df.index.name = 'datetime'
    sum_df = sum_df.reset_index()
    sum_record = f"{base_dir}/spot/{label_name}/stat.csv"
    sum_df.to_csv(sum_record, index=False)
    total_sum /= total_num
    return total_sum, total_num

ratio_names = [(0.01,'ls0'),(0.008,'ls1'),(0.006,'ls2'),(0.005,'ls3'),(0.004,'ls4'),(0.003,'ls5')]
