import datetime
import os.path
import typing_extensions
from typing import Tuple
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import gzip
from data_downloader.file_checker import build_filepath
import calendar
from data_loader.factor_tools import add_hf_factors


def get_last_month(dt_curr: datetime.datetime):
    if dt_curr.month == 1:
        return datetime.datetime(year=dt_curr.year - 1, month=12, day=1)
    else:
        return datetime.datetime(year=dt_curr.year, month=dt_curr.month - 1, day=1)


def get_data_hour(dt: datetime.datetime):
    if dt.minute == 0 and dt.second == 0:
        return dt - datetime.timedelta(hours=1)
    else:
        return datetime.datetime(year=dt.year, month=dt.month, day=dt.day, hour=dt.hour, tzinfo=dt.tzinfo)


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


class TimeSeriesDataset(Dataset):
    def __init__(
            self,
            data_dir: str,
            market: str,
            time_list: list,
            interval: int,
            seq_len_seconds: int = 600,
            drop_head=0,
            mid_type="factor_k5m",
            low_type="factor_k1h",
    ):
        # assert seq_len_seconds == 600, "本实现默认 seq_len=3600"
        self.ticks_cache = HourlyCache(data_dir, market, "ticks")
        self.labels_cache = None
        self.labels_cache_date = None
        self.kline_mid = MonthlyDataCache(data_dir, market, mid_type)  # 5个
        self.kline_low = MonthlyDataCache(data_dir, market, low_type)  # 12个
        self.start_time = time_list[0]
        self.end_time = time_list[-1] + datetime.timedelta(hours=1)
        self.nr = int((len(time_list) * 3600 - drop_head - seq_len_seconds) / interval)
        self.interval = interval
        self.seq_len_seconds = seq_len_seconds
        self.drop_head = drop_head


    def _get_labels(self, dt_curr: datetime.datetime):
        data_hour = get_data_hour(dt_curr)
        if is_same_hour(self.labels_cache_date, data_hour):
            # print(f"[0]Return cache of date: {self.cache_date_now}")
            return self.labels_cache.loc[dt_curr.timestamp()]
        self.labels_cache = load_hourly_data(self.ticks_cache.data_dir, self.ticks_cache.market, data_hour, "class_labels")
        if self.labels_cache is not None:
            # timestamp,ls_choice_5m,ls_choice_15m,ls_choice_30m,ls_choice_60m,ls_choice_180m,volat_5m,volat_15m,volat_30m,volat_60m,volat_180m
            columns = ['ls_choice_5m', 'volat_5m', 'ls_choice_15m', 'volat_15m', 'ls_choice_30m', 'volat_30m', 'ls_choice_60m', 'volat_60m', 'ls_choice_180m', 'volat_180m']
            self.labels_cache = self.labels_cache[columns]
            self.labels_cache_date = data_hour
            return self.labels_cache.loc[dt_curr.timestamp()]


    def get_data_row_num(self):
        return self.nr

    def _get_lf_mid_seq(self, end_dt: datetime):
        with_factors = self.kline_mid.get_klines(end_dt)
        return with_factors.loc[with_factors.index <= int(end_dt.timestamp())].iloc[-self.seq_len_seconds:]

    def _get_lf_low_seq(self, end_dt: datetime) -> pd.DataFrame:
        with_factors = self.kline_low.get_klines(end_dt)
        return with_factors.loc[with_factors.index <= int(end_dt.timestamp())].iloc[-self.seq_len_seconds:]

    def __len__(self):
        return self.nr

    def get_datetime(self, idx: int) -> datetime.datetime:
        return self.start_time + datetime.timedelta(seconds=idx * self.interval + self.seq_len_seconds + self.drop_head)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # print(f"Get data:{idx}")
        # === 1. 计算输入序列的结束时间（即当前“现在”）===
        now = self.start_time + datetime.timedelta(seconds=idx * self.interval + self.seq_len_seconds + self.drop_head)
        target_start = now
        # === 2. 获取高频特征 X_high ===
        dt_curr = now
        df = self.ticks_cache.get_hf_data(dt_curr, add_factor=True)
        data_start = now - datetime.timedelta(hours=1)  # 不包含
        # add_hf_factors(df)
        X_high = df.loc[(df.index <= int(now.timestamp()))].iloc[-self.seq_len_seconds:].copy()
        # === 3. 获取低频特征（全部基于 now 时间点向前取历史）===
        X_mid = self._get_lf_mid_seq(end_dt=now)  # (3600, 3) # 有问题
        X_low = self._get_lf_low_seq(end_dt=now)  # (3600, 3) # 有问题

        X_high['t_of_day'] = (X_high.index % 86400) / 86400
        X_low = X_low.copy()
        X_low['tod_low'] = (X_low.index % 86400) / 86400
        X_mid = X_mid.copy()
        X_mid['tod_mid'] = (X_mid.index % 86400) / 86400
        if (X_high.shape[0] != X_low.shape[0]) or (X_low.shape[0] != X_mid.shape[0]):
            msg = f"X_high.shape = {X_high.shape}, X_low.shape={X_low.shape}, X_mid={X_mid.shape}"
            print(msg)
            raise Exception(msg)
        # === 4. 拼接所有特征 ===
        '''
        
        X = np.concatenate([
            X_high,  # (seq_len, F)
            X_low,  # (seq_len, x)
            X_mid,  #
        ], axis=1)  # (seq_len, F+x)
        '''
        # columns = list(X_high.columns)
        # for col in X_low.columns:
        #     if not col.endswith("_low"):
        #         col = col + "_low"
        #     columns.append(col)
        # for col in X_mid.columns:
        #     if not col.endswith("_mid"):
        #         col = col + "_mid"
        #     columns.append(col)

        y = self._get_labels(now)

        # === 6. 转为 Tensor ===
        x_high_tensor = torch.FloatTensor(X_high.values)
        x_mid_tensor = torch.FloatTensor(X_mid.values)
        x_low_tensor = torch.FloatTensor(X_low.values)
        y_tensor = torch.FloatTensor(y.values.ravel())
        return x_high_tensor, x_mid_tensor, x_low_tensor, y_tensor


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
    df.ffill(inplace=True)
    return df


def cut_data(df: pd.DataFrame, start_dt: datetime.datetime, nr: int):
    return df.loc[df.index <= int(start_dt.timestamp())][-nr:]


def fill_invalid(col: pd.Series, val: float):
    col = col.fillna(val)
    return col.replace([np.inf, -np.inf], val)


def compute_price_from_amount_volume(
        df,
        amount_col='amount',
        volume_col='volume',
        price_clip_threshold=1000000.0,
        min_volume_abs=1e-7
):
    """
    安全计算 price = amount / volume，并进行数值校验：
      - 若 |volume| < min_volume_abs → 返回 0.0
      - 若 |price| > price_clip_threshold → 返回 0.0
      - 否则返回 amount / volume

    参数:
        df (pd.DataFrame): 输入 DataFrame
        amount_col (str): 金额列名
        volume_col (str): 成交量列名
        price_clip_threshold (float): 价格绝对值阈值，默认 200000
        min_volume_abs (float): 体积最小绝对值阈值，默认 1e-6

    返回:
        pd.Series: 校验后的 price 列
    """
    amount = df[amount_col]
    volume = df[volume_col]

    # 第一步：安全除法，避免除零
    raw_price = np.where(
        np.abs(volume) < min_volume_abs,
        0.0,
        amount / volume
    )

    # 第三步：校验价格是否超出合理范围
    final_price = np.where(
        np.abs(raw_price) > price_clip_threshold,
        0.0,
        raw_price
    )

    return pd.Series(final_price, index=df.index, name='price')


def normalize_hfd(hf_data: pd.DataFrame):
    btc_base = 50000
    base_factor = 1 / btc_base
    hf_avg_price = hf_data["ask_0_price"] + hf_data["bid_0_price"]
    hf_avg_price_base = hf_avg_price * (base_factor * 0.5)

    last_ask_price = 0
    last_bid_price = 0
    for i in range(0, 20):
        # price_n = f"hf_data[ask_{i}_price] = (hf_data[ask_{i}_price] / hf_data[ask_{i-1}_price])/1 * 10000"
        hf_ask_base = hf_data[f"ask_{i}_price"] * base_factor
        hf_bid_base = hf_data[f"bid_{i}_price"] * base_factor
        if i > 0:
            ask_price_dif = hf_data[f"ask_{i}_price"] - last_ask_price
            last_ask_price = hf_data[f"ask_{i}_price"]
            hf_data[f"ask_{i}_price"] = ask_price_dif
            bid_price_dif = last_bid_price - hf_data[f"bid_{i}_price"]
            last_bid_price = hf_data[f"bid_{i}_price"]
            hf_data[f"bid_{i}_price"] = bid_price_dif
        else:
            last_ask_price = hf_data[f"ask_{i}_price"]
            hf_data[f"ask_{i}_price"] = hf_ask_base
            last_bid_price = hf_data[f"bid_{i}_price"]
            hf_data[f"bid_{i}_price"] = hf_bid_base

        hf_data[f"ask_{i}_vol"] = hf_data[f"ask_{i}_vol"] * hf_ask_base
        hf_data[f"ask_{i}_order"] = hf_data[f"ask_{i}_order"] * hf_ask_base
        hf_data[f'ask_{i}_trade'] = hf_data[f"ask_{i}_trade"] * hf_ask_base

        hf_data[f"bid_{i}_vol"] = hf_data[f"bid_{i}_vol"] * hf_bid_base
        hf_data[f"bid_{i}_order"] = hf_data[f"bid_{i}_order"] * hf_bid_base
        hf_data[f'bid_{i}_trade'] = hf_data[f"bid_{i}_trade"] * hf_bid_base

    hf_data['total_sell_amount'] = compute_price_from_amount_volume(hf_data, "total_sell_amount", "total_sell_vol") * base_factor
    hf_data['total_buy_amount'] = compute_price_from_amount_volume(hf_data, "total_buy_amount", "total_buy_vol") * base_factor
    hf_data['total_sell_order_amount'] = compute_price_from_amount_volume(hf_data, "total_sell_order_amount",
                                                                          "total_sell_order_vol") * base_factor
    hf_data['total_buy_order_amount'] = compute_price_from_amount_volume(hf_data, "total_buy_order_amount",
                                                                         "total_buy_order_vol") * base_factor
    hf_data['high'] = hf_data['high'] * base_factor
    hf_data['low'] = hf_data['low'] * base_factor
    return hf_avg_price_base


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

    def load_hourly_data(self, dt, add_factor=False) -> pd.DataFrame:
        if is_same_hour(self.cache_date_now, dt):
            if self.cache_data_now is not None:
                return self.cache_data_now
        if is_same_hour(self.cache_date_last, dt):
            if self.cache_data_last is not None:
                return self.cache_data_last
        try:
            data = load_hourly_data(self.data_dir, self.market, dt, self.data_type)
            if data is None:
                return None
            normalize_hfd(data)
            if add_factor:
                last_hour = dt - datetime.timedelta(hours=1)
                last_data = self.load_hourly_data(last_hour, False)
                if last_data is not None:
                    all_data = pd.concat([last_data[data.columns], data])
                else:
                    all_data = data
                with_factor = add_hf_factors(all_data).bfill()
                return with_factor.iloc[with_factor.index > dt.timestamp()][:3600]
            return data.copy()
        except Exception as e:
            return None

    def exists(self, dt_curr: datetime.datetime):
        file_path = build_filepath(base=self.data_dir, biz='spot', data_type=self.data_type,
                                   market=self.market, dt=dt_curr)
        return os.path.exists(file_path)

    def get_hf_data(self, dt_curr: datetime.datetime, add_factor=False):
        data_hour = get_data_hour(dt_curr)
        if is_same_hour(self.cache_date_now, data_hour):
            # print(f"[0]Return cache of date: {self.cache_date_now}")
            return self.cache_data_merged
        # 需要load新的了

        cache_data_last = self.cache_data_now
        cache_date_last = self.cache_date_now
        # 全新load
        cache_data_now = self.load_hourly_data(data_hour, add_factor)
        # print(f"[1]Load data of hour:{data_hour}")

        if cache_data_now is None:
            return None
        cache_date_now = data_hour

        if self.need_1hour:
            last_hour = data_hour - datetime.timedelta(hours=1)
            if not is_same_hour(cache_date_last, last_hour):
                # print(f"[2]Reset data of {last_hour}")
                cache_data_last = self.load_hourly_data(last_hour, add_factor)
                if cache_data_last is not None:
                    cache_date_last = last_hour
                else:
                    cache_date_last = None
        else:
            cache_data_last = None

        if cache_data_last is not None:
            cache_data_merged = pd.concat([cache_data_last, cache_data_now])
        else:
            cache_data_merged = cache_data_now.copy()
        cache_data_merged = cache_data_merged.ffill()
        if 'high' in cache_data_now.columns:
            # print("ffill high and low")
            first_line_index = cache_data_now.index[0]
            first_high = cache_data_now.loc[first_line_index, 'high']
            if pd.isna(first_high):
                cache_data_now.loc[first_line_index, 'high'] = cache_data_merged.loc[first_line_index]['high']
                cache_data_now.loc[first_line_index, 'low'] = cache_data_merged.loc[first_line_index]['low']

        self.cache_data_now = cache_data_now
        self.cache_date_now = cache_date_now
        self.cache_data_last = cache_data_last
        self.cache_date_last = cache_date_last
        self.cache_data_merged = cache_data_merged
        return cache_data_merged


def test_monthly_datacache():
    data_dir = "/Users/zephyr/codes/alpha_spring/data_spring/data"
    market = "BTC_USDT"
    start_time = datetime.datetime(year=2023, month=3, day=1, tzinfo=datetime.timezone.utc)
    end_time = datetime.datetime(year=2025, month=11, day=1, tzinfo=datetime.timezone.utc)
    start_time = datetime.datetime(year=2024, month=1, day=1, tzinfo=datetime.timezone.utc)
    end_time = datetime.datetime(year=2024, month=1, day=2, tzinfo=datetime.timezone.utc)
    biz = 'spot'
    data_type = "candlesticks_5m"
    data_type = "candlesticks_1m"
    mdc = MonthlyDataCache(data_dir, market, data_type)
    dt_curr = start_time
    mdc.get_klines(dt_curr)
    self = mdc


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
        file_path = build_filepath(base=self.data_dir, biz='spot', data_type=self.data_type, market=self.market, dt=dt)
        if os.path.exists(file_path):
            with gzip.open(file_path, 'rt') as f:
                return pd.read_csv(f)
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
        gap = cache_data_merged['timestamp'].iloc[1] - cache_data_merged['timestamp'].iloc[0]
        cache_data_merged['timestamp'] = cache_data_merged['timestamp'] + gap
        cache_data_merged.set_index('timestamp', inplace=True)
        self.cache_data_merged = cache_data_merged
        return self.cache_data_merged


def get_all_file_list(data_dir: str, biz: str, data_type: str, market: str, start_time: datetime.datetime,
                      end_time: datetime.datetime, interval=60, seq_len=3600, drop_head=60, low_type="factor_k1h",
                      mid_type="factor_k5m"):
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
                if len(time_list) > 0:
                    print(f"Add new segment:[{format_to_hours(time_list[0])}-{format_to_hours(time_list[-1])}]")
                    file_mergable.append(
                        TimeSeriesDataset(data_dir, market, time_list, interval, seq_len, drop_head, mid_type,
                                          low_type))
                    time_list.clear()
                else:
                    print(f"drop one hour{format_to_hours(time_list[0])}")
        start_time = start_time + one_hour

    if len(time_list) != 0:
        if len(time_list) > 0:
            print(f"Add new segment:[{format_to_hours(time_list[0])}-{format_to_hours(time_list[-1])}]")
            file_mergable.append(
                TimeSeriesDataset(data_dir, market, time_list, interval, seq_len, drop_head, mid_type, low_type))
            time_list.clear()
        else:
            print(f"drop one hour{format_to_hours(time_list[0])}")
    return file_mergable


class SegmentSets(Dataset):
    all_segments: typing_extensions.List[typing_extensions.Tuple[int, int, TimeSeriesDataset]]
    total_row_num: int

    def __init__(self, all_segments: typing_extensions.List[TimeSeriesDataset]):
        self.all_segments = list()
        total_num = 0
        for seg in all_segments:
            end_num = total_num + seg.get_data_row_num()
            self.all_segments.append((total_num, end_num, seg))
            total_num = end_num
        self.total_row_num = total_num

    def __len__(self):
        return self.total_row_num

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        for start, end, seg in self.all_segments:
            if start <= idx < end:
                # print(f"Get {idx - start} from seg[{start}-{end}]")
                return seg[idx - start]


def test_file_list():
    data_dir = "/Users/zephyr/codes/alpha_spring/data_spring/data"
    market = "BTC_USDT"
    start_time = datetime.datetime(year=2023, month=3, day=1, tzinfo=datetime.timezone.utc)
    end_time = datetime.datetime(year=2025, month=11, day=1, tzinfo=datetime.timezone.utc)
    start_time = datetime.datetime(year=2024, month=1, day=1, tzinfo=datetime.timezone.utc)
    end_time = datetime.datetime(year=2024, month=1, day=2, tzinfo=datetime.timezone.utc)
    biz = 'spot'
    data_type = "labels"
    all_list = get_all_file_list(data_dir, biz, data_type, market, start_time, end_time, 60, 600, 60)

    ss = SegmentSets(all_list)
    ss_len = ss.__len__()
    a = ss[0]
    a = ss[49]
    a = ss[50]
    a = ss[51]
    a = ss[109]
    a = ss[110]
    a = ss[111]
    a = ss[169]
    a = ss[170]

# test_file_list()

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
