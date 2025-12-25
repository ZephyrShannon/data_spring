import pandas as pd
import numpy as np


def calculate_sma(df: pd.DataFrame, price_col: str, window: int) -> pd.Series:
    """
    计算简单移动平均线 (Simple Moving Average)

    数据要求:
        - DataFrame 列: 包含 price_col
        - price_col (str): 用于计算的价格列名 (e.g., 'close')

    前置数据要求:
        - 至少需要 window 行数据

    返回:
        - pd.Series: SMA 值，不足 window 行的地方为 NaN
    """
    if price_col not in df.columns:
        raise ValueError(f"列 '{price_col}' 不存在于 DataFrame 中")
    return df[price_col].rolling(window=window).mean()


def calculate_ema(df: pd.DataFrame, price_col: str, span: int) -> pd.Series:
    """
    计算指数移动平均线 (Exponential Moving Average)

    数据要求:
        - DataFrame 列: 包含 price_col
        - price_col (str): 用于计算的价格列名 (e.g., 'close')
        - span (int): EMA 的跨度 (对应 alpha = 2 / (span + 1))

    前置数据要求:
        - 理论上需要足够长的数据以使 EMA 收敛，实践中通常认为 3*span 行左右可获得稳定值

    返回:
        - pd.Series: EMA 值
    """
    if price_col not in df.columns:
        raise ValueError(f"列 '{price_col}' 不存在于 DataFrame 中")
    # adjust=False 使得 alpha = 2 / (span + 1)
    return df[price_col].ewm(span=span, adjust=False).mean()


def calculate_macd(df: pd.DataFrame, price_col: str = 'close') -> pd.DataFrame:
    """
    计算 MACD 指标

    数据要求:
        - DataFrame 列: 包含 price_col
        - price_col (str): 用于计算的价格列名 (默认 'close')

    前置数据要求:
        - 理论上无限，但实践中需要至少几十行数据才能稳定。
        - 内部使用 EMA(12) 和 EMA(26)，所以收敛速度取决于较长的 EMA(26)。

    返回:
        - pd.DataFrame: 包含 'DIF', 'DEA', 'BAR' 列
    """
    if price_col not in df.columns:
        raise ValueError(f"列 '{price_col}' 不存在于 DataFrame 中")

    ema_12 = df[price_col].ewm(span=12, adjust=False).mean()
    ema_26 = df[price_col].ewm(span=26, adjust=False).mean()
    dif = ema_12 - ema_26
    dea = dif.ewm(span=9, adjust=False).mean()
    bar = (dif - dea) * 2

    return pd.DataFrame({'DIF': dif, 'DEA': dea, 'BAR': bar})


def calculate_bollinger_bands(df: pd.DataFrame, price_col: str = 'close', window: int = 20,
                              num_std: float = 2) -> pd.DataFrame:
    """
    计算布林带 (Bollinger Bands)

    数据要求:
        - DataFrame 列: 包含 price_col
        - price_col (str): 用于计算的价格列名 (默认 'close')
        - window (int): 计算均值和标准差的窗口大小 (默认 20)
        - num_std (float): 标准差的倍数 (默认 2)

    前置数据要求:
        - 至少需要 window 行数据

    返回:
        - pd.DataFrame: 包含 'BB_Middle', 'BB_Upper', 'BB_Lower', 'BB_Width' 列
    """
    if price_col not in df.columns:
        raise ValueError(f"列 '{price_col}' 不存在于 DataFrame 中")

    middle_band = df[price_col].rolling(window=window).mean()
    std_dev = df[price_col].rolling(window=window).std(ddof=0)  # ddof=0 for population std
    upper_band = middle_band + (std_dev * num_std)
    lower_band = middle_band - (std_dev * num_std)
    band_width = (upper_band - lower_band) / middle_band  # Optional

    return pd.DataFrame({
        'BB_Middle': middle_band,
        'BB_Upper': upper_band,
        'BB_Lower': lower_band,
        'BB_Width': band_width
    })


def calculate_rsi(df: pd.DataFrame, price_col: str = 'close', window: int = 14) -> pd.Series:
    """
    计算相对强弱指数 (Relative Strength Index)

    数据要求:
        - DataFrame 列: 包含 price_col
        - price_col (str): 用于计算的价格列名 (默认 'close')
        - window (int): 计算 RSI 的周期 (默认 14)

    前置数据要求:
        - 至少需要 window + 1 行数据 (因为需要计算 t 和 t-1 的差值)

    返回:
        - pd.Series: RSI 值 (0-100)
    """
    if price_col not in df.columns:
        raise ValueError(f"列 '{price_col}' 不存在于 DataFrame 中")

    delta = df[price_col].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_williams_r(df: pd.DataFrame, high_col: str, low_col: str, close_col: str, window: int = 14) -> pd.Series:
    """
    计算威廉指标 (%R)

    数据要求:
        - DataFrame 列: 包含 high_col, low_col, close_col
        - high_col (str): 最高价列名
        - low_col (str): 最低价列名
        - close_col (str): 收盘价列名
        - window (int): 计算周期 (默认 14)

    前置数据要求:
        - 至少需要 window 行数据

    返回:
        - pd.Series: Williams %R 值 (-100 到 0)
    """
    required_cols = [high_col, low_col, close_col]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"缺少列: {missing_cols}")

    highest_high = df[high_col].rolling(window=window).max()
    lowest_low = df[low_col].rolling(window=window).min()
    williams_r = (highest_high - df[close_col]) / (highest_high - lowest_low) * -100
    return williams_r


def calculate_stochastic(df: pd.DataFrame, high_col: str, low_col: str, close_col: str, k_window: int = 14,
                         d_window: int = 3) -> pd.DataFrame:
    """
    计算随机指标 (Stochastic Oscillator)

    数据要求:
        - DataFrame 列: 包含 high_col, low_col, close_col
        - high_col (str): 最高价列名
        - low_col (str): 最低价列名
        - close_col (str): 收盘价列名
        - k_window (int): %K 的计算周期 (默认 14)
        - d_window (int): %D 的 SMA 周期 (默认 3)

    前置数据要求:
        - 至少需要 max(k_window, k_window + d_window - 1) 行数据，即 k_window + d_window - 1

    返回:
        - pd.DataFrame: 包含 '%K', '%D' 列
    """
    required_cols = [high_col, low_col, close_col]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"缺少列: {missing_cols}")

    highest_high = df[high_col].rolling(window=k_window).max()
    lowest_low = df[low_col].rolling(window=k_window).min()
    percent_k = (df[close_col] - lowest_low) / (highest_high - lowest_low) * 100
    percent_d = percent_k.rolling(window=d_window).mean()  # SMA of %K

    return pd.DataFrame({'%K': percent_k, '%D': percent_d})

import pandas as pd
import numpy as np

def calculate_rolling_obv(
    df: pd.DataFrame,
    close_col: str,
    volume_col: str,
    window: int
) -> pd.Series:
    """
    计算滑动窗口内的 OBV 风格净成交量（非累计！）

    参数:
        df: 输入 DataFrame
        close_col: 收盘价列名
        volume_col: 成交量列名
        window: 滑动窗口大小（整数，>=2）

    返回:
        pd.Series: 每个时间点对应窗口内的净 OBV 值（前 window-1 个为 NaN）
    """
    if window < 2:
        raise ValueError("window 必须 >= 2，因为需要比较前一日价格")

    if len(df) == 0:
        return pd.Series([], dtype='float64', index=df.index)

    close = df[close_col]
    volume = df[volume_col]

    # 1. 计算每日方向：+1, -1, 0（基于与前一日比较）
    direction = np.where(
        close > close.shift(1), 1,
        np.where(close < close.shift(1), -1, 0)
    )

    # 2. 计算每日 OBV 贡献量（第0天无前值，设为0）
    daily_obv_flow = direction * volume
    daily_obv_flow.iloc[0] = 0  # 第一天无法比较，贡献为0

    # 3. 滑动窗口求和（过去 window 天，包含当天）
    rolling_obv = daily_obv_flow.rolling(window=window, min_periods=window).sum()

    return rolling_obv.astype('float64')


def calculate_atr(df: pd.DataFrame, high_col: str, low_col: str, close_col: str, window: int = 14) -> pd.Series:
    """
    计算平均真实波幅 (Average True Range)

    数据要求:
        - DataFrame 列: 包含 high_col, low_col, close_col
        - high_col (str): 最高价列名
        - low_col (str): 最低价列名
        - close_col (str): 收盘价列名
        - window (int): 计算 ATR 的周期 (默认 14)

    前置数据要求:
        - 至少需要 window + 1 行数据 (因为需要 t-1 的 close 来计算 TR)

    返回:
        - pd.Series: ATR 值
    """
    required_cols = [high_col, low_col, close_col]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"缺少列: {missing_cols}")

    high = df[high_col]
    low = df[low_col]
    close = df[close_col]

    # 计算 True Range (TR)
    tr0 = high - low
    c_shift = close.shift(1)
    tr1 = (high - c_shift).abs()
    tr2 = (low - c_shift).abs()
    tr = pd.DataFrame({'tr0': tr0, 'tr1': tr1, 'tr2': tr2}).max(axis=1)

    # 计算 ATR (使用 EMA 平滑)
    atr = tr.ewm(span=window, adjust=False).mean()
    return atr


def calculate_roc(df: pd.DataFrame, price_col: str, window: int) -> pd.Series:
    """
    计算变动率指标 (Rate of Change)

    数据要求:
        - DataFrame 列: 包含 price_col
        - price_col (str): 用于计算的价格列名
        - window (int): 计算 ROC 的周期

    前置数据要求:
        - 至少需要 window + 1 行数据

    返回:
        - pd.Series: ROC 值 (百分比)
    """
    if price_col not in df.columns:
        raise ValueError(f"列 '{price_col}' 不存在于 DataFrame 中")
    # .shift(window) 将 t-window 的价格移到 t 行进行计算
    roc = (df[price_col] - df[price_col].shift(window)) / df[price_col].shift(window) * 100
    return roc



def adaptive_normalize(x, window, quantile=0.9, min_scale=1e-6):
    abs_x = x.abs()
    # 滚动分位数
    scale = abs_x.rolling(window=window, min_periods=window//2).quantile(quantile)
    # 防止 scale 太小（如全零）
    scale = np.maximum(scale, min_scale)
    return np.clip(x / scale, -3, 3)


def normalize_obv(obv, window=30):
    """
    对 OBV 进行滚动 z-score 归一化
    """
    # 计算滚动均值和标准差（仅历史数据）
    ma = obv.rolling(window=window, min_periods=int(window/2)).mean()
    std = obv.rolling(window=window, min_periods=int(window/2)).std()

    # 避免除零
    obv_norm = (obv - ma) / ((std + 1e-8) * 2)

    # 裁剪极端值（保留99%分位内信息）
    return np.clip(obv_norm, -3, 3)


def normalize_atr_state_log(atr, window=20, low=0.25, high=4.0):
    """
    计算 ATR 相对状态，并进行 clip + log 缩放
    """
    atr_ma = atr.rolling(window=window, min_periods=1).mean()

    # Step 1: 计算相对波动率（ATR / MA）
    atr_ratio = atr / (atr_ma + 1e-8)

    # Step 2: Clip 到 [low, high]
    atr_clipped = np.clip(atr_ratio, low, high)

    # Step 3: 取自然对数（log 缩放）
    atr_log = np.log(atr_clipped)

    return atr_log


def normalize_stochastic(k_or_d): # [0，100]→ [0, 1]
    return k_or_d * 0.01

def normalize_bollinger_width(x, max_width=0.5):
    # 经验：99% 场景下 < 0.5（50%）
    bw_clipped = np.clip(x, 0, max_width)
    return bw_clipped / max_width  # → [0, 1]

def normalize_rsi(x):  # [0，100]→ [0, 1]
    return x * 0.01

def normalize_bollinger_width(bw, max_width=0.5):
    # 经验：99% 场景下 < 0.5（50%）
    bw_clipped = np.clip(bw, 0, max_width)
    return bw_clipped / max_width  # → [0, 1]

def normalize_wr(x):
    return (x + 100) * 0.01
    # 或映射到 [0,1]: return (df[wr_col] + 100) / 100


def normalize_volume_rolling(vol, window=20):
    # 方法1：滚动 z-score
    vol_norm = (vol - vol.rolling(window).mean()) / (vol.rolling(window).std() + 1e-8)

    # 方法2：更稳健——用中位数和 MAD（抗异常值）
    # median = vol.rolling(window).median()
    # mad = (vol - median).abs().rolling(window).median()
    # vol_norm = (vol - median) / (mad + 1e-8)

    return np.clip(vol_norm, -5, 5)

def add_low_freq_factors(df: pd.DataFrame, price_factor) -> pd.DataFrame:
    ma24 = calculate_sma(df, "close", 24)
    ma120 = calculate_sma(df, "close", 120)
    #ma250 = calculate_sma(df, "price", 250)
    macd = calculate_macd(df, "close")[["DIF","DEA"]]
    atr24 = calculate_atr(df, "high", "low", "close", 24)
    atr120 = calculate_atr(df, "high", "low", "close", 120)
    obv24 = calculate_rolling_obv(df, "close", "volume", 24)
    obv120 = calculate_rolling_obv(df, "close", "volume", 120)

    df['obv24_low'] = normalize_obv(obv24, 24)  # obv24 * df['open'] * (1 / (300 * 4))
    df['obv120_low'] = normalize_obv(obv120, 30) #obv120 * df['open'] * (1 / (300 * 5))
    df['volume'] = normalize_volume_rolling(df['volume'])

    df['high'] = df['high'] * price_factor
    df['close'] = df['close'] * price_factor
    df['low'] = df['low'] * price_factor
    df['open'] = df['open'] * price_factor
    df['ma24_low'] = ma24 * price_factor
    df['ma120_low'] = ma120 * price_factor
    macd_factor = price_factor * 100 # 百分比
    df['dif_low'] = macd['DIF'] * macd_factor
    df['dea_low'] = macd['DEA'] * macd_factor
    df['atr24_low'] = normalize_atr_state_log(atr24, window=20, low=0.2, high=5) # atr24 * 0.01
    df['atr120_low'] = normalize_atr_state_log(atr120, window=20, low=0.33, high=3)
    return df

def add_mid_freq_factors(df:pd.DataFrame, price_factor, freq_min=1) -> pd.DataFrame:
    ma20 = calculate_sma(df, "close", 20)
    ema12 = calculate_ema(df, "close", 12)
    ema26 = calculate_ema(df, "close", 26)
    macd = calculate_macd(df, "close")[["BAR"]]
    bolling = calculate_bollinger_bands(df, "close", 20, 2)[["BB_Upper", "BB_Lower", "BB_Width"]]
    rsi = calculate_rsi(df)
    wr = calculate_williams_r(df, "high", "low", "close")
    stoch = calculate_stochastic(df, "high", "low", "close")
    # ['ma20_mid', 'ema12_mid', 'ema26_mid', 'b_up_mid', "b_lo_mid"]

    df['ma20_mid'] = ma20 * price_factor
    df['ema12_mid'] = ema12 * price_factor
    df['ema26_mid'] = ema26 * price_factor
    df['b_up_mid'] = bolling['BB_Upper'] * price_factor
    df['b_lo_mid'] = bolling['BB_Lower'] * price_factor

    df['bar_mid'] = adaptive_normalize(macd['BAR'], int(300/freq_min))
    df['b_wd_mid'] = adaptive_normalize(bolling['BB_Width'], 30)  #normalize_bollinger_width(bolling['BB_Width'])
    df['rsi_mid'] = normalize_rsi(rsi)
    df['wr_mid'] = normalize_wr(wr)
    df['stoc_K_mid'] = normalize_stochastic(stoch['%K'])
    df['stoc_D_mid'] = normalize_stochastic(stoch['%D'])

    df['volume'] = normalize_volume_rolling(df['volume'], 30)
    df['high'] = df['high'] * price_factor
    df['close'] = df['close'] * price_factor
    df['low'] = df['low'] * price_factor
    df['open'] = df['open'] * price_factor
    return df


def add_hf_factors(df: pd.DataFrame, price_factor = 1/50000) -> pd.DataFrame:
    ask_roc5 = calculate_roc(df, 'ask_0_price', 30)
    bid_roc5 = calculate_roc(df, "bid_0_price", 30)
    ask_atr = calculate_atr(df, 'high', 'low', 'ask_0_price', 30)
    bid_atr= calculate_atr(df, 'high', 'low', 'bid_0_price', 30)


    ask_atr = normalize_atr_state_log(ask_atr, window=20, low=0.33, high=3) # ask_atr * 50000
    bid_atr = normalize_atr_state_log(bid_atr, window=20, low=0.33, high=3) # bid_atr * 50000
    df = df.assign(
        ask_roc = ask_roc5,
        bid_roc = bid_roc5,
        ask_atr = ask_atr,
        bid_atr = bid_atr,
    )
    return df


# --- 示例用法 ---
if __name__ == "__main__":
    # 创建示例数据
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    data = {
        'open': np.random.rand(100) * 100 + 150,
        'high': np.random.rand(100) * 100 + 160,
        'low': np.random.rand(100) * 100 + 140,
        'close': np.random.rand(100) * 100 + 155,
        'volume': np.random.randint(1000, 10000, size=100)
    }
    df = pd.DataFrame(data, index=dates)

    # 计算各种指标
    df['MA5'] = calculate_sma(df, 'close', 5)
    df['EMA12'] = calculate_ema(df, 'close', 12)
    df['EMA26'] = calculate_ema(df, 'close', 26)

    macd_df = calculate_macd(df)
    df = pd.concat([df, macd_df], axis=1)

    bb_df = calculate_bollinger_bands(df)
    df = pd.concat([df, bb_df], axis=1)

    df['RSI'] = calculate_rsi(df)
    df['W%R'] = calculate_williams_r(df, 'high', 'low', 'close')

    stoch_df = calculate_stochastic(df, 'high', 'low', 'close')
    df = pd.concat([df, stoch_df], axis=1)

    df['OBV'] = calculate_obv(df, 'close', 'volume')
    df['ATR'] = calculate_atr(df, 'high', 'low', 'close')
    df['ROC'] = calculate_roc(df, 'close', 10)

    print(df.tail(10))