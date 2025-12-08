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


def calculate_obv(df: pd.DataFrame, close_col: str, volume_col: str) -> pd.Series:
    """
    计算能量潮指标 (On-Balance Volume)

    数据要求:
        - DataFrame 列: 包含 close_col, volume_col
        - close_col (str): 收盘价列名
        - volume_col (str): 成交量列名

    前置数据要求:
        - 需要所有历史数据以获得准确的累计值

    返回:
        - pd.Series: OBV 值
    """
    required_cols = [close_col, volume_col]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"缺少列: {missing_cols}")

    obv = pd.Series(index=df.index, dtype='float64')
    obv.iloc[0] = df[volume_col].iloc[0]  # 初始化

    # 使用 vectorized operations
    price_change = df[close_col].diff()
    volume = df[volume_col]

    # 创建变化标志: 1 for up, -1 for down, 0 for unchanged
    change_sign = np.sign(price_change)
    # 如果价格不变，则不改变 OBV (保持 0，后续 cumsum 会处理)
    # 但通常实现是价格不变时 OBV 也不变，所以我们用前一个值
    # 这里简化处理，价格不变时 volume 不计入

    # 计算每日 OBV 变化量
    obv_change = change_sign * volume
    # 累计求和得到 OBV
    obv = obv_change.cumsum()
    # 修正初始值 (如果第一个价格变化不为0，则需要调整)
    # 更稳健的方法是从第二个开始计算，第一个单独赋值
    # obv.iloc[0] = volume.iloc[0] if not pd.isna(volume.iloc[0]) else 0
    # for i in range(1, len(df)):
    #     if price_change.iloc[i] > 0:
    #         obv.iloc[i] = obv.iloc[i-1] + volume.iloc[i]
    #     elif price_change.iloc[i] < 0:
    #         obv.iloc[i] = obv.iloc[i-1] - volume.iloc[i]
    #     else:
    #         obv.iloc[i] = obv.iloc[i-1]
    # 上面循环太慢，用向量化替代
    # Vectorized version seems correct based on cumsum logic.
    return obv


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
    tr1 = (high - close.shift(1)).abs()
    tr2 = (low - close.shift(1)).abs()
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


def add_low_freq_factors(df: pd.DataFrame) -> pd.DataFrame:
    ma60 = calculate_sma(df, "close", 60)
    ma120 = calculate_sma(df, "close", 120)
    #ma250 = calculate_sma(df, "price", 250)
    macd = calculate_macd(df, "close")[["DIF","DEA"]]
    atr60 = calculate_atr(df, "high", "low", "close", 60)
    atr120 = calculate_atr(df, "high", "low", "close", 120)
    obv = calculate_obv(df, "close", "volume")
    # 12 col
    df['ma60_5m'] = ma60
    df['ma120_5m'] = ma120
    df['dif_5m'] = macd['DIF']
    df['dea_5m'] = macd['DEA']
    #result_df['ma250'] = ma250
    df['atr60'] = atr60
    df['atr120'] = atr120
    df['obv'] = obv
    return df


def add_1m_factors(df:pd.DataFrame) -> pd.DataFrame:
    ma20 = calculate_sma(df, "close", 20)
    ema12 = calculate_ema(df, "close", 12)
    ema26 = calculate_ema(df, "close", 26)
    macd = calculate_macd(df, "close")[["BAR"]]
    bolling = calculate_bollinger_bands(df, "close", 20, 2)[["BB_Upper", "BB_Lower", "BB_Width"]]
    rsi = calculate_rsi(df)
    wr = calculate_williams_r(df, "high", "low", "close")
    stoch = calculate_stochastic(df, "high", "low", "close")
    df['ma20_1m'] = ma20
    df['ema12_1m'] = ema12
    df['ema26_1m'] = ema26
    df['bar_1m'] = macd['BAR']
    df['b_up_1m'] = bolling['BB_Upper']
    df['b_lo_1m'] = bolling['BB_Lower']
    df['b_wd_1m'] = bolling['BB_Width']
    df['rsi_1m'] = rsi
    df['wr_1m'] = wr
    df['stoc_K_1m'] = stoch['%K']
    df['stoc_D_1m'] = stoch['%D']
    return df


def add_hf_factors(df: pd.DataFrame) -> pd.DataFrame:
    ask_roc5 = calculate_roc(df, 'ask_0_price', 5)
    bid_roc5 = calculate_roc(df, "bid_0_price", 5)
    df['ask_roc'] = ask_roc5
    df['bid_roc'] = bid_roc5
    ask_atr = calculate_atr(df, 'high', 'low', 'ask_0_price', 15)
    bid_atr= calculate_atr(df, 'high', 'low', 'bid_0_price', 15)
    df['ask_atr'] = ask_atr
    df['bid_atr'] = bid_atr
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