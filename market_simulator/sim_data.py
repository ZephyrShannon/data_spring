import pandas as pd
import numpy as np
import os
import datetime
from typing import Optional, Union
import warnings

warnings.filterwarnings('ignore')


def generate_random_kline_from_index(
        input_df: pd.DataFrame,
        start_price: float = 100.0,
        max_daily_change: float = 0.005,  # 0.5%
        add_overnight_gap: bool = True,
        volatility_factor: float = 1.0,
        random_seed: Optional[int] = None
) -> pd.DataFrame:
    """
    从输入DataFrame的索引生成随机K线数据

    :param input_df: 输入DataFrame，用于获取时间索引
    :param start_price: 起始价格
    :param max_daily_change: 最大单步涨跌幅（如0.005表示0.5%）
    :param add_overnight_gap: 是否添加隔夜跳空缺口
    :param volatility_factor: 波动率因子，大于1增加波动，小于1减少波动
    :param random_seed: 随机种子，用于结果可复现
    :return: 包含'Open'和'Close'列的K线DataFrame
    """

    if random_seed is not None:
        np.random.seed(random_seed)

    # 获取时间索引
    time_index = input_df.index

    if len(time_index) == 0:
        raise ValueError("输入DataFrame的索引不能为空")

    # 检查索引是否为时间类型
    if not isinstance(time_index, pd.DatetimeIndex):
        # 如果不是时间索引，尝试转换为时间索引或创建默认索引
        try:
            time_index = pd.to_datetime(time_index)
        except:
            # 如果转换失败，创建等间距的时间索引
            time_index = pd.date_range(
                start='2024-01-01',
                periods=len(time_index),
                freq='1min'
            )

    # 生成价格序列
    n_points = len(time_index)
    prices = np.zeros(n_points)
    opens = np.zeros(n_points)

    # 起始价格
    current_price = start_price

    # 确定时间频率（用于判断是否隔夜）
    if len(time_index) > 1:
        time_diff = time_index[1] - time_index[0]
    else:
        time_diff = pd.Timedelta('1day')  # 默认

    for i in range(n_points):
        # 判断是否是新的一天（用于添加隔夜跳空）
        is_new_day = False
        if i > 0 and add_overnight_gap:
            current_time = time_index[i]
            prev_time = time_index[i - 1]

            # 判断是否跨越了交易日（简单判断：日期不同）
            if current_time.date() != prev_time.date():
                is_new_day = True

        # 生成开盘价
        if i == 0:
            # 第一天开盘价就是起始价格
            opens[i] = start_price
        elif is_new_day:
            # 新的一天：开盘价 = 前一天收盘价 + 随机隔夜跳空
            overnight_change = np.random.uniform(-max_daily_change * 2, max_daily_change * 2)
            opens[i] = current_price * (1 + overnight_change)
        else:
            # 同一天内：开盘价 = 前一日收盘价
            opens[i] = current_price

        # 生成收盘价（在开盘价基础上随机变动）
        if i == 0:
            # 第一根K线：收盘价 = 开盘价 + 小幅变动
            daily_change = np.random.uniform(-max_daily_change, max_daily_change)
            prices[i] = opens[i] * (1 + daily_change * volatility_factor)
        else:
            # 后续K线：基于前收盘价随机变动
            daily_change = np.random.uniform(-max_daily_change, max_daily_change)
            prices[i] = current_price * (1 + daily_change * volatility_factor)

        # 确保价格为正数
        prices[i] = max(prices[i], 0.01)
        opens[i] = max(opens[i], 0.01)

        # 更新当前价格（用于下一次迭代）
        current_price = prices[i]

    # 创建K线DataFrame
    kline_df = pd.DataFrame(
        {
            'Open': opens,
            'Close': prices
        },
        index=time_index
    )

    # 可选：添加一些技术分析中常见的模式（如支撑阻力）
    kline_df = _add_market_patterns(kline_df, volatility_factor)

    return kline_df


def _add_market_patterns(kline_df: pd.DataFrame, volatility_factor: float = 1.0) -> pd.DataFrame:
    """
    添加一些市场模式，使数据更真实

    :param kline_df: K线数据
    :param volatility_factor: 波动率因子
    :return: 增强后的K线数据
    """
    df = kline_df.copy()
    n_points = len(df)

    # 1. 添加微小噪声到开盘价（实际市场中开盘价可能不完全等于前收盘）
    if n_points > 1:
        noise = np.random.normal(0, df['Close'].std() * 0.001, n_points)
        df['Open'] = df['Open'] + noise
        df['Open'] = df['Open'].abs()  # 确保为正数

    # 2. 模拟支撑阻力：价格在一定范围内震荡
    if n_points > 50:
        # 计算移动平均作为"均衡价格"
        ma_20 = df['Close'].rolling(window=20, min_periods=1).mean()

        # 价格倾向于回归移动平均
        for i in range(20, n_points):
            deviation = df.loc[df.index[i], 'Close'] - ma_20.iloc[i]

            # 如果偏离过大，增加回归压力
            if abs(deviation) > df['Close'].std() * 0.5:
                regression_force = -deviation * 0.1 * volatility_factor
                # 稍微调整收盘价
                df.loc[df.index[i], 'Close'] += regression_force

    # 3. 确保开盘价和收盘价的合理性
    # 实际市场中，开盘价和收盘价通常不会相差太离谱
    max_intraday_change = df['Close'].std() * 3
    for i in range(n_points):
        change = abs(df['Close'].iloc[i] - df['Open'].iloc[i])
        if change > max_intraday_change:
            # 如果日内变动过大，调整收盘价
            direction = 1 if df['Close'].iloc[i] > df['Open'].iloc[i] else -1
            df.loc[df.index[i], 'Close'] = df['Open'].iloc[i] + direction * max_intraday_change * 0.8

    return df


def generate_random_signals_from_kline(
        kline_df: pd.DataFrame,
        signal_type: str = 'random',
        lookback_period: int = 20,
        threshold: float = 0.5,
        random_seed: Optional[int] = None
) -> pd.DataFrame:
    """
    基于K线数据生成随机交易信号

    :param kline_df: K线数据
    :param signal_type: 信号类型 - 'random', 'trend', 'mean_reversion', 'mixed'
    :param lookback_period: 回顾周期
    :param threshold: 信号阈值
    :param random_seed: 随机种子
    :return: 包含信号的DataFrame
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    signals = pd.DataFrame(index=kline_df.index)

    if signal_type == 'random':
        # 纯随机信号
        signals['Long_sign'] = (np.random.random(len(kline_df)) > threshold).astype(int)
        signals['Short_sign'] = (np.random.random(len(kline_df)) > threshold).astype(int)

    elif signal_type == 'trend':
        # 趋势跟踪信号（价格高于移动平均时做多，低于时做空）
        prices = kline_df['Close']
        ma = prices.rolling(window=lookback_period, min_periods=1).mean()

        signals['Long_sign'] = (prices > ma).astype(int)
        signals['Short_sign'] = (prices < ma).astype(int)

    elif signal_type == 'mean_reversion':
        # 均值回归信号（价格偏离移动平均时反向操作）
        prices = kline_df['Close']
        ma = prices.rolling(window=lookback_period, min_periods=1).mean()
        std = prices.rolling(window=lookback_period, min_periods=1).std()
        z_score = (prices - ma) / (std + 1e-8)

        # 价格过低时做多，过高时做空
        signals['Long_sign'] = (z_score < -1).astype(int)
        signals['Short_sign'] = (z_score > 1).astype(int)

    elif signal_type == 'mixed':
        # 混合信号：70%时间跟随趋势，30%时间均值回归
        prices = kline_df['Close']
        ma = prices.rolling(window=lookback_period, min_periods=1).mean()
        std = prices.rolling(window=lookback_period, min_periods=1).std()
        z_score = (prices - ma) / (std + 1e-8)

        # 趋势部分
        trend_long = (prices > ma).astype(int)
        trend_short = (prices < ma).astype(int)

        # 均值回归部分
        reversion_long = (z_score < -1).astype(int)
        reversion_short = (z_score > 1).astype(int)

        # 随机选择模式
        mode_selector = np.random.random(len(kline_df))

        signals['Long_sign'] = np.where(
            mode_selector < 0.7, trend_long, reversion_long
        )
        signals['Short_sign'] = np.where(
            mode_selector < 0.7, trend_short, reversion_short
        )

    # 生成信号概率（基于信号强度）
    signals['Long_prob'] = _generate_signal_probabilities(
        signals['Long_sign'], kline_df['Close']
    )
    signals['Short_prob'] = _generate_signal_probabilities(
        signals['Short_sign'], kline_df['Close']
    )

    # 确保多空信号不同时为1（实际交易中通常不会同时开多又开空）
    conflicting_signals = (signals['Long_sign'] == 1) & (signals['Short_sign'] == 1)
    if conflicting_signals.any():
        # 随机保留一个信号
        for idx in signals[conflicting_signals].index:
            if np.random.random() > 0.5:
                signals.loc[idx, 'Short_sign'] = 0
            else:
                signals.loc[idx, 'Long_sign'] = 0

    return signals


def _generate_signal_probabilities(
        signals: pd.Series,
        prices: pd.Series,
        base_confidence: float = 0.7
) -> pd.Series:
    """
    为信号生成合理的概率值

    :param signals: 信号序列 (0/1)
    :param prices: 价格序列
    :param base_confidence: 基础置信度
    :return: 概率序列
    """
    n = len(signals)
    probabilities = np.zeros(n)

    for i in range(n):
        if signals.iloc[i] == 1:
            # 有信号时，生成较高的概率值
            if i == 0:
                probabilities[i] = base_confidence + np.random.uniform(0, 0.3)
            else:
                # 考虑近期价格行为：如果价格在预期的方向移动，置信度更高
                recent_return = (prices.iloc[i] - prices.iloc[max(0, i - 5)]) / prices.iloc[max(0, i - 5)]

                # 随机基础值 + 基于近期表现的调整
                base_prob = base_confidence + np.random.uniform(-0.2, 0.3)

                # 如果近期走势与信号方向一致，增加置信度
                if (signals.iloc[i] == 1 and recent_return > 0) or \
                        (signals.iloc[i] == 0 and recent_return < 0):
                    base_prob = min(0.95, base_prob + 0.1)

                probabilities[i] = np.clip(base_prob, 0.55, 0.95)
        else:
            # 无信号时，概率较低但不为零（模型可能有轻微倾向）
            probabilities[i] = np.random.uniform(0.3, 0.5)

    return pd.Series(probabilities, index=signals.index)


# ==================== 测试和示例 ====================

def test_random_kline_generation():
    """测试随机K线生成函数"""

    print("测试1: 生成基本K线数据")
    print("-" * 50)

    # 创建一个示例输入DataFrame
    sample_dates = pd.date_range(
        start='2024-01-01 09:30:00',
        end='2024-01-05 15:00:00',
        freq='1min'
    )
    sample_df = pd.DataFrame(
        {'Dummy': np.random.randn(len(sample_dates))},
        index=sample_dates
    )

    print(f"输入数据形状: {sample_df.shape}")
    print(f"时间范围: {sample_df.index[0]} 到 {sample_df.index[-1]}")
    print(f"总数据点数: {len(sample_df)}")

    # 生成K线数据
    kline_data = generate_random_kline_from_index(
        input_df=sample_df,
        start_price=100.0,
        max_daily_change=0.005,  # 0.5%
        add_overnight_gap=True,
        volatility_factor=1.0,
        random_seed=42
    )

    print(f"\n生成的K线数据形状: {kline_data.shape}")
    print(f"K线数据列: {kline_data.columns.tolist()}")

    # 显示统计信息
    print("\nK线数据统计:")
    print(f"开盘价范围: {kline_data['Open'].min():.2f} - {kline_data['Open'].max():.2f}")
    print(f"收盘价范围: {kline_data['Close'].min():.2f} - {kline_data['Close'].max():.2f}")

    # 计算收益率
    returns = kline_data['Close'].pct_change().dropna()
    print(f"收益率统计:")
    print(f"  均值: {returns.mean() * 100:.4f}%")
    print(f"  标准差: {returns.std() * 100:.4f}%")
    print(f"  最小值: {returns.min() * 100:.4f}%")
    print(f"  最大值: {returns.max() * 100:.4f}%")

    # 检查连续性
    large_gaps = (abs(returns) > 0.01).sum()  # 寻找超过1%的跳空
    print(f"\n大幅跳空(>1%)数量: {large_gaps} ({large_gaps / len(returns) * 100:.1f}%)")

    # 显示前几行数据
    print("\n前5行K线数据:")
    print(kline_data.head())

    print("\n最后5行K线数据:")
    print(kline_data.tail())

    return kline_data, sample_df


def visualize_kline_data(kline_data: pd.DataFrame, title: str = "随机生成的K线数据"):
    """可视化K线数据"""
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8))

    # 绘制价格曲线
    ax1.plot(kline_data.index, kline_data['Close'], label='收盘价',
             color='blue', linewidth=1, alpha=0.7)
    ax1.plot(kline_data.index, kline_data['Open'], label='开盘价',
             color='green', linewidth=0.5, alpha=0.5, linestyle='--')

    ax1.set_title(title, fontsize=14)
    ax1.set_ylabel('价格', fontsize=12)
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)

    # 格式化x轴
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)

    # 绘制收益率分布
    returns = kline_data['Close'].pct_change().dropna() * 100  # 转换为百分比

    ax2.hist(returns, bins=50, edgecolor='black', alpha=0.7)
    ax2.axvline(x=returns.mean(), color='red', linestyle='--',
                label=f'均值: {returns.mean():.3f}%')
    ax2.axvline(x=returns.mean() + returns.std(), color='orange',
                linestyle=':', alpha=0.7, label='±1标准差')
    ax2.axvline(x=returns.mean() - returns.std(), color='orange',
                linestyle=':', alpha=0.7)

    ax2.set_xlabel('收益率 (%)', fontsize=12)
    ax2.set_ylabel('频次', fontsize=12)
    ax2.set_title('收益率分布', fontsize=12)
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # 打印更多统计信息
    print("\n详细统计信息:")
    print(f"价格序列自相关性(滞后1): {kline_data['Close'].autocorr(lag=1):.4f}")
    print(f"价格序列自相关性(滞后5): {kline_data['Close'].autocorr(lag=5):.4f}")

    # 检查是否基本连续
    max_return = abs(returns).max()
    print(f"最大单步收益率: {max_return:.4f}%")
    if max_return < 1.0:  # 最大变动小于1%
        print("✅ 数据基本连续：最大单步变动 < 1%")
    else:
        print("⚠️  注意：存在较大跳空")


def generate_complete_test_data(
        n_points: int = 1000,
        start_date: str = '2024-01-01 09:30:00',
        freq: str = '1min',
        signal_type: str = 'mixed',
        random_seed: int = 42
) -> tuple:
    """
    生成完整的测试数据（K线 + 信号）

    :param n_points: 数据点数
    :param start_date: 开始日期
    :param freq: 频率
    :param signal_type: 信号类型
    :param random_seed: 随机种子
    :return: (kline_data, signals, long_sign, short_sign, long_prob, short_prob)
    """
    # 创建时间索引
    dates = pd.date_range(start=start_date, periods=n_points, freq=freq)
    dummy_df = pd.DataFrame({'Dummy': np.zeros(n_points)}, index=dates)

    # 生成K线数据
    kline_data = generate_random_kline_from_index(
        input_df=dummy_df,
        start_price=100.0,
        max_daily_change=0.005,
        add_overnight_gap=True,
        volatility_factor=1.0,
        random_seed=random_seed
    )

    # 生成信号数据
    signals = generate_random_signals_from_kline(
        kline_df=kline_data,
        signal_type=signal_type,
        lookback_period=20,
        threshold=0.5,
        random_seed=random_seed
    )

    # 分离信号到不同DataFrame（符合sim_trade函数的输入格式）
    long_sign = pd.DataFrame({'Long_sign': signals['Long_sign']})
    short_sign = pd.DataFrame({'Short_sign': signals['Short_sign']})
    long_prob = pd.DataFrame({'Long_prob': signals['Long_prob']})
    short_prob = pd.DataFrame({'Short_prob': signals['Short_prob']})

    return kline_data, signals, long_sign, short_sign, long_prob, short_prob


# 更简洁的一行版本
def generate_probabilities_simple(signal_df: pd.DataFrame, col_name: str, random_seed:Optional[int] = None) -> pd.DataFrame:

    if random_seed is not None:
        np.random.seed(random_seed)

    # 获取信号序列
    """一行代码版本"""
    signals = signal_df['signal'].values
    n = len(signals)
    probs = np.where(
        signals == 0,
        np.random.uniform(0, 0.5, n),
        np.random.uniform(0.5, 1, n)
    )
    return pd.DataFrame({col_name: probs}, index=signal_df.index)

def test():
    long_sign = "../data/15m/signal_long_2601081604.csv"
    long_df = pd.read_csv(long_sign, index_col='datetime', parse_dates=True)
    short_sign = "../data/15m/signal_short_2601081604.csv"
    short_df = pd.read_csv(short_sign, index_col='datetime', parse_dates=True)
    kline = "../data/15m/kline.csv"
    kline_df = pd.read_csv(kline, index_col='datetime', parse_dates=True)
    long_prob = "../data/15m/long_prob.csv"
    long_prob_df = pd.read_csv(long_prob, index_col='datetime', parse_dates=True)
    short_prob = "../data/15m/short_prob.csv"
    short_prob_df = pd.read_csv(short_prob, index_col='datetime', parse_dates=True)



