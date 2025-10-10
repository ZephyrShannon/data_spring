import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler

"""
1 训练数据准备工具
a. resample工具，[1s 5s 15s 30s 暂无] 1min 2min 5min 10min 15min 30min 1h 2h 3h 6h 12h 1d  [4] + 12
b. 添加数据延迟，用于预测: [1s 5s 15s 30s 暂无] 1m  2m 5m 15m 30m 1h 2h 3h 6h 12h 1d 2d 3d 7d 15d 30d  [4] + 12
c. benchmark： BTC 、ETH、 BNB、 SOL、POLYGON、TRON、 BCH、 MOR、DOGI + 加权数据  要18中
d. 归一化数据：
"""


def shift_and_resample(raw_data: pd.DataFrame, resample_time, shift_num):
    """
    @brief 原始数据添加一个1-2个最小时间间隔作为输入数据，因为我们不能预测未来的，所以只能在00:01:00分时看到00:00:00-00:00:59的数据
    """
    shifted_data = raw_data.shift(shift_num)
    return resample_data_2_freq(raw_data, resample_time)


"""
输出结果再shift n个时间片，作为未来预测数据的真值
"""
def resample_and_shift(raw_data: pd.DataFrame, resample_time, delay_num):
    resampled = resample_data_2_freq(raw_data, resample_time)
    return resampled.shift(delay_num)

def resample_data_2_freq(raw_data: pd.DataFrame, resample_time):
    resampled_data = raw_data.set_index('open_time')
    # 定义不同的聚合函数
    aggregation_functions = {
        'open': 'first',  # 第一个值
        'high': 'max',  # 最大值
        'low': 'min',  # 最小值
        'close': 'last',  # 最后一个值
        'volume': 'sum',  # 求和
        'quote_volume': 'sum',  # 求和
        'trade_num': 'sum',  # 求和
        'buyer_volume': 'sum',  # 求和
        'buyer_quote_volume': 'sum',  # 求和
        'close_time': 'max'  # 最大值
    }
    # 使用resample对数据进行重采样，并应用不同的聚合函数
    resampled_data = resampled_data.resample(resample_time).agg(aggregation_functions)
    return resampled_data


def draw_lines(df: pd.DataFrame):
    # 绘制折线图
    plt.figure(figsize=(14, 7))  # 设置图形大小
    df.plot(kind='line')  # 直接使用 plot 方法绘制所有列的折线图
    # 添加标题和标签
    plt.title('Time Series Data')
    plt.xlabel('Time')
    plt.ylabel('Value')
    # 显示图表
    plt.show()
    plt.savefig('time_series_plot.png', dpi=300)


def create_etf_value(stocks_dict, window):
    """
    创建一个合成ETF价值的时间序列。

    参数:
        stocks (list of DataFrame): 每个元素是一个DataFrame，包含至少'quote_volume'和其他数值列。
        window (int): 移动平均的窗口大小。

    返回:
        DataFrame: 合成ETF的价值序列。
    """
    # 初始化一个空DataFrame来存储所有股票的移动平均成交额
    all_volumes = pd.DataFrame()

    # 获取所有股票的数值列名
    numeric_columns = None
    for stock_name, stock in stocks_dict.items():
        # 如果是第一次循环，记录数值列名
        if numeric_columns is None:
            numeric_columns = stock.select_dtypes(include=[np.floating]).columns.tolist()

        # 计算每个股票每日成交额的移动平均，并保存到all_volumes中
        if 'quote_volume' in stock.columns:
            ma = stock['quote_volume'].rolling(window).mean()
            ma.bfill(inplace=True)
            # 使用前向填充处理移动平均值中的NaN
            all_volumes[stock_name] = ma

    # 计算总成交额
    total_volume = all_volumes.sum(axis=1)

    # 计算每个股票的权重
    weights = all_volumes.div(total_volume, axis=0)
    weights.bfill(inplace=True)

    etf_value = pd.DataFrame(index=weights.index)
    for col in numeric_columns:
        if col != 'quote_volume':
            # 对于每个数值列，计算加权和
            weighted_sum = sum(stocks_dict[stock_name][col] * weights[stock_name] for stock_name in stocks_dict)
            etf_value[col] = weighted_sum

    return etf_value


def normalize_data(data, fit_scaler=False, scaler=None):
    """
    对数据进行归一化。

    参数:
        data (DataFrame): 要归一化的数据。
        fit_scaler (bool): 是否需要拟合缩放器。如果是 False，则使用传入的缩放器。
        scaler (MinMaxScaler): 已经拟合的缩放器。

    返回:
        DataFrame: 归一化后的数据。
        MinMaxScaler: 拟合后的缩放器。
    """
    # 创建 MinMaxScaler 实例
    if not fit_scaler:
        if scaler is None:
            raise ValueError("Scaler must be provided when fit_scaler is False.")
        else:
            normalized_data = pd.DataFrame(scaler.transform(data), columns=data.columns, index=data.index)
    else:
        scaler = MinMaxScaler()
        normalized_data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns, index=data.index)

    return normalized_data, scaler


def cut_data_4_training(full_data: pd.DataFrame, training=0.85, validation=0.1):
    # 划分训练集、验证集和测试集
    # 现在的算法好像不需要？
    total_length = len(full_data)
    train_size = int(total_length * training)  # 85% 用于训练
    val_size = int(total_length * validation)  # 10% 用于验证
    test_size = total_length - train_size - val_size  # 剩余的 5% 用于测试

    # 分割数据
    train_data = full_data.iloc[:train_size]
    val_data = full_data.iloc[train_size:train_size + val_size]
    test_data = full_data.iloc[train_size + val_size:]
    return train_data, val_data, test_data


def test():
    from data_loader import data_reader
    symbol, start_year_month, end_year_month = 'CAKEUSDT', "2022-05","2023-05"
    import os
    os.chdir("/")
    raw_data = data_reader.read_data_by_month_range(symbol, start_year_month, end_year_month)

    symbol2 = 'VTHOUSDT'
    raw_data2 = data_reader.read_data_by_month_range(symbol2, start_year_month, end_year_month)
    symbol3 = "ACAUSDT"
    raw_data3 = data_reader.read_data_by_month_range(symbol3, start_year_month, end_year_month)
    stocks_dict = {symbol: raw_data, symbol2: raw_data2, symbol3: raw_data3}
    window = 60 * 24
    benchmark = create_etf_value(stocks_dict, window)