import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict
import warnings

warnings.filterwarnings('ignore')


def sim_trade(kline: pd.DataFrame, long_sign_df: pd.DataFrame, short_sign_df: pd.DataFrame,
              long_prob: pd.DataFrame, short_prob: pd.DataFrame,
              trans_fee: float = 0.0) -> Tuple[float, pd.DataFrame, pd.DataFrame]:
    """
    简单的模拟交易系统

    :param kline: k线数据，需要有两列['Open','Close']，索引为时间
    :param long_sign_df: 多开信号（0/1）['Long_sign']
    :param short_sign_df: 空开信号（0/1）['Short_sign']
    :param long_prob: 多开信号概率 ['Long_prob']
    :param short_prob: 空开信号概率 ['Short_prob']
    :param trans_fee: 交易费率（双边，例如0.0002表示0.02%）
    :return: 
        1. 全时长总的收益率
        2. DataFrame：[开仓时间, 开仓价格, 开仓信号概率, 平仓时间, 平仓价格, 平仓信号概率, 持仓时长（分钟）, 持仓损益, 收益率]
        3. DataFrame：[开仓时间, 开仓价格, 开仓信号概率, 平仓时间, 平仓价格, 平仓信号概率, 持仓时长（分钟）, 持仓损益, 收益率]
    """

    # 1. 数据预处理和验证
    kline = kline.copy()
    long_sign_df = long_sign_df.copy()
    short_sign_df = short_sign_df.copy()
    long_prob = long_prob.copy()
    short_prob = short_prob.copy()

    # 确保所有DataFrame索引一致
    all_indices = kline.index
    long_sign_df = long_sign_df.reindex(all_indices).fillna(0)
    short_sign_df = short_sign_df.reindex(all_indices).fillna(0)
    long_prob = long_prob.reindex(all_indices).fillna(0)
    short_prob = short_prob.reindex(all_indices).fillna(0)

    # 验证列名
    required_kline_cols = ['Open', 'Close']
    required_signal_cols = ['Long_sign', 'Short_sign']
    required_prob_cols = ['Long_prob', 'Short_prob']

    if not all(col in kline.columns for col in required_kline_cols):
        raise ValueError(f"kline必须包含列: {required_kline_cols}")

    if 'signal' not in long_sign_df.columns:
        raise ValueError("long_sign必须包含'Long_sign'列")
    if 'signal' not in short_sign_df.columns:
        raise ValueError("short_sign必须包含'Short_sign'列")
    if 'Long_prob' not in long_prob.columns:
        raise ValueError("long_prob必须包含'Long_prob'列")
    if 'Short_prob' not in short_prob.columns:
        raise ValueError("short_prob必须包含'Short_prob'列")

    # 2. 模拟交易逻辑
    def simulate_trades(signal_series: pd.Series, prob_series: pd.Series,
                        position_type: str) -> pd.DataFrame:
        """
        模拟单方向交易

        :param signal_series: 信号序列 (0/1)
        :param prob_series: 概率序列
        :param position_type: 'long' 或 'short'
        :return: 交易记录DataFrame
        """
        trades = []
        in_position = False
        entry_time = None
        entry_price = None
        entry_prob = None

        for i, (time, signal) in enumerate(signal_series.items()):
            close_price = kline.loc[time, 'Close']
            current_prob = prob_series.loc[time]

            if not in_position and signal == 1:
                # 开仓
                in_position = True
                entry_time = time
                entry_price = close_price
                entry_prob = current_prob

            elif in_position and signal == 0:
                # 平仓
                exit_time = time
                exit_price = close_price
                exit_prob = current_prob

                # 计算持仓时长（分钟）
                if isinstance(entry_time, pd.Timestamp) and isinstance(exit_time, pd.Timestamp):
                    duration_minutes = (exit_time - entry_time).total_seconds() / 60
                else:
                    duration_minutes = i  # 如果时间不是Timestamp，用索引差

                # 计算损益
                if position_type == 'long':
                    # 做多：平仓价 - 开仓价
                    pnl = exit_price - entry_price
                else:  # short
                    # 做空：开仓价 - 平仓价
                    pnl = entry_price - exit_price

                # 扣除交易费用（双边）
                fee = (entry_price + exit_price) * trans_fee
                net_pnl = pnl - fee

                # 计算收益率
                return_rate = (net_pnl / entry_price) * 100 if entry_price != 0 else 0

                trades.append({
                    'entry_time': entry_time,
                    'entry_price': entry_price,
                    'entry_prob': entry_prob,
                    'exit_time': exit_time,
                    'exit_price': exit_price,
                    'exit_prob': exit_prob,
                    'duration_minutes': duration_minutes,
                    'pnl': net_pnl,
                    'return_rate': return_rate
                })

                # 重置状态
                in_position = False
                entry_time = None
                entry_price = None
                entry_prob = None

        # 如果最后还有持仓，强制平仓
        if in_position:
            exit_time = signal_series.index[-1]
            exit_price = kline.loc[exit_time, 'Close']
            exit_prob = prob_series.loc[exit_time]

            if isinstance(entry_time, pd.Timestamp) and isinstance(exit_time, pd.Timestamp):
                duration_minutes = (exit_time - entry_time).total_seconds() / 60
            else:
                duration_minutes = len(signal_series) - 1

            if position_type == 'long':
                pnl = exit_price - entry_price
            else:
                pnl = entry_price - exit_price

            fee = (entry_price + exit_price) * trans_fee
            net_pnl = pnl - fee
            return_rate = (net_pnl / entry_price) * 100 if entry_price != 0 else 0

            trades.append({
                'entry_time': entry_time,
                'entry_price': entry_price,
                'entry_prob': entry_prob,
                'exit_time': exit_time,
                'exit_price': exit_price,
                'exit_prob': exit_prob,
                'duration_minutes': duration_minutes,
                'pnl': net_pnl,
                'return_rate': return_rate
            })

        return pd.DataFrame(trades)

    # 3. 执行模拟交易
    long_trades = simulate_trades(long_sign_df['signal'], long_prob['Long_prob'], 'long')
    short_trades = simulate_trades(short_sign_df['signal'], short_prob['Short_prob'], 'short')

    # 4. 计算总收益率
    total_investment = 0
    total_pnl = 0

    if not long_trades.empty:
        total_pnl += long_trades['pnl'].sum()
        total_investment += long_trades['entry_price'].sum()

    if not short_trades.empty:
        total_pnl += short_trades['pnl'].sum()
        total_investment += short_trades['entry_price'].sum()

    if total_investment > 0:
        total_return_rate = (total_pnl / total_investment) * 100
    else:
        total_return_rate = 0.0

    return total_return_rate, long_trades, short_trades


# ==================== 辅助函数：计算交易统计指标 ====================

def calculate_trade_stats(trades_df: pd.DataFrame, position_type: str) -> Dict:
    """
    计算交易统计指标

    :param trades_df: 交易记录DataFrame
    :param position_type: 'long' 或 'short'
    :return: 统计指标字典
    """
    if trades_df.empty:
        return {
            'total_trades': 0,
            'win_rate': 0,
            'avg_return': 0,
            'avg_duration': 0,
            'total_pnl': 0,
            'max_win': 0,
            'max_loss': 0,
            'sharpe_ratio': 0,
            'profit_factor': 0
        }

    stats = {
        'total_trades': len(trades_df),
        'win_trades': len(trades_df[trades_df['pnl'] > 0]),
        'lose_trades': len(trades_df[trades_df['pnl'] < 0]),
        'even_trades': len(trades_df[trades_df['pnl'] == 0]),
        'total_pnl': trades_df['pnl'].sum(),
        'avg_pnl': trades_df['pnl'].mean(),
        'avg_return': trades_df['return_rate'].mean(),
        'avg_duration': trades_df['duration_minutes'].mean(),
        'max_win': trades_df['pnl'].max(),
        'max_loss': trades_df['pnl'].min(),
        'std_return': trades_df['return_rate'].std(),
    }

    # 胜率
    stats['win_rate'] = (stats['win_trades'] / stats['total_trades'] * 100) if stats['total_trades'] > 0 else 0

    # 夏普比率（简化版）
    if stats['std_return'] > 0:
        stats['sharpe_ratio'] = stats['avg_return'] / stats['std_return']
    else:
        stats['sharpe_ratio'] = 0

    # 盈亏比（Profit Factor）
    total_win = trades_df[trades_df['pnl'] > 0]['pnl'].sum()
    total_loss = abs(trades_df[trades_df['pnl'] < 0]['pnl'].sum())
    stats['profit_factor'] = total_win / total_loss if total_loss > 0 else float('inf')

    return stats


def plot_trade_results(kline: pd.DataFrame, long_trades: pd.DataFrame,
                       short_trades: pd.DataFrame, title: str = "交易模拟结果"):
    """
    可视化交易结果

    :param kline: K线数据
    :param long_trades: 多头交易记录
    :param short_trades: 空头交易记录
    :param title: 图表标题
    """
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), gridspec_kw={'height_ratios': [3, 1]})

    # 绘制价格曲线
    ax1.plot(kline.index, kline['Close'], label='收盘价', color='black', alpha=0.7, linewidth=1)

    # 绘制多头交易
    if not long_trades.empty:
        for _, trade in long_trades.iterrows():
            entry_time = trade['entry_time']
            exit_time = trade['exit_time']
            entry_price = trade['entry_price']
            exit_price = trade['exit_price']

            # 绘制交易线段
            color = 'green' if trade['pnl'] > 0 else 'red'
            ax1.plot([entry_time, exit_time], [entry_price, exit_price],
                     color=color, linewidth=2, alpha=0.7)

            # 标记开仓点
            ax1.scatter(entry_time, entry_price, color='green', marker='^', s=100,
                        label='多开' if _ == 0 else "")
            # 标记平仓点
            ax1.scatter(exit_time, exit_price, color='red', marker='v', s=100,
                        label='平仓' if _ == 0 else "")

    # 绘制空头交易
    if not short_trades.empty:
        for _, trade in short_trades.iterrows():
            entry_time = trade['entry_time']
            exit_time = trade['exit_time']
            entry_price = trade['entry_price']
            exit_price = trade['exit_price']

            # 绘制交易线段
            color = 'blue' if trade['pnl'] > 0 else 'orange'
            ax1.plot([entry_time, exit_time], [entry_price, exit_price],
                     color=color, linewidth=2, alpha=0.7, linestyle='--')

            # 标记开仓点
            ax1.scatter(entry_time, entry_price, color='blue', marker='v', s=100,
                        label='空开' if len(long_trades) == 0 and _ == 0 else "")

    ax1.set_title(title, fontsize=14)
    ax1.set_ylabel('价格', fontsize=12)
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)

    # 格式化x轴
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)

    # 绘制累积收益率曲线
    if not long_trades.empty or not short_trades.empty:
        # 合并所有交易
        all_trades = pd.concat([long_trades, short_trades], ignore_index=True)
        all_trades = all_trades.sort_values('exit_time')

        # 计算累积收益率
        cumulative_returns = [0]
        cumulative_value = [100]  # 起始资金100
        times = [kline.index[0]]

        for _, trade in all_trades.iterrows():
            return_rate = trade['return_rate']
            cumulative_returns.append(cumulative_returns[-1] + return_rate)
            cumulative_value.append(cumulative_value[-1] * (1 + return_rate / 100))
            times.append(trade['exit_time'])

        ax2.plot(times, cumulative_returns, label='累积收益率(%)', color='purple', linewidth=2)
        ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax2.set_ylabel('累积收益率 (%)', fontsize=12)
        ax2.set_xlabel('时间', fontsize=12)
        ax2.legend(loc='upper left')
        ax2.grid(True, alpha=0.3)

        # 格式化x轴
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)

    plt.tight_layout()
    plt.show()


def generate_sample_data(num_points: int = 1000) -> tuple:
    """
    生成示例数据用于测试

    :param num_points: 数据点数
    :return: (kline, long_sign, short_sign, long_prob, short_prob)
    """
    # 生成时间序列
    dates = pd.date_range(start='2024-01-01', periods=num_points, freq='1min')

    # 生成随机价格（模拟市场波动）
    np.random.seed(42)
    base_price = 100
    returns = np.random.normal(0.0001, 0.01, num_points)  # 日化波动率约1%
    price = base_price * np.exp(np.cumsum(returns))

    # 创建K线数据
    kline = pd.DataFrame({
        'Open': price * (1 + np.random.normal(0, 0.001, num_points)),
        'Close': price
    }, index=dates)

    # 生成随机信号（示例逻辑：价格上穿/下穿移动平均线）
    ma_short = pd.Series(price).rolling(window=20).mean()
    ma_long = pd.Series(price).rolling(window=50).mean()

    # 多头信号：短线上穿长线
    long_signal = (ma_short > ma_long) & (ma_short.shift(1) <= ma_long.shift(1))
    # 空头信号：短线下穿长线
    short_signal = (ma_short < ma_long) & (ma_short.shift(1) >= ma_long.shift(1))

    # 生成信号概率（基于价格与均线的距离）
    price_series = pd.Series(price, index=dates)
    long_prob_values = np.clip((price_series - ma_long) / ma_long * 10, 0, 1)
    short_prob_values = np.clip((ma_long - price_series) / ma_long * 10, 0, 1)

    # 创建信号DataFrame
    long_sign = pd.DataFrame({'Long_sign': long_signal.astype(int)}, index=dates).fillna(0)
    short_sign = pd.DataFrame({'Short_sign': short_signal.astype(int)}, index=dates).fillna(0)
    long_prob = pd.DataFrame({'Long_prob': long_prob_values}, index=dates).fillna(0)
    short_prob = pd.DataFrame({'Short_prob': short_prob_values}, index=dates).fillna(0)

    return kline, long_sign, short_sign, long_prob, short_prob


# ==================== 使用示例 ====================

def example_usage():
    """使用示例"""
    print("生成示例数据...")
    kline, long_sign, short_sign, long_prob, short_prob = generate_sample_data(500)

    print("运行模拟交易...")
    total_return, long_trades, short_trades = sim_trade(
        kline=kline,
        long_sign_df=long_sign,
        short_sign_df=short_sign,
        long_prob=long_prob,
        short_prob=short_prob,
        trans_fee=0.0002  # 0.02%交易费率
    )

    print(f"\n{'=' * 60}")
    print(f"总收益率: {total_return:.2f}%")
    print(f"{'=' * 60}\n")

    # 计算并显示交易统计
    if not long_trades.empty:
        long_stats = calculate_trade_stats(long_trades, 'long')
        print("多头交易统计:")
        print(f"  交易次数: {long_stats['total_trades']}")
        print(f"  胜率: {long_stats['win_rate']:.1f}%")
        print(f"  平均收益率: {long_stats['avg_return']:.2f}%")
        print(f"  平均持仓时间: {long_stats['avg_duration']:.1f}分钟")
        print(f"  总盈亏: {long_stats['total_pnl']:.2f}")
        print(f"  盈亏比: {long_stats['profit_factor']:.2f}")
        print(f"  夏普比率: {long_stats['sharpe_ratio']:.2f}")
        print()

    if not short_trades.empty:
        short_stats = calculate_trade_stats(short_trades, 'short')
        print("空头交易统计:")
        print(f"  交易次数: {short_stats['total_trades']}")
        print(f"  胜率: {short_stats['win_rate']:.1f}%")
        print(f"  平均收益率: {short_stats['avg_return']:.2f}%")
        print(f"  平均持仓时间: {short_stats['avg_duration']:.1f}分钟")
        print(f"  总盈亏: {short_stats['total_pnl']:.2f}")
        print(f"  盈亏比: {short_stats['profit_factor']:.2f}")
        print(f"  夏普比率: {short_stats['sharpe_ratio']:.2f}")
        print()

    # 显示前几笔交易记录
    print("前5笔多头交易记录:")
    if not long_trades.empty:
        print(long_trades.head().to_string())
    else:
        print("无多头交易")

    print("\n前5笔空头交易记录:")
    if not short_trades.empty:
        print(short_trades.head().to_string())
    else:
        print("无空头交易")

    # 绘制交易图表
    print("\n绘制交易图表...")
    plot_trade_results(kline, long_trades, short_trades, title="示例交易模拟")

    return total_return, long_trades, short_trades


# ==================== 高级功能：回测分析 ====================

def backtest_analysis(kline: pd.DataFrame, long_trades: pd.DataFrame,
                      short_trades: pd.DataFrame) -> Dict:
    """
    回测分析

    :return: 分析结果字典
    """
    analysis = {}

    # 合并所有交易
    all_trades = pd.concat([long_trades, short_trades], ignore_index=True)

    if all_trades.empty:
        return analysis

    # 按时间排序
    all_trades = all_trades.sort_values('exit_time')

    # 计算资金曲线
    initial_capital = 10000  # 初始资金
    capital = initial_capital
    capital_curve = [capital]
    time_points = [kline.index[0]]

    for _, trade in all_trades.iterrows():
        # 假设每次交易使用固定比例的资金
        position_value = capital * 0.1  # 每次使用10%的资金
        return_rate = trade['return_rate'] / 100  # 转换为小数

        # 更新资金
        capital += position_value * return_rate
        capital_curve.append(capital)
        time_points.append(trade['exit_time'])

    analysis['capital_curve'] = pd.Series(capital_curve, index=time_points)
    analysis['final_capital'] = capital
    analysis['total_return_pct'] = (capital / initial_capital - 1) * 100

    # 计算最大回撤
    cumulative_max = analysis['capital_curve'].cummax()
    drawdown = (analysis['capital_curve'] - cumulative_max) / cumulative_max
    analysis['max_drawdown'] = drawdown.min() * 100  # 百分比

    # 计算年化收益率
    time_span = (analysis['capital_curve'].index[-1] - analysis['capital_curve'].index[0]).days
    years = max(time_span / 365, 1 / 365)  # 至少一天
    analysis['annualized_return'] = ((1 + analysis['total_return_pct'] / 100) ** (1 / years) - 1) * 100

    return analysis

