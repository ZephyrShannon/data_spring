'''
 权重币是领先性指标，一旦其涨或跌，其他股票也会跟随涨跌。
 这里我们就是要遍历所有股票，找到有这种跟随性的数字货币组合
 注意: 这种跟随性会随着时间切片长度的不同而发生不同的变化，
      1、大跨度切片的跟随性是大的波浪，并且我们希望它有比较恒定的beta值，
      2、小跨度的切片则是小的波浪，
      我们要查找从大跨度来说具有稳定的跟随性和beta值的币对，在小跨度层面发现跟随性脱钩，beta值变化时，我们认为就具有对冲套利的可能性了
'''
import os
from abc import ABC, abstractmethod
import pandas as pd

class LeaderAndFollower(ABC):
    """
     @param benchmarks 基准，大约20 种
     @param followers 跟随者 大约500种
     @param window_sizes 窗口长列表 15m、60m、3h、12h、2d长度 5 种
     @param delay_epochs 延迟时间  1m  2m 5m 15m 30m 1h 2h 3h 6h 12h 1d 2d 大约12种
    """
    def __init__(self, benchmarks: dict, followers: dict, window_sizes: list, delay_epochs: list, save_dir: str):
        self.benchmarks = benchmarks
        self.followers = followers
        self.window_sizes = window_sizes
        self.delay_epochs = delay_epochs
        self.save_dir = save_dir

    @abstractmethod
    def find_best_models(self):
        # 找到最好的模型.
        pass


class OneToOneFollow(LeaderAndFollower):
    def __init__(self, benchmarks: dict, followers: dict, window_sizes: list, delay_epochs: list):
        super().__init__(benchmarks, followers, window_sizes, delay_epochs)

    def find_best_models(self):
        """
        遍历所有benchmarks 和 followers，和 delay
        每一个目录对应着一个benchmark，
        每一个大图，对应着一个follower的线图，展示一个时间范围内的预测效果的变化
        每一个子图，对应着一个windows size
        每一条线，对应着一个delay
        这样就能一眼看出每一个benchmark下预测效果最好的股票了。
        然后需要考虑预测效果s
        :return:
        """
        for key, benchmark in self.benchmarks.items():
            os.makedirs(os.path.join(self.save_dir, key), exist_ok=True)
            for follower in self.followers:
                for window_size in self.window_sizes:
                    for delay_epoch in self.delay_epochs:
                        try:
                            power = self.calc_delay_prediction_power(benchmark, follower, window_size, delay_epoch)
                        except Exception as e:
                            import traceback
                            print(traceback.format_exc())

    def calc_delay_prediction_power(self, benchmark, follower, window_size, delay_epoch) -> pd.tseries:
        '''
        计算跟随者相关性
        :param benchmark:
        :param follower:
        :param window_size:
        :param delay_epoch:
        :return:
        '''

        return None


def calculate_rolling_correlation(benchmark, follower, delay, window_size):
    """
    计算两个DataFrame Da和Db之间的时间序列相关性。

    参数:
    benchmark (pd.DataFrame): 第一个DataFrame，包含时间序列数据。
    follower (pd.DataFrame): 第二个DataFrame，包含时间序列数据。
    delay (int): Db相对于Da的延迟时间片数量。
    window_size (int): 计算相关性的滑动窗口长度。
    返回:
    pd.DataFrame: 包含每个时间点上Da和Db之间相关性的DataFrame。
    """
    # 确保DataFrame有时间索引
    if not isinstance(benchmark.index, pd.DatetimeIndex):
        raise ValueError("DataFrame should have a DatetimeIndex.")

    # 将follower向后延迟delay个时间片
    follower_lagged = follower.shift(delay)

    # 初始化一个空DataFrame来存储相关性结果
    correlation_results = pd.DataFrame(index=benchmark.index)

    # 遍历每个列并计算相关性
    for column in benchmark.columns:
        # 计算滑动窗口内的相关性
        rolling_corr = benchmark[column].rolling(window=window_size).corr(follower_lagged[column])

        # 填充相关性结果DataFrame
        correlation_results[f'corr_{column}'] = rolling_corr

    return correlation_results

def test():
    # 示例数据
