import gzip

import numpy as np
import pandas as pd
import torch
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from typing import List, Tuple
import pickle
from data_downloader.create_hf_label import build_filepath, load_spot_ticks_and_renames
import os
data_prefix = "./data"
markets = ["BTC_USDT", "ETH_USDT"]
low_freq_data_types = ["candlesticks_1m", "candleSticks_5m"]

# =====================================================
# 用户提供的函数（假设已实现）
# =====================================================
def has_get_high_freq_data(date_and_hour: datetime) -> bool:
    for market in markets:
        file_path = build_filepath(data_prefix, biz = 'spot', data_type = 'ticks', market=market, dt =date_and_hour)
        if not os.path.exists(file_path):
            return False
    """判断某小时是否有数据"""
    return True


def get_high_freq_data(date_and_hour: datetime) -> np.ndarray:
    """
    返回 (N, 163) 数组，第0列是 timestamp，其余162个是特征
    """
    all_datas = []
    for market in markets:
        d = load_spot_ticks_and_renames(data_prefix, market, date_and_hour)
        d = d.set_index('Timestamp')
        renames = {}
        m_name = market.split('_')[0]
        for col in d.columns:
            renames[col] = f"{m_name}_{col}"
        all_datas.append(d.rename(renames, axis=1))
    return pd.concat(all_datas, axis=1)


def get_low_freq_data(date_and_hour: datetime) -> np.ndarray:
    """
    date_and_hour：整小时点时间
    """
    kline_type1 = "candlesticks_5m" # 最近6 小时
    kline_type2 = "candlesticks_1h" # 6-24 小时
    kline_type3 = "candlesticks_1d" # 最近15天
    COLUMNS = ["Timestamp",	"Volume", "Low", "Open"]
    all_datas = []
    for market in low_freq_data_types:
        kline1_file_path = build_filepath(data_prefix, "spot", kline_type1, market, date_and_hour)
        with gzip.open(kline1_file_path, 'rt') as f:
            df3 = pd.read_csv(f, header=None, names=COLUMNS)
        kline2_file_path = build_filepath(data_prefix, "spot", kline_type2, market, date_and_hour)
        with gzip.open(kline2_file_path, 'rt') as f:
            df2 = pd.read_csv(f, header=None, names=COLUMNS)
        kline3_file_path = build_filepath(data_prefix, "spot", kline_type3, market, date_and_hour)
        with gzip.open(kline3_file_path, 'rt') as f:
            df3 = pd.read_csv(f, header=None, names=COLUMNS)
        df1 = df1.set_index("Timestamp", axis=1)
        df2 = df2.set_index("Timestamp", axis=1)
        df3 = df3.set_index("Timestamp", axis=1)
        # TODO 生成新的dataframe，新的dataframe包括12行，分别是整点分钟+5倍数的分钟，每一行包括
        #  1、df1 （5m频率数据）中对应分钟及其后的12 * 6 个数据点数据，需要Volume、Low、Hig
        #  2、df2 （1h频率数据）中对应小时及其后的18个数据点数据，需要Volume、Low、Hig
        #  3、df3 （1d频率数据）中对应天及其后15个时间点数据，，需要Volume、Low、Hig



def get_labels_data(date_and_hour: datetime) -> np.ndarray:
    """
    返回 (N, K) 数组，K 个回归标签（如：未来3/5/10秒收益率、风险率等）
    必须与高频数据的 timestamp 对齐
    """
    with gzip.open(file_path, 'rt') as f:
        df = pd.read_csv(f, header=None, names=COLUMNS)
    return df, freq
    pass


# =====================================================
# 主类：TimeSeriesDataBuilder（定制版）
# =====================================================
class TimeSeriesDataBuilder:
    """
    针对你的数据结构定制：
    - 高频：(N, 163)，含 timestamp
    - 低频：(5,)，含 timestamp
    - 标签：(N, K)，多目标回归
    - 按小时切分，自动跳过缺失
    """

    def __init__(self, seq_len: int = 100):
        """
        :param seq_len: 滑动窗口长度（单位：秒）
        """
        self.seq_len = seq_len
        self.scaler_high = StandardScaler()
        self.scaler_low = StandardScaler()

        # 用于归一化的数据收集
        self.all_high_features = []  # 收集所有高频特征（不含 timestamp）
        self.all_low_features = []  # 收集所有低频特征（不含 timestamp）

        # 最终样本
        self.X_high_list = []  # (N, seq_len, 162)
        self.X_low_list = []  # (N, 4)
        self.y_list = []  # (N, K)

    def scan_continuous_segments(self, start_dt: datetime, end_dt: datetime) -> List[List[datetime]]:
        """扫描连续时间段（按小时）"""
        current_dt = start_dt.replace(minute=0, second=0, microsecond=0)
        end_dt = end_dt.replace(minute=0, second=0, microsecond=0)

        segments = []
        current_segment = []

        while current_dt <= end_dt:
            if has_get_high_freq_data(current_dt):
                current_segment.append(current_dt)
            else:
                if current_segment:
                    segments.append(current_segment)
                    current_segment = []
                print(f"⚠️  数据缺失：{current_dt} 小时无数据，已切分。")
            current_dt += timedelta(hours=1)

        if current_segment:
            segments.append(current_segment)

        print(f"✅ 共找到 {len(segments)} 个连续数据段。")
        for i, seg in enumerate(segments):
            print(f"    段 {i + 1}: {seg[0]} → {seg[-1]} ({len(seg)} 小时)")
        return segments

    def load_and_collect_for_scaling(self, segments: List[List[datetime]]):
        """收集数据以拟合归一化模型（去掉 timestamp）"""
        print("🔍 正在收集数据以训练归一化模型...")

        for segment in segments:
            for dt in segment:
                if not has_get_high_freq_data(dt):
                    continue

                high_data = get_high_freq_data(dt)  # (N, 163)
                low_data = get_low_freq_data(dt)  # (5,)

                # ✅ 去除 timestamp 列（第0列）
                high_features = high_data[:, 1:]  # (N, 162)
                low_features = low_data[1:]  # (4,)

                self.all_high_features.append(high_features)
                self.all_low_features.append(low_features)

        if not self.all_high_features:
            raise ValueError("❌ 未找到任何高频数据，请检查数据源。")

        # 合并并拟合 scaler
        all_high_array = np.vstack(self.all_high_features)  # (Total_T, 162)
        all_low_array = np.array(self.all_low_features)  # (Total_H, 4)

        self.scaler_high.fit(all_high_array)
        self.scaler_low.fit(all_low_array)

        print(f"✅ 归一化模型训练完成：高频 {all_high_array.shape}，低频 {all_low_array.shape}")

    def generate_samples_from_segment(self, segment: List[datetime]):
        """从连续段生成滑动窗口样本"""
        print(f"🔄 正在从 {segment[0]} → {segment[-1]} 生成样本...")

        full_high_features = []
        full_labels = []
        full_low_features = []  # 每秒对应一个低频向量（重复该小时的值）

        for dt in segment:
            if not has_get_high_freq_data(dt):
                continue

            high_data = get_high_freq_data(dt)  # (N, 163)
            labels = get_labels_data(dt)  # (N, K)
            low_data = get_low_freq_data(dt)  # (5,)

            # 提取特征（去 timestamp）
            high_features = high_data[:, 1:]  # (N, 162)
            low_features = low_data[1:]  # (4,)

            # 归一化（使用已拟合的 scaler）
            high_scaled = self.scaler_high.transform(high_features)  # (N, 162)

            full_high_features.append(high_scaled)
            full_labels.append(labels)
            # 将该小时的低频特征重复 N 次，与高频对齐
            full_low_features.extend([low_features] * len(high_data))

        if not full_high_features:
            return

        # 合并
        full_X_high = np.vstack(full_high_features)  # (Total_T, 162)
        full_y = np.vstack(full_labels)  # (Total_T, K)
        full_X_low = np.array(full_low_features)  # (Total_T, 4)

        # 滑动窗口
        for i in range(len(full_X_high) - self.seq_len):
            x_high = full_X_high[i:i + self.seq_len]  # (seq_len, 162)
            x_low = full_X_low[i + self.seq_len - 1]  # (4,) 取窗口结束时刻的低频
            y = full_y[i + self.seq_len]  # (K,) 预测 t+seq_len 时刻的标签

            self.X_high_list.append(x_high)
            self.X_low_list.append(x_low)
            self.y_list.append(y)

        print(f"✅ 从该段生成 {len(self.X_high_list)} 个样本。")

    def build(self, start_dt: datetime, end_dt: datetime) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """主入口函数"""
        segments = self.scan_continuous_segments(start_dt, end_dt)
        if not segments:
            raise ValueError("❌ 没有找到任何有效数据段。")

        self.load_and_collect_for_scaling(segments)

        for segment in segments:
            self.generate_samples_from_segment(segment)

        # 转为 Tensor
        X_high = torch.tensor(np.array(self.X_high_list), dtype=torch.float32)  # (N, 100, 162)
        X_low = torch.tensor(np.array(self.X_low_list), dtype=torch.float32)  # (N, 4)
        y = torch.tensor(np.array(self.y_list), dtype=torch.float32)  # (N, K)

        print(f"🎉 数据生成完成！共 {len(y)} 个样本。")
        print(f"   X_high: {X_high.shape}")
        print(f"   X_low:  {X_low.shape}")
        print(f"   y:      {y.shape}")

        return X_high, X_low, y

    def save_scalers(self, path_prefix: str = "scalers"):
        """保存 scaler 用于推理"""
        with open(f"{path_prefix}_high.pkl", "wb") as f:
            pickle.dump(self.scaler_high, f)
        with open(f"{path_prefix}_low.pkl", "wb") as f:
            pickle.dump(self.scaler_low, f)
        print(f"✅ Scaler 已保存至 {path_prefix}_*.pkl")

    def load_scalers(self, path_prefix: str = "scalers"):
        """加载 scaler"""
        with open(f"{path_prefix}_high.pkl", "rb") as f:
            self.scaler_high = pickle.load(f)
        with open(f"{path_prefix}_low.pkl", "rb") as f:
            self.scaler_low = pickle.load(f)
        print(f"✅ Scaler 已加载")