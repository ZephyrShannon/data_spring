# split_deals_by_hour.py
# 正确版本：文件名为 {Market-Type}_{YYYYMMDD}_{HH}.csv
# 例如：BTC_USDT_20230425_04.csv
import numpy as np
import pandas as pd
from pathlib import Path
import gzip
from typing import Generator

def float_to_us_timestamp(ts: float) -> int:
    """
    将浮点秒时间戳转为整数微秒（用于高精度 datetime 解析）
    示例: 1680307201.460528 → 1680307201460528
    """
    return int(ts * 1_000_000)

def parse_timestamp_hour(ts: float) -> str:
    """
    将浮点时间戳转为 YYYYMMDD_HH（保留微秒精度）
    """
    ts_us = float_to_us_timestamp(ts)
    dt = pd.to_datetime(ts_us, unit='us')
    return dt.strftime('%Y%m%d_%H')

def parse_timestamp_date(ts: float) -> str:
    """
    → YYYYMMDD
    """
    ts_us = float_to_us_timestamp(ts)
    dt = pd.to_datetime(ts_us, unit='us')
    return dt.strftime('%Y%m%d')

def read_deals(file_path: Path) -> pd.DataFrame:
    """
    直接读取整个 deal 文件，返回一个 DataFrame
    支持 .csv 和 .csv.gz 格式
    自动处理 header、缺失值、类型转换
    """
    print(f"📊 正在加载文件: {file_path}")

    # 自动检测压缩
    compression = 'gzip' if str(file_path).endswith('.gz') else None

    # 尝试读取（跳过格式错误行）
    try:
        df = pd.read_csv(
            file_path,
            sep=',',
            header=None,
            names=['Timestamp', 'Deal_id', 'Price', 'Volume', 'Side'],
            dtype='string',  # 全部作为 string 读入，避免 early conversion error
            compression=compression,
            on_bad_lines='skip',  # 跳过格式错误的行
            low_memory=False
        )
    except Exception as e:
        raise RuntimeError(f"❌ 无法读取文件 {file_path}: {e}")

    if df.empty:
        print("⚠️  警告：文件为空或全部为坏行")
        return pd.DataFrame(columns=['Timestamp', 'Deal_id', 'Price', 'Volume', 'Side'])


    # 3. 类型转换（带容错）
    df['Timestamp_us'] = (pd.to_numeric(df['Timestamp'], errors='coerce') * 1e6).astype(np.int64)
    return df


def extract_market_type(file_path: Path) -> str:
    """
    从文件名提取 Market-Type
    示例：
        BTC_USDT_202304.csv.gz → BTC_USDT
        BTC_USDT_202304.csv → BTC_USDT
    """
    stem = file_path.stem  # 去掉 .gz 或 .csv
    parts = stem.split('-')
    if len(parts) == 2:
        return parts[0]
    raise ValueError(f"无法解析 Market-Type: {file_path}")

def preprocess_monthly_deals(monthly_file: Path, output_root: Path = None):
    """
    修复时间切分问题：确保每个小时的数据完整，边界正确
    """
    print(f"🔍 处理月级 deal 文件: {monthly_file}")

    market_type = extract_market_type(monthly_file)
    if not market_type:
        return

    if output_root is None:
        output_root = monthly_file.parent / "deals"

    # ==============================
    # 1. 加载数据（高精度）
    # ==============================
    df = read_deals(monthly_file)
    if df.empty:
        print("❌ 没有有效数据")
        return

    # 转为 datetime（微秒精度）
    df['datetime'] = pd.to_datetime(df['Timestamp_us'], unit='us', utc=True)

    # ==============================
    # 2. 确定时间范围 → 按小时对齐
    # ==============================
    min_dt = df['datetime'].min().floor('h')  # 向下取整到小时
    max_dt = df['datetime'].max().ceil('h')   # 向上取整到小时

    # 生成每小时的时间窗口 [start, end)
    hourly_bins = pd.date_range(start=min_dt, end=max_dt, freq='h', name='hour_start')
    print(f"⏳ 时间范围: [{min_dt} 到 {max_dt})")

    written_files = 0

    # ==============================
    # 3. 按每小时窗口切分（关键修复）
    # ==============================
    for i in range(len(hourly_bins) - 1):
        start_time = hourly_bins[i]          # YYYY-MM-DD HH:00:00
        end_time = hourly_bins[i + 1]        # YYYY-MM-DD (HH+1):00:00

        # 提取该小时内的数据 [start, end)
        mask = (df['datetime'] >= start_time) & (df['datetime'] <= end_time)
        hour_chunk = df[mask]

        if hour_chunk.empty:
            # 可选：跳过空文件，或创建空文件
            # print(f"🟡 跳过空小时: {start_time}")
            continue
        # 格式化文件名
        date_str = start_time.strftime('%Y%m%d')
        hour_key = start_time.strftime('%Y%m%d%H')  # e.g., 20230401_00

        date_dir = output_root / date_str
        date_dir.mkdir(parents=True, exist_ok=True)

        filepath = date_dir / f"{market_type}-{hour_key}.csv"

        # 保存原始字段，保留 6 位小数
        cols = ['Timestamp', 'Deal_id', 'Price', 'Volume', 'Side']
        hour_chunk[cols].to_csv(filepath, index=False, sep=',', float_format='%.6f')

        written_files += 1

    print(f"🎉 预处理完成！生成 {written_files} 个文件")

# ========================
# 主函数
# ========================
def main():
    import sys
    if len(sys.argv) < 2:
        print("📌 用法: python split_deals_by_hour.py <monthly_deals_file> [output_root]")
        print("示例: python split_deals_by_hour.py BTC_USDT_202304.csv.gz ./deals")
        sys.exit(1)

    monthly_file = Path(sys.argv[1])
    output_root = Path(sys.argv[2]) if len(sys.argv) > 2 else None

    if not monthly_file.exists():
        print(f"❌ 文件不存在: {monthly_file}")
        sys.exit(1)

    preprocess_monthly_deals(monthly_file, output_root)

if __name__ == '__main__':
    main()
    pass


def test():
    monthly_file = Path("/Users/zephyr/Documents/code/python/TestEnv/data/spot/deals/202304/BTC_USDT-202304.csv.gz")
    output_root = Path("/Users/zephyr/Documents/code/python/TestEnv/data/spot/deals/202304/")
    preprocess_monthly_deals(monthly_file, output_root)
