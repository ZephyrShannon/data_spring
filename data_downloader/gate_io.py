#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import os
import sys
from datetime import datetime, timedelta, timezone
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from typing import List
from urllib.parse import urljoin
from deal_split import preprocess_monthly_deals
import requests
import yaml
import recreate_ticks

# 全局 Session
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "GateDataProcessor/1.0"})

logger = logging.getLogger("GateDownloader")  # 日志对象


def setup_logger(log_dir: str) -> logging.Logger:
    global logger
    # 获取或创建 logger
    logger = logging.getLogger("GateDownloader")

    # 避免重复添加 handler（重要：防止重复输出）
    if logger.hasHandlers():
        logger.handlers.clear()

    logger.setLevel(logging.INFO)

    # 创建日志目录
    log_dir = Path(log_dir)
    log_dir.mkdir(exist_ok=True)

    # 1. 文件处理器：每天轮转，保留7天
    file_handler = TimedRotatingFileHandler(
        log_dir / "gate_download.log",
        when="midnight",
        interval=1,
        backupCount=7,
        encoding="utf-8"
    )
    file_handler.setLevel(logging.INFO)

    # 2. 控制台处理器（用于 print）
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # 定义日志格式
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # 添加两个处理器
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def load_config(config_path: str) -> dict:
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        print(f"配置文件加载成功: {config_path}")
        return config
    except Exception as e:
        print(f"加载配置文件失败: {e}")
        sys.exit(1)


def parse_datetime(dt_str: str) -> datetime:
    for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            return datetime.strptime(dt_str, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    raise ValueError(f"无法解析时间格式: {dt_str}")


# =============================
# 数据类型与时间粒度映射表
# =============================
TIME_GRANULARITY = {
    "deals": "monthly",
    "candlesticks_1m": "monthly",
    "candlesticks_5m": "monthly",
    "candlesticks_1h": "monthly",
    "candlesticks_4h": "monthly",
    "candlesticks_1d": "monthly",
    "trades": "monthly",
    "mark_prices": "monthly",
    "funding_applies": "monthly",
    "funding_updates": "monthly",
    "orderbooks": "hourly",           # 原始深度数据（CSV.GZ）
    "orderbooks_slice": "hourly",     # 我们生成的目标格式（JSONL）
}


def generate_time_intervals(start_utc: str, end_utc: str, granularity: str) -> List[datetime]:
    start_utc = parse_datetime(start_utc)
    end_utc = parse_datetime(end_utc)

    intervals = []

    if granularity == "hourly":
        current = start_utc.replace(minute=0, second=0, microsecond=0)
        while current < end_utc:
            intervals.append(current)
            current += timedelta(hours=1)
    elif granularity == "daily":
        current = start_utc.replace(hour=0, minute=0, second=0, microsecond=0)
        while current < end_utc:
            intervals.append(current)
            current += timedelta(days=1)
    elif granularity == "monthly":
        current = start_utc.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        while current < end_utc:
            intervals.append(current)
            if current.month == 12:
                current = current.replace(year=current.year + 1, month=1)
            else:
                current = current.replace(month=current.month + 1)
    else:
        raise ValueError(f"不支持的时间粒度: {granularity}")

    return intervals


def build_url(biz: str, data_type: str, market: str, dt: datetime) -> str:
    year = dt.strftime("%Y")
    month = dt.strftime("%m")
    day = dt.strftime("%d")
    hour = dt.strftime("%H")

    base = "https://download.gatedata.org/"
    path = f"{biz}/{data_type}/{year}{month}/"

    filename = f"{market}-{year}{month}"

    if data_type == "orderbooks_slice":
        # 注意：我们不再从服务器下载这个，而是自己生成
        filename += f"{day}{hour}.gz"
    elif data_type.startswith("candlesticks_"):
        filename += f".csv.gz"
    elif data_type == "deals" and biz == "spot":
        filename += ".csv.gz"
    elif data_type in ["trades", "mark_prices", "funding_applies", "funding_updates"]:
        filename += f".csv.gz"
    else:
        filename += f"{day}{hour}.csv.gz"

    return urljoin(base, path + filename)


def get_local_filepath(base_dir: str, url: str) -> Path:
    base = "https://download.gatedata.org/"
    rel_path = url.replace(base, "")
    return Path(base_dir) / rel_path


def check_if_downloaded(history_file: str, url: str) -> bool:
    if not os.path.exists(history_file):
        return False
    with open(history_file, 'r', encoding='utf-8') as f:
        return any(url.strip() == line.strip() for line in f)


def mark_as_downloaded(history_file: str, url: str):
    with open(history_file, 'a', encoding='utf-8') as f:
        f.write(url + "\n")


def download_file(url: str, temp_path: Path) -> bool:
    """直接下载，不支持断点续传"""
    temp_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with SESSION.get(url, stream=True, timeout=30) as resp:
            if resp.status_code != 200:
                logger.warning(f"HTTP {resp.status_code}: {url}")
                return False
            with open(temp_path, 'wb') as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        logger.info(f"下载完成: {temp_path}")
        return True
    except Exception as e:
        logger.error(f"下载失败 {url}: {e}")
        temp_path.unlink(missing_ok=True)
        return False

def check_and_create_ticks(filepath: Path, dt: datetime):
    root_parent = filepath.parent.parent.parent
    parent = root_parent / "deals"
    date_folds = f"{dt.year}{dt.month:02d}/{dt.year}{dt.month:02d}{dt.day:02d}"
    parent /= date_folds
    deal_path = parent / filepath.name[:-3]
    tick_parent = root_parent / ("ticks/" + date_folds)
    tick_name = filepath.name[:-3]
    tick_file = tick_parent / tick_name
    print(f"Recreate ticks: {filepath} + {deal_path} -> {tick_file}")
    if not tick_file.exists():
        recreate_ticks.generate_ticks(filepath, deal_path, tick_file)

def download_data(base_dir, biz_list, markets, types, start_utc, end_utc, history_file):
    logger.info(f"开始下载任务...")
    logger.info(f"时间范围: {start_utc} ~ {end_utc}")

    grouped_types = {"hourly": [], "daily": [], "monthly": []}
    for t in types:
        gran = TIME_GRANULARITY.get(t, "hourly")
        grouped_types[gran].append(t)

    for gran in ["monthly", "daily", "hourly"]:
        type_list = grouped_types[gran]
        if not type_list:
            continue
        print(f"处理 {gran} 粒度数据: {type_list}")
        logger.info(f"处理 {gran} 粒度数据: {type_list}")
        time_list = generate_time_intervals(start_utc, end_utc, gran)

        for biz in biz_list:
            for market in markets:
                for data_type in type_list:
                    for dt in time_list:
                        url = build_url(biz, data_type, market, dt)
                        filepath = get_local_filepath(base_dir, url)
                        logger.info(f"Download {url} to {filepath}")

                        # 普通文件处理
                        in_history_file = check_if_downloaded(history_file, url)
                        if filepath.exists() and in_history_file:
                            logger.info(f"已存在或已下载，跳过: {filepath}")
                            if not in_history_file:
                                mark_as_downloaded(history_file, url)
                        else:
                            if not download_file(url, filepath):
                                logger.warning(f"下载失败: {url}")
                                continue
                            else:
                                mark_as_downloaded(history_file, url)
                        if data_type == "orderbooks":
                            check_and_create_ticks(filepath, dt)
                        elif data_type == "deals":
                            pass
                            #preprocess_monthly_deals(filepath, filepath.parent)



# =====================================================
# 主流程
# =====================================================
def download_with_config(config: dict):
    base_dir = config['download']['base_dir']
    biz_list = config['download']['biz']
    markets = config['download']['market']
    types = config['download']['types']
    start_utc = config['download']['start_date']
    end_utc = config['download']['end_date']
    history_file = config['history_file']
    download_data(base_dir, biz_list, markets, types, start_utc, end_utc, history_file)


def main():
    global logger
    if len(sys.argv) < 2:
        print("❌ 用法: python read_arg.py <参数>")
        print("📌 请传入至少一个命令行参数。")
        sys.exit(1)

        # 获取第一个参数（索引为 1，因为 sys.argv[0] 是脚本名）
    arg = sys.argv[1]
    print(f"是否为有效路径: {Path(arg).exists()}")
    config_file = arg

    if not os.path.exists(config_file):
        print(f"错误：配置文件不存在: {config_file}")
        sys.exit(1)

    config = load_config(config_file)
    log_dir = config.get("logging", {}).get("log_dir", "./logs")
    logger = setup_logger(log_dir)

    download_with_config(config)


if __name__ == "__main__":
    main()
    pass
