import pandas as pd
import os
from datetime import datetime, timedelta


def build_filepath(base: str, biz: str, data_type: str, market: str, dt: datetime) -> str:
    year = dt.strftime("%Y")
    month = dt.strftime("%m")
    day = dt.strftime("%d")
    hour = dt.strftime("%H")

    path = f"{biz}/{data_type}/{year}{month}/"

    filename = f"{market}-{year}{month}"

    if data_type == "candlesticks_30s":
        filename += f"{day}.csv.gz"
    elif data_type.startswith("candlesticks_"):
        filename += f".csv.gz"
    elif data_type == 'daily_deals' and biz == 'spot':
        path = f"{biz}/deals/{year}{month}/{year}{month}{day}/"
        filename += f"{day}{hour}.csv"
    elif data_type == "deals" and biz == "spot":
        filename += ".csv.gz"
    elif data_type in ["trades", "mark_prices", "funding_applies", "funding_updates"]:
        filename += ".csv.gz"
    elif data_type == 'labels':
        path += f"{year}{month}{day}/"
        filename += f"{day}{hour}.csv.gz"
    else:
        path += f"{year}{month}{day}/"
        filename += f"{day}{hour}.csv.gz"
    return "/".join([base, path + filename])


def load_config(config_path: str) -> dict:
    import yaml
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        print(f"配置文件加载成功: {config_path}")
        return config
    except Exception as e:
        print(f"加载配置文件失败: {e}")
        sys.exit(1)


def main():
    import sys
    if sys.argv.__len__() != 2:
        return

    config = load_config(sys.argv[1])
    markets = config['download']['market']
    start_utc = config['download']['start_date']
    end_utc = config['download']['end_date']
    base_dir = config['download']['base_dir']
    for market in markets:
        run(base_dir, start_utc, end_utc, market)

def test():
    data_dir = "/Users/zephyr/codes/alpha_spring/data_spring/data"
    start_dt = "2023-03-03"
    end_dt = "2023-03-04"
    market = "BTC_USDT"

def run(data_dir, start_dt, end_dt, market):
    start_dt = datetime.fromisoformat(f"{start_dt}:00+00:00")
    end_dt = datetime.fromisoformat(f"{end_dt}:00+00:00")
    biz = 'spot'
    data_type = 'ticks'
    last_dt = None
    while start_dt < end_dt:
        input_time = start_dt
        result = check_data(data_dir, biz, data_type,market, input_time)
        dt = start_dt.strftime("%Y-%m-%d %H")
        if not result["success"]:
            print(f'{market}@{dt}: {result}')
        elif start_dt.hour == 23:
            print(f'{market}@{dt}: pass')
        start_dt += timedelta(hours=1)


def check_data(data_dir: str, biz: str, data_type: str, market: str, input_time):
    file_path = build_filepath(data_dir, biz, data_type, market, input_time)
    return validate_csv_gz_file(file_path, input_time)


def validate_csv_gz_file(file_path: str, input_time: datetime) -> dict:
    """
    校验一个 csv.gz 文件的时间戳是否符合预期。

    参数:
        file_path (str): .csv.gz 文件路径
        input_time (datetime): 输入的时间点

    返回:
        dict: 包含校验结果和消息
    """
    result = {
        'success': False,
        'details': {}
    }

    start_ts = input_time.timestamp()

    # 1. 检查文件是否存在
    if not os.path.exists(file_path):
        result["details"] = f"文件不存在: {file_path}"
        return result

    try:
        # 2. 使用 pandas 读取 .csv.gz 文件
        df = pd.read_csv(file_path, compression='gzip')

        if df.empty:
            result["details"] = f"{file_path} 内容为空"
            return result

        if df.shape[0] != 3600:
            result["details"] = f"{file_path} 行数{df.shape[0]} != 3600"
            return result


        # 获取第一行和最后一行的 Timestamp
        actual_first = df.iloc[0]['timestamp']
        actual_last = df.iloc[-1]['timestamp']

        # 转换 input_time 为 timezone-naive（如果需要）
        if int(actual_first) != start_ts + 1:
            result["details"] = f"{file_path} start_date is not correct"
            return result

        if int(actual_last) != start_ts + 3600:
            result["details"] = f"{file_path} end_date is not correct"
            return result

    except Exception as e:
        result['details'] = f"读取或校验文件时出错: {str(e)}"

    result['success'] = True

    return result

# === 使用示例 ===
if __name__ == "__main__":
    main()
    pass