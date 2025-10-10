import pandas as pd
import os
import zipfile
from datetime import datetime, timedelta
import sys


def read_data_by_month_with_header(name, year, month):
    """
    从无列头的CSV文件中读取数据，并根据指定列名创建DataFrame。

    :param name: 数据目录的名称
    :param year: 年份
    :param month: 月份，需保证为两位数
    :return: 包含指定月份数据及列头的pandas DataFrame
    """
    # 定义期望的列名列表
    column_names = [
        'open_time', 'open', 'high', 'low',
        'close', 'volume', 'close_time',
        'quote_volume', 'trade_num',
        'buyer_volume', 'buyer_quote_volume', 'ignore'
    ]

    # 拼接完整的目录路径和ZIP文件名
    base_dir = f"data/{name}/data/spot/monthly/klines/{name}/1m/"
    zip_filename = f"{name}-1m-{year}-{month:02d}.zip"
    zip_path = os.path.join(base_dir, zip_filename)

    # 确保ZIP文件存在
    if not os.path.exists(zip_path):
        print(f"文件不存在: {zip_path}")
        return None

    # 使用with语句临时解压ZIP文件，读取其中的CSV文件
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        # 假设ZIP文件中只有一个CSV文件，获取CSV文件名
        csv_filenames = [f for f in zip_ref.namelist() if f.endswith('.csv')]
        if not csv_filenames:
            print("ZIP文件中没有找到CSV文件")
            return None

        # 读取CSV文件内容到DataFrame，并指定列名
        with zip_ref.open(csv_filenames[0]) as csv_file:
            # 注意这里设置header=None，因为源文件没有列头，并通过names参数指定列名
            df = pd.read_csv(csv_file, header=None, names=column_names)
    return df



def read_data_by_month_range(NAME, start_year_month, end_year_month, data_reading_function = read_data_by_month_with_header):
    """
    遍历指定日期范围内的每个月份，调用给定的函数读取数据，并合并成一个DataFrame。

    :param NAME: 数据集的名称或标识符
    :param start_year_month: 开始年月字符串，格式为"YYYY-MM"
    :param end_year_month: 结束年月字符串，格式为"YYYY-MM"
    :param data_reading_function: 用于读取单个月份数据的函数，该函数应接收(NAME, year, month)并返回DataFrame
    :return: 合并后的DataFrame
    """
    # 将输入的年月字符串转换为datetime对象，以便进行月份遍历
    start_dt = datetime.strptime(start_year_month, '%Y-%m')
    end_dt = datetime.strptime(end_year_month, '%Y-%m')

    # 初始化空的DataFrame用于存储所有月份的数据
    combined_data = pd.DataFrame()

    # 遍历指定日期范围内的每个月份
    current_dt = start_dt
    while current_dt <= end_dt:
        # 分离年和月
        year_str = current_dt.strftime('%Y')
        month_str = current_dt.strftime('%m')

        # 调用提供的函数读取当前月份的数据
        monthly_data = data_reading_function(NAME, int(year_str), int(month_str))

        # 将本月数据追加到总数据集中
        if not monthly_data.empty:
            combined_data = pd.concat([combined_data, monthly_data], ignore_index=True)

        # 移动到下一个月
        current_dt += timedelta(days=32)  # 加32天确保跨月，下月1号
        current_dt = current_dt.replace(day=1)
    combined_data['index'] = pd.to_datetime(combined_data['open_time'], unit='ms')
    #combined_data['close_time'] = pd.to_datetime(combined_data['close_time'], unit='ms')
    return combined_data.set_index("index")

def test():
    symbol, start_year_month, end_year_month = 'CAKEUSDT', "2022-05","2023-05"
    import os
    os.chdir("/")
    raw_data = read_data_by_month_range(symbol, start_year_month, end_year_month)

    symbol = 'VTHOUSDT'
    raw_data2 = read_data_by_month_range(symbol, start_year_month, end_year_month)
    symbol = "ACAUSDT"
    raw_data3 = read_data_by_month_range(symbol, start_year_month, end_year_month)



# 使用示例
#if __name__ == "__main__":

def main():
    if len(sys.argv) < 4:
        print("使用方法: python3 data_reader.py symbol 年-月-日")
        sys.exit(1)
    symbol = sys.argv[1]
    start_year_month = sys.argv[2]
    end_year_month = sys.argv[3]
    combined_df = read_data_by_month_range(symbol, start_year_month, end_year_month, read_data_by_month_with_header)
    print("Combined DataFrame:")
    print(combined_df)
    combined_df.to_csv(symbol+"_"+start_year_month+"_"+end_year_month+".csv")