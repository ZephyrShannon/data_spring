# generate_orderbook_with_deltas.py

import csv
import datetime
import gzip
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Generator

import numpy as np
import pandas as pd
import sys
import logging
import logging
logger = logging.getLogger("GateDownloader")  # 日志对象


class Tick:
    def __init__(self, price: int):
        self.price = price
        self.cum_vol = 0.
        self.trade_vol = 0.
        self.order_vol = 0.

    def get_tick(self):
        # 成交了trade_vol， 盘口增加了order_vol
        return self.trade_vol + self.order_vol



def load_deals(deals_path: Path) -> pd.DataFrame:
    """加载 deals 数据"""
    print(f"📊 加载成交数据: {deals_path}")
    if str(deals_path).endswith('.gz'):
        df = pd.read_csv(deals_path, sep=',', compression='gzip')
    else:
        df = pd.read_csv(deals_path, sep=',')

    df['Timestamp'] = pd.to_numeric(df['Timestamp'], errors='raise')
    df = df.sort_values('Timestamp').reset_index(drop=True)
    return df


class TicksRecord:
    def __init__(self, deals_file, order_length=3600, step_ms=1000, precision=1e6):
        self.precision = precision
        self.deals = load_deals(deals_file)
        self.buy_ticks = dict()
        self.sell_ticks = dict()
        self.updated_sell_prices = dict()
        self.updated_buy_prices = dict()
        self.start_ts = None
        self.end_ts = None
        self.step_ms = step_ms
        self.order_length = order_length
        self.file_end_ts = None

    def get_buy_tick(self, raw_price: float):
        price_int = int(raw_price * self.precision)
        tick = self.buy_ticks.get(price_int)
        if tick is None:
            tick = Tick(price_int)
            self.buy_ticks[price_int] = tick
        return tick

    def get_sell_tick(self, raw_price: float):
        price_int = int(raw_price * self.precision)
        tick = self.sell_ticks.get(price_int)
        if tick is None:
            tick = Tick(price_int)
            self.sell_ticks[price_int] = tick
        return tick

    def on_update_deals(self, side: int,  price: float, volume: int):
        """更新盘口并记录挂单变动"""

        if side == 1:
            tick = self.get_sell_tick(price)
            self.updated_sell_prices[tick.price] = tick
        else:
            tick = self.get_buy_tick(price)
            self.updated_buy_prices[tick.price] = tick
        tick.trade_vol += volume

    def on_update_orderbook(self, side: int, action: str, price: float, volume: float):
        """更新盘口并记录挂单变动"""

        if side == 1:
            tick = self.get_sell_tick(price)
            self.updated_sell_prices[tick.price] = tick
        else:
            tick = self.get_buy_tick(price)
            self.updated_buy_prices[tick.price] = tick

        if action == 'set':
            tick.cum_vol = max(0.0, volume)
        elif action == 'make':
            tick.order_vol += volume
            tick.cum_vol += volume
        elif action == 'take':
            tick.order_vol -= volume
            tick.cum_vol -= volume

            if abs(tick.cum_vol) < 1e-12:
                tick.cum_vol = 0
        else:
            print(f"⚠️ 未知 Action: {action}")

    def reset_record(self):
        for k,v in self.updated_sell_prices.items():
            v.trade_vol = 0
            v.order_vol = 0
            if v.cum_vol == 0:
                self.sell_ticks.pop(k)
        for k, v in self.updated_buy_prices.items():
            v.trade_vol = 0
            v.order_vol = 0
            if v.cum_vol == 0:
                # k 是price， 这个price的档位空了，则清除
                self.buy_ticks.pop(k)
        self.updated_buy_prices.clear()
        self.updated_sell_prices.clear()

    def build_top20_snapshot(self) -> Dict[str, float]:
        """构建 top20 快照"""
        row = {'timestamp': round(self.end_ts, 1)}
        # 输入orders
        deals_df = self.deals
        mask = (deals_df['Timestamp'] > self.start_ts) & (deals_df['Timestamp'] <= self.end_ts)
        trades = deals_df[mask]
        for _, t in trades.iterrows():
            # Timestamp,Deal_id,Price,Volume,Side
            vol = t['Volume']
            price = t['Price']
            side = t['Side']
            self.on_update_deals(side, price, vol)

        # Ask (升序)
        ask_prices = sorted(self.sell_ticks.keys())
        total_sell_vol = 0
        total_sell_amount = 0
        total_sell_order_amount = 0
        total_sell_order_vol = 0

        total_buy_vol = 0
        total_buy_amount = 0
        total_buy_order_amount = 0
        total_buy_order_vol = 0

        max_price = np.nan
        min_price = np.nan

        for i in range(20):
            p = ask_prices[i] if i < len(ask_prices) else 0.0
            p_float = round(p/self.precision, 10)
            v = self.sell_ticks.get(p)
            row[f'ask_{i}_price'] = p_float
            if v is not None:
                row[f'ask_{i}_vol'] = v.cum_vol
                order = v.get_tick()
                row[f'ask_{i}_order'] = order
                row[f'ask_{i}_trade'] = v.trade_vol
                total_sell_order_vol += order
                total_sell_order_amount += (order * p_float)
                total_sell_vol += v.trade_vol
                total_sell_amount += v.trade_vol * p_float
                if v.trade_vol != 0.0:
                    if not max_price < p_float:
                        max_price = p_float
                    if not min_price > p_float:
                        min_price = p_float

            else:
                row[f'ask_{i}_vol'] = 0
                row[f'ask_{i}_order'] = 0
                row[f'ask_{i}_trade'] = 0


        # Bid (降序)
        bid_prices = sorted(self.buy_ticks.keys(), reverse=True)
        for i in range(20):
            p = bid_prices[i] if i < len(bid_prices) else 0.0
            p_float = round(p / self.precision, 10)
            v = self.buy_ticks.get(p)
            row[f'bid_{i}_price'] = p_float
            if v is not None:
                row[f'bid_{i}_vol'] = v.cum_vol
                order = v.get_tick()
                row[f'bid_{i}_order'] = order
                row[f'bid_{i}_trade'] = v.trade_vol
                total_buy_order_vol += order
                total_buy_order_amount += (order * p_float)
                total_buy_vol += v.trade_vol
                total_buy_amount += v.trade_vol * p_float
                if v.trade_vol != 0.0:
                    if not max_price < p_float:
                        max_price = p_float
                    if not min_price > p_float:
                        min_price = p_float
            else:
                row[f'bid_{i}vol'] = 0
                row[f'bid_{i}_order'] = 0
                row[f'bid_{i}_trade'] = 0

        row["total_buy_order_vol"] = total_buy_order_vol
        row['total_buy_order_amount'] = total_buy_order_amount
        row['total_buy_vol'] = total_buy_vol
        row['total_buy_amount'] = total_buy_amount

        row['total_sell_order_vol'] = total_sell_order_vol
        row['total_sell_order_amount'] = total_sell_order_amount
        row['total_sell_vol'] = total_sell_vol
        row['total_sell_amount'] = total_sell_amount

        row['high'] = max_price
        row['low'] = min_price

        # 清除记录
        self.reset_record()
        return row

    def process_line(self, line: str) -> List[Dict[str, float]]:
        """
        处理 orderbook 一行
        返回: (snapshot, prev_ts, curr_ts)
        """
        try:
            parts = [p.strip() for p in line.split(',')]
            if len(parts) < 5:
                return []

            raw_ts = parts[0]
            ts = float(raw_ts)
            side = int(parts[1])  # 1: ask, 2: bid
            action = parts[2]
            price = float(parts[3])
            volume = float(parts[4])

            if int(ts) == 1682899300.0:
                return []

            if self.start_ts is None:
                self.start_ts = int(ts / 3600) * 3600.0
                if ts != self.start_ts:
                    logger.warning(f"Found missing_ticks of {self.start_ts}")
                self.end_ts = self.start_ts + 1
                self.file_end_ts = ts + self.order_length
            snapshot_lists = []
            if ts > self.file_end_ts:
                return snapshot_lists
            if ts - self.end_ts >= 1:
                logger.warning(f"Found missing_ticks of {self.start_ts}")
            while self.end_ts < ts:
                snapshot = self.build_top20_snapshot()
                self.start_ts = self.end_ts
                self.end_ts = self.end_ts + 1
                snapshot_lists.append(snapshot)
            if ts > self.file_end_ts:
                return snapshot_lists

            # 更新盘口状态 & 挂单统计
            self.on_update_orderbook(side, action, price, volume)
            return snapshot_lists

        except Exception as e:
            logger.error(f"解析失败: {line} | {e}")
            return []


    def generate(self, orderbook_path: str, output_path: str):
        output_path = Path(output_path)

        """主流程"""
        fieldnames = (
            ['timestamp'] +
            [f'ask_{i}_price' for i in range(20)] +
            [f'ask_{i}_vol' for i in range(20)] +
            [f'ask_{i}_order' for i in range(20)] +
            [f'ask_{i}_trade' for i in range(20)] +
            [f'bid_{i}_price' for i in range(20)] +
            [f'bid_{i}_vol' for i in range(20)] +
            [f'bid_{i}_order' for i in range(20)] +
            [f'bid_{i}_trade' for i in range(20)] +
            ['total_sell_vol', 'total_sell_amount',
             'total_buy_vol', 'total_buy_amount',
             'total_sell_order_vol', 'total_sell_order_amount',
             'total_buy_order_vol', 'total_buy_order_amount', "high", "low"]
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)

        # 打开 orderbook 文件
        open_func = gzip.open if str(orderbook_path).endswith('.gz') else open
        mode = 'rt' if str(orderbook_path).endswith('.gz') else 'r'
        output_file = str(output_path)
        if not output_file.endswith(".gz"):
            output_file += ".gz"
        with open_func(orderbook_path, mode, encoding='utf-8') as f_ob, \
            gzip.open(output_file, 'wt', newline='', encoding='utf-8', compresslevel=6) as f_out:

            writer = csv.DictWriter(f_out, fieldnames=fieldnames)
            writer.writeheader()

            # 处理 header
            try:
                header = next(f_ob)
            except StopIteration:
                print("❌ orderbook 文件为空")
                return

            has_header = all(h in header for h in ['Timestamp', 'Side'])
            if not has_header:
                snap_list = self.process_line(header)
                if snap_list:
                    for snap in snap_list:
                        writer.writerow(snap)

            # 处理剩余行
            for line in f_ob:
                if not line.strip():
                    continue
                snap_list = self.process_line(line)
                if snap_list:
                    for snap in snap_list:
                        writer.writerow(snap)
            while self.file_end_ts >= self.end_ts:
                    snap = self.build_top20_snapshot()
                    self.start_ts = self.end_ts
                    self.end_ts = self.end_ts + 1
                    if snap:
                        writer.writerow(snap)


def generate_ticks(orderbook_file:str, deals_file:str, output_file:str):
    if not Path(orderbook_file).exists():
        print(f"orderbook 文件不存在: {orderbook_file}")
        sys.exit(1)
    if not Path(deals_file).exists():
        print(f"deals 文件不存在: {deals_file}")
        sys.exit(1)
    gen = TicksRecord(deals_file)
    gen.generate(orderbook_file, output_file)


def main():
    if len(sys.argv) < 4:
        raise Exception("3 args required! orderbook_file:Path, deals_file:Path, output_file:")

    orderbook_file = sys.argv[1]
    deals_file = sys.argv[2]
    output_file = sys.argv[3]
    generate_ticks(orderbook_file, deals_file, output_file)


# ========================
# 主函数
# ========================
if __name__ == '__main__':
    #main()
    pass

def test_main():
    "data/spot/ticks/202303/20230311/BTC_USDT-2023031102.csv.gz"
    orderbook_path = "/Users/zephyr/codes/alpha_spring/data_spring/data/spot/orderbooks/202305/BTC_USDT-2023050100.csv.gz"
    deals_path = "/Users/zephyr/codes/alpha_spring/data_spring/data/spot/deals/202305/20230501/BTC_USDT-2023050100.csv"
    output_path = "./ticks_test.csv"
    gen = TicksRecord(deals_path)
    gen.generate(orderbook_path, output_path)

