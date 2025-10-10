# generate_orderbook_with_deltas.py

import csv
import datetime
import gzip
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Generator
import pandas as pd
import sys

class Tick:
    def __init__(self):
        self.buy_trade_vol = 0
        self.sell_trade_vol = 0
        self.buy_order_vol = 0
        self.sell_order_vol = 0

    def get_buy_order(self):
        # 卖掉了buy_trade_vol， 盘口增加了buy_order_vol
        return self.buy_trade_vol + self.buy_order_vol

    def get_sell_order(self):
        # 同上
        return self.sell_trade_vol + self.sell_order_vol


class TicksRecord:
    def __init__(self, precision=1e6):
        self.precision = precision
        self.ticks = dict()

    def get_tick(self, raw_price: float):
        price_int = raw_price * self.precision
        tick = self.ticks.get(price_int)
        if tick is None:
            tick = Tick()
            self.ticks[price_int] = tick
        return tick


class OrderbookWithDeltasGenerator:
    def __init__(self):
        self.asks = {}
        self.bids = {}
        self.current_ts: Optional[float] = None

        # 用于统计当前时间窗口内的 make/take 变动
        self.ob_vol_b = 0.0  # buy side volume delta (make - take)
        self.ob_vol_s = 0.0  # sell side volume delta
        self.ob_amount_b = 0.0
        self.ob_amount_s = 0.0

    def build_top20_snapshot(self) -> Dict[str, float]:
        """构建 top20 快照"""
        row = {'timestamp': round(self.current_ts, 1)}

        # Ask (升序)
        ask_prices = sorted(self.asks.keys())
        for i in range(20):
            p = ask_prices[i] if i < len(ask_prices) else 0.0
            v = self.asks.get(p, 0.0)
            row[f'ask_price_{i}'] = round(p, 10)
            row[f'ask_vol_{i}'] = round(v, 10)

        # Bid (降序)
        bid_prices = sorted(self.bids.keys(), reverse=True)
        for i in range(20):
            p = bid_prices[i] if i < len(bid_prices) else 0.0
            v = self.bids.get(p, 0.0)
            row[f'bid_price_{i}'] = round(p, 10)
            row[f'bid_vol_{i}'] = round(v, 10)

        return row

    def reset_delta_counters(self):
        """重置本周期挂单变动计数器"""
        self.ob_vol_b = 0.0
        self.ob_vol_s = 0.0
        self.ob_amount_b = 0.0
        self.ob_amount_s = 0.0

    def update_book_and_delta(self, side: int, action: str, price: float, volume: float):
        """更新盘口并记录挂单变动"""
        book = self.asks if side == 1 else self.bids
        is_sell = (side == 1)

        if action == 'set':
            book[price] = max(0.0, volume)
        elif action == 'make':
            book[price] = book.get(price, 0.0) + volume
            # 记录挂单增量
            if is_sell:
                self.ob_vol_s += volume
                self.ob_amount_s += price * volume
            else:
                self.ob_vol_b += volume
                self.ob_amount_b += price * volume
        elif action == 'take':
            current = book.get(price, 0.0)
            new_amount = max(0.0, current - volume)
            if new_amount < 1e-12:
                book.pop(price, None)
            else:
                book[price] = new_amount
            # take 表示吃掉挂单，相当于负向挂单（可理解为隐式撤单）
            if is_sell:
                self.ob_vol_s -= volume
                self.ob_amount_s -= price * volume
            else:
                self.ob_vol_b += volume
                self.ob_amount_b += price * volume
        else:
            print(f"⚠️ 未知 Action: {action}")

    def process_line(self, line: str) -> Tuple[Optional[Dict], float, float]:
        """
        处理 orderbook 一行
        返回: (snapshot, prev_ts, curr_ts)
        """
        try:
            parts = [p.strip() for p in line.split(',')]
            if len(parts) < 5:
                return None, self.current_ts, self.current_ts

            raw_ts = parts[0]
            ts = float(raw_ts)
            side = int(parts[1])  # 1: ask, 2: bid
            action = parts[2]
            price = float(parts[3])
            volume = float(parts[4])

            prev_ts = self.current_ts
            snapshot = None

            # === 时间切换：生成上一个周期快照 ===
            if self.current_ts is not None and ts != self.current_ts:
                snapshot = self.build_top20_snapshot()

            self.current_ts = ts

            # 更新盘口状态 & 挂单统计
            self.update_book_and_delta(side, action, price, volume)

            return snapshot, prev_ts, ts

        except Exception as e:
            print(f"❌ 解析失败: {line} | {e}")
            return None, self.current_ts, self.current_ts

    def load_deals(self, deals_path: Path) -> pd.DataFrame:
        """加载 deals 数据"""
        print(f"📊 加载成交数据: {deals_path}")
        if str(deals_path).endswith('.gz'):
            df = pd.read_csv(deals_path, sep=',', compression='gzip')
        else:
            df = pd.read_csv(deals_path, sep=',')

        df['Timestamp'] = pd.to_numeric(df['Timestamp'], errors='raise')
        df = df.sort_values('Timestamp').reset_index(drop=True)
        return df

    def aggregate_ticks_in_range(self, deals_df: pd.DataFrame,
                                  start_ts: float, end_ts: float) -> Dict[str, float]:
        """
        统计 [start_ts, end_ts) 区间内的成交汇总
        """
        mask = (deals_df['Timestamp'] >= start_ts) & (deals_df['Timestamp'] < end_ts)
        trades = deals_df[mask]

        buy_ticks = dict()
        sell_ticks = dict()

        for _, t in trades.iterrows():
            vol = t['Volume']
            amo = vol * t['Price']
            if t['Side'] == 2:  # Buy
                tb_vol += vol
                tb_amo += amo
            elif t['Side'] == 1:  # Sell
                ts_vol += vol
                ts_amo += amo

        return {
            'tb_vol': round(tb_vol, 10),
            'tb_amo': round(tb_amo, 10),
            'ts_vol': round(ts_vol, 10),
            'ts_amo': round(ts_amo, 10)
        }

    def aggregate_trades_in_range(self, deals_df: pd.DataFrame,
                                  start_ts: float, end_ts: float) -> Dict[str, float]:
        """
        统计 [start_ts, end_ts) 区间内的成交汇总
        """
        mask = (deals_df['Timestamp'] >= start_ts) & (deals_df['Timestamp'] < end_ts)
        trades = deals_df[mask]

        tb_vol = ts_vol = 0.0
        tb_amo = ts_amo = 0.0

        for _, t in trades.iterrows():
            vol = t['Volume']
            amo = vol * t['Price']
            if t['Side'] == 2:  # Buy
                tb_vol += vol
                tb_amo += amo
            elif t['Side'] == 1:  # Sell
                ts_vol += vol
                ts_amo += amo

        return {
            'tb_vol': round(tb_vol, 10),
            'tb_amo': round(tb_amo, 10),
            'ts_vol': round(ts_vol, 10),
            'ts_amo': round(ts_amo, 10)
        }

    def add_delta_features(self, snapshot: Dict, trade_stats: Dict) -> Dict:
        """添加报撤差字段"""
        ob_vol_b = self.ob_vol_b
        ob_vol_s = self.ob_vol_s
        ob_amo_b = ob_vol_b * snapshot.get('bid_price_0', 0.0)  # 近似用 top1 价格
        ob_amo_s = ob_vol_s * snapshot.get('ask_price_0', 0.0)

        tb_vol, tb_amo = trade_stats['tb_vol'], trade_stats['tb_amo']
        ts_vol, ts_amo = trade_stats['ts_vol'], trade_stats['ts_amo']

        db_vol = round(ob_vol_b - tb_vol, 10)
        db_amo = round(ob_amo_b - tb_amo, 10)
        ds_vol = round(ob_vol_s - ts_vol, 10)
        ds_amo = round(ob_amo_s - ts_amo, 10)

        snapshot.update({
            # orderbook buy volume
            'ob_vol_b': round(ob_vol_b, 10),
            # orderbook buy amount
            'ob_amo_b': round(ob_amo_b, 10),
            # orderbook sell volume
            'ob_vol_s': round(ob_vol_s, 10),
            # orderbook sell amount
            'ob_amo_s': round(ob_amo_s, 10),

            # trade buy volume
            'tb_vol': tb_vol,
            # trade buy amount
            'tb_amo': tb_amo,
            # trade sell volume
            'ts_vol': ts_vol,
            # trade sell amount
            'ts_amo': ts_amo,

            # cancel buy volume ,有可能为0，但amount 不为0
            'db_vol': db_vol,
            # cancel buy amount
            'db_amo': db_amo,
            # cancel sell volume
            'ds_vol': ds_vol,
            # cancel sell amount
            'ds_amo': ds_amo
        })
        return snapshot

    def generate(self, orderbook_path: Path, deals_path: Path, output_path: Path):
        """主流程"""
        fieldnames_base = (
            ['timestamp'] +
            [f'ask_price_{i}' for i in range(20)] +
            [f'bid_price_{i}' for i in range(20)] +
            [f'ask_vol_{i}' for i in range(20)] +
            [f'bid_vol_{i}' for i in range(20)]
        )

        # 扩展字段名
        delta_fields = [
            'ob_vol_b', 'ob_amo_b', 'ob_vol_s', 'ob_amo_s',
            'tb_vol', 'tb_amo', 'ts_vol', 'ts_amo',
            'db_vol', 'db_amo', 'ds_vol', 'ds_amo'
        ]
        fieldnames = fieldnames_base + delta_fields

        output_path.parent.mkdir(parents=True, exist_ok=True)

        # 打开 orderbook 文件
        open_func = gzip.open if str(orderbook_path).endswith('.gz') else open
        mode = 'rt' if str(orderbook_path).endswith('.gz') else 'r'

        with open_func(orderbook_path, mode, encoding='utf-8') as f_ob, \
             open(output_path, 'w', newline='', encoding='utf-8') as f_out:

            writer = csv.DictWriter(f_out, fieldnames=fieldnames)
            writer.writeheader()

            # 加载 deals 数据一次性
            deals_df = self.load_deals(deals_path)

            # 处理 header
            try:
                header = next(f_ob)
            except StopIteration:
                print("❌ orderbook 文件为空")
                return

            has_header = all(h in header for h in ['Timestamp', 'Side'])
            if not has_header:
                snap, prev_ts, curr_ts = self.process_line(header)
                if snap:
                    trade_stats = self.aggregate_trades_in_range(deals_df, prev_ts, curr_ts)
                    snap = self.add_delta_features(snap, trade_stats)
                    writer.writerow(snap)
                self.reset_delta_counters()

            start_ts = None
            end_ts = None
            # 处理剩余行
            for line in f_ob:
                if not line.strip():
                    continue
                snap, prev_ts, curr_ts = self.process_line(line)

                if snap:
                    # 计算 [prev_ts, curr_ts) 内的成交
                    trade_stats = self.aggregate_trades_in_range(deals_df, prev_ts, curr_ts)
                    dt = datetime.datetime.fromtimestamp(prev_ts)
                    if start_ts == None:
                        start_ts = prev_ts
                        end_ts = start_ts + 3600.0
                    if start_ts < snap['timestamp'] <= end_ts:
                        snap = self.add_delta_features(snap, trade_stats)
                        writer.writerow(snap)
                    # 每个新周期开始前重置计数器
                    self.reset_delta_counters()

            # 最后一帧
            final_snap = self.build_top20_snapshot()
            if start_ts < final_snap['timestamp'] <= end_ts:
                final_trade_stats = self.aggregate_trades_in_range(deals_df, self.current_ts, self.current_ts + 0.1)
                final_snap = self.add_delta_features(final_snap, final_trade_stats)

                writer.writerow(final_snap)

        print(f"✅ 带报撤差的盘口已生成: {output_path}")



def generate_ticks(orderbook_file:Path, deals_file:Path, output_file: Path):
    if not orderbook_file.exists():
        print(f"❌ orderbook 文件不存在: {orderbook_file}")
        sys.exit(1)
    if not deals_file.exists():
        print(f"❌ deals 文件不存在: {deals_file}")
        sys.exit(1)

    gen = OrderbookWithDeltasGenerator()
    gen.generate(orderbook_file, deals_file, output_file)

def main():
    if len(sys.argv) < 4:
        raise Exception("3 args required! orderbook_file:Path, deals_file:Path, output_file:")
        pass
    orderbook_file = Path(sys.argv[1])
    deals_file = Path(sys.argv[2])
    output_file = Path(sys.argv[3])
    generate_ticks(orderbook_file, deals_file, output_file)
# ========================
# 主函数
# ========================
if __name__ == '__main__':
    # main()
    pass


def test_main():
    orderbook_path = Path("/Users/zephyr/Documents/code/python/TestEnv/data/spot/orderbooks/202304/BTC_USDT-2023042501.csv")
    deals_path = Path("/Users/zephyr/Documents/code/python/TestEnv/data/spot/deals/202304/20230425/BTC_USDT-2023042501.csv")
    output_path = Path("./ticks_test.csv")
    gen = OrderbookWithDeltasGenerator()
    self = gen
    gen.generate(orderbook_path, deals_path, output_path)
