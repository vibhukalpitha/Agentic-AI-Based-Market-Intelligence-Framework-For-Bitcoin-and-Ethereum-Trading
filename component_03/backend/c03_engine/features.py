"""Layer 1 feature builder for live closed bars (Feature Contract v1.0.0). Input: one closed 1-minute kline plus the count of large fills from the @trade stream."""
import math
from collections import deque
import numpy as np

FLOOR = {'BTCUSDT': 2.539920310163499, 'ETHUSDT': 27.765804660846563}     # volume_acceleration std floor (base units)
LARGE = {'BTCUSDT': 0.1, 'ETHUSDT': 2.0}                                    # large-trade threshold (base units)
ORDER = ['log_returns', 'trade_flow_imbalance', 'hl_spread_ratio', 'rush_order_density', 'volume_acceleration', 'large_trade_count']


class FeatureBuilder:
    def __init__(self, sym, prev_close=None, prev_volumes=()):
        self.sym, self.prev_close, self.vols = sym, prev_close, deque(prev_volumes, maxlen=15)

    def push(self, bar):
        """bar: dict open, high, low, close, volume, trades, taker_buy_base, large_count. Returns the 6 features (list) or None while fewer than 15 prior closed bars exist."""
        c, h, l, v = float(bar['close']), float(bar['high']), float(bar['low']), float(bar['volume'])
        lr = 0.0 if self.prev_close is None else math.log(c / self.prev_close)
        buy = float(bar['taker_buy_base']); sell = v - buy; den = buy + sell; tfi = (buy - sell) / (den + 1e-8) if den > 0 else 0.0
        hl = (h - l) / c; rush = float(bar['trades']) / (v + 1e-8)
        if len(self.vols) == 15:
            a = np.asarray(self.vols, 'float64'); va = (v - a.mean()) / max(a.std(ddof=1), FLOOR[self.sym]); ok = True
        else: va, ok = 0.0, False
        self.prev_close = c; self.vols.append(v)
        return [lr, tfi, hl, rush, va, float(bar['large_count'])] if ok else None
