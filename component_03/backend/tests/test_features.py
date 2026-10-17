import os, sys, numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from c03_engine.features import FeatureBuilder, FLOOR


def test_feature_builder_matches_contract():
    rg = np.random.default_rng(1); n = 80; close = 60000 * np.exp(np.cumsum(rg.normal(0, 1e-3, n))); vol = rg.gamma(2, 3, n) + 0.1
    df = pd.DataFrame({'open': close, 'high': close * (1 + abs(rg.normal(0, 5e-4, n))), 'low': close * (1 - abs(rg.normal(0, 5e-4, n))), 'close': close, 'volume': vol, 'trades': rg.integers(50, 400, n), 'taker_buy_base': vol * rg.uniform(.2, .8, n), 'large_count': rg.integers(0, 4, n)})
    fb = FeatureBuilder('BTCUSDT'); got = [fb.push(r) for r in df.to_dict('records')]
    lr = np.log(df.close / df.close.shift(1)).fillna(0).to_numpy(); buy = df.taker_buy_base; sell = df.volume - buy; tfi = ((buy - sell) / (buy + sell + 1e-8)).to_numpy()
    hl = ((df.high - df.low) / df.close).to_numpy(); rush = (df.trades / (df.volume + 1e-8)).to_numpy()
    m15 = df.volume.shift(1).rolling(15).mean(); s15 = df.volume.shift(1).rolling(15).std(ddof=1); va = ((df.volume - m15) / np.maximum(s15, FLOOR['BTCUSDT'])).to_numpy()
    assert all(g is None for g in got[:15]) and all(g is not None for g in got[15:])
    for k in range(15, n): assert np.allclose(got[k], [lr[k], tfi[k], hl[k], rush[k], va[k], df.large_count[k]], atol=1e-12), k
