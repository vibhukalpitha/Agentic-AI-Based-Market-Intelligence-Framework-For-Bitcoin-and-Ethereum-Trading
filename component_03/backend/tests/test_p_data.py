"""p_data_archive: minute statistics on a small synthetic aggTrades file (known answers), chunk boundaries inside a minute, zip round trip."""
import io, os, sys, zipfile
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'tools'))
import p_data_archive as P


def make_csv(n_min=5, per=40, seed=0):
    rg = np.random.default_rng(seed); rows = []; aid = 0; fid = 100; T0 = 1_700_000_040_000
    for m in range(n_min):
        for k in range(per):
            q = round(float(rg.choice([0.01, 0.01, 0.02, 0.5])), 8); cnt = int(rg.integers(1, 4)); rows.append((aid, 60000 + m + k * .1, q, fid, fid + cnt - 1, T0 + m * 60000 + k * 1000, bool(rg.integers(0, 2)), True)); aid += 1; fid += cnt
    return pd.DataFrame(rows), rows


def test_stats_and_chunks(tmp_path):
    df, rows = make_csv(); csv = df.to_csv(header=False, index=False).replace('True', 'True').replace('False', 'False')
    full = P.stream_minutes(io.StringIO(csv), 'BTCUSDT', chunk_rows=10 ** 6); small = P.stream_minutes(io.StringIO(csv), 'BTCUSDT', chunk_rows=37)     # chunk cuts inside minutes
    assert len(full) == 5 and full.drop(columns='ts').equals(small.drop(columns='ts')) or np.allclose(full.drop(columns='ts').to_numpy(float), small.drop(columns='ts').to_numpy(float))
    d = pd.DataFrame(rows, columns=P.COLS); d['m'] = d.time // 60000; g = d[d.m == d.m.iloc[0]]; f = full.iloc[0]
    assert f.n_agg == len(g) and abs(f.volume - g.qty.sum()) < 1e-9 and f.n_trades == (g.last_id - g.first_id + 1).sum()
    assert f.large_count == (g.qty >= 0.1).sum() and abs(f.taker_buy_base - g[~g.is_buyer_maker].qty.sum()) < 1e-9
    assert f.high == g.price.max() and f.low == g.price.min() and f.open == g.price.iloc[0] and f.close == g.price.iloc[-1]
    side = (~g.is_buyer_maker).astype(int).to_numpy(); best = cur = 0; last = None
    for s in side: cur = cur + 1 if s == last else 1; last = s; best = max(best, cur) if s == 1 else best
    assert f.buy_run_max == best and f.same_size_prev == int((g.qty.to_numpy()[1:] == g.qty.to_numpy()[:-1]).sum())


def test_zip_roundtrip(tmp_path):
    df, _ = make_csv(3); zp = tmp_path / 'x.zip'
    with zipfile.ZipFile(zp, 'w') as z: z.writestr('BTCUSDT-aggTrades-2024-08-01.csv', df.to_csv(header=False, index=False))
    out = P.process_zip(str(zp), 'BTCUSDT'); assert len(out) == 3 and list(out.columns) == P.OUT_COLS


def test_large_def_counts(tmp_path):
    import p_data_large_def as L
    rows = [(i, 1.0, q, 0, 0, 1_700_000_040_000 + 1000 * i, True, True) for i, q in enumerate([0.05, 0.1, 0.1, 0.2, 0.09, 0.5])]; zp = tmp_path / 'a.zip'
    with zipfile.ZipFile(zp, 'w') as z: z.writestr('a.csv', pd.DataFrame(rows).to_csv(header=False, index=False))
    c = L.count_zip(str(zp), 'BTCUSDT', L.AGG_COLS); m = 1_700_000_040_000 // 60000
    assert c['>=x1.0'][m] == 4 and c['>x1.0'][m] == 2 and c['>=x0.5'][m] == 6 and c['>=x2.0'][m] == 2
