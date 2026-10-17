"""Input checks for the C03 engine: what happens when data is missing, garbage, all zeros or out of order.

Rule: a minute that fails a check is NOT given to the models (so nothing bad enters the rolling memory) and is NOT shown as NORMAL.
It is reported as state NO_DATA with the reason in plain words; the next valid minute is scored normally.
Nothing here changes a model, a feature formula, a normalisation, the fusion or a threshold. The limits below are physical sanity limits
(at least 5 times the largest value seen in the Aug 2024 replay data) and were never used to tune anything.
"""
import math

import numpy as np

FEATS = ['log_returns', 'trade_flow_imbalance', 'hl_spread_ratio', 'rush_order_density', 'volume_acceleration', 'large_trade_count']
WARMUP_MIN = 1500

# (lowest allowed, highest allowed) per feature; None = no limit on that side
LIMITS = {
    'log_returns': (-0.5, 0.5),                     # a 40 % move inside one minute is not a market move
    'trade_flow_imbalance': (-1.0000001, 1.0000001),  # (buy - sell) / (buy + sell) can only lie in [-1, 1]
    'hl_spread_ratio': (0.0, 0.5),                  # (high - low) / close
    'rush_order_density': (0.0, 1.0e6),             # trades per unit of volume
    'volume_acceleration': (-1000.0, 1000.0),
    'large_trade_count': (0.0, 1.0e6),              # a count: never negative
}

TEXT = {
    'bad_shape': 'The minute does not have the 6 expected numbers.',
    'not_a_number': 'A value is empty or not a number (NaN / inf / text).',
    'all_zero': 'All six values are exactly zero. A real market minute is never like that, so this looks like a missing or filled-in row.',
    'out_of_range': 'A value is outside the range a real market minute can have.',
    'bad_timestamp': 'The time stamp cannot be read.',
    'duplicate_or_out_of_order': 'This minute is not newer than the last scored minute (duplicate or out of order).',
    'rebuilding_features': 'Re-building the 15-minute history after a bad minute; scoring resumes automatically.',
    'missing_field': 'A required field is missing from the bar.',
    'non_positive_price': 'A price is zero, negative or not a number.',
    'high_below_low': 'High is lower than low.',
    'ohlc_inconsistent': 'Open or close lies outside the high-low range.',
    'negative_volume': 'Volume is negative or not a number.',
    'negative_trades': 'The number of trades is negative or not a number.',
    'taker_gt_volume': 'Taker-buy volume is larger than the total volume.',
    'zero_volume_with_trades': 'Volume is zero but trades were reported.',
    'missing_columns': 'Required columns are missing.',
    'too_few_bars': 'Not enough history to warm the engine up.',
    'bad_rows': 'The history contains rows that fail the checks.',
}


class DataError(ValueError):
    """Raised for whole-input problems (warm-up history too short, columns missing, bad rows). Message is written for the user."""

    def __init__(self, code, message):
        super().__init__(message)
        self.code, self.message = code, message


def problem(code, detail=None):
    return {'code': code, 'text': TEXT[code], 'detail': detail}


def _is_num(x):
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def check_features(vec):
    """one minute of the six Layer-1 features -> list of problems (empty = OK)"""
    try:
        a = np.asarray(vec, dtype='float64').reshape(-1)
    except (TypeError, ValueError):
        return [problem('not_a_number', 'could not convert to numbers')]
    if a.shape[0] != len(FEATS):
        return [problem('bad_shape', f'got {a.shape[0]} values')]
    bad = [FEATS[i] for i in range(len(FEATS)) if not np.isfinite(a[i])]
    if bad:
        return [problem('not_a_number', ', '.join(bad))]
    if not a.any():
        return [problem('all_zero')]
    out = []
    for i, f in enumerate(FEATS):
        lo, hi = LIMITS[f]
        if (lo is not None and a[i] < lo) or (hi is not None and a[i] > hi):
            out.append(problem('out_of_range', f'{f} = {a[i]:.6g}'))
    return out


def check_bar(bar):
    """one closed 1-minute bar (open, high, low, close, volume, trades, taker_buy_base) -> list of problems (empty = OK)"""
    need = ['open', 'high', 'low', 'close', 'volume', 'trades', 'taker_buy_base']
    miss = [k for k in need if not isinstance(bar, dict) or bar.get(k) is None]
    if miss:
        return [problem('missing_field', ', '.join(miss))]
    if not all(_is_num(bar[k]) for k in need):
        return [problem('not_a_number', ', '.join(k for k in need if not _is_num(bar[k])))]
    o, h, l, c, v, n, tb = (float(bar[k]) for k in need)
    out = []
    if min(o, h, l, c) <= 0:
        out.append(problem('non_positive_price'))
    else:
        tol = 1e-9 * max(h, 1.0)
        if h < l - tol:
            out.append(problem('high_below_low'))
        elif h < max(o, c) - tol or l > min(o, c) + tol:
            out.append(problem('ohlc_inconsistent'))
    if v < 0:
        out.append(problem('negative_volume'))
    if n < 0:
        out.append(problem('negative_trades'))
    if v >= 0 and tb > v * (1 + 1e-9) + 1e-12:
        out.append(problem('taker_gt_volume'))
    if v == 0 and n > 0:
        out.append(problem('zero_volume_with_trades'))
    if tb < 0:
        out.append(problem('negative_volume', 'taker-buy volume is negative'))
    return out


def check_warmup(F):
    """history used to warm the engine up (DataFrame with the six feature columns); raises DataError with a clear message"""
    missing = [f for f in FEATS if f not in getattr(F, 'columns', [])]
    if missing:
        raise DataError('missing_columns', f'Required columns are missing: {", ".join(missing)}. Expected: {", ".join(FEATS)}.')
    n = len(F)
    if n < WARMUP_MIN:
        raise DataError('too_few_bars', f'{n:,} minutes were given, but the engine needs at least {WARMUP_MIN:,} (7 days = 10,080 is preferred) to start.')
    try:
        A = F[FEATS].to_numpy('float64')
    except (TypeError, ValueError):
        raise DataError('bad_rows', 'Some values are text or cannot be read as numbers. Clean the file and try again.')
    nonfinite = ~np.isfinite(A).all(axis=1)
    zero = (~nonfinite) & (~A.any(axis=1))
    oor = np.zeros(n, bool)
    for i, f in enumerate(FEATS):
        lo, hi = LIMITS[f]
        col = np.nan_to_num(A[:, i], nan=0.0, posinf=0.0, neginf=0.0)
        oor |= (col < lo) | (col > hi)
    bad = nonfinite | zero | oor
    if bad.any():
        idx = np.flatnonzero(bad)[:5].tolist()
        raise DataError('bad_rows', f'{int(bad.sum()):,} of {n:,} history rows fail the data checks (empty values: {int(nonfinite.sum())}, all zeros: {int(zero.sum())}, impossible values: {int(oor.sum())}); first rows: {idx}. Remove or repair them; the engine does not guess missing minutes.')
    return n


# ---------------------------------------------------------------------------------------------- file check (report only)
MAGIC = [(b'\x89PNG', 'a PNG image'), (b'\xff\xd8\xff', 'a JPEG image'), (b'GIF8', 'a GIF image'), (b'%PDF', 'a PDF document'),
         (b'PK\x03\x04', 'a ZIP / Office file'), (b'MZ', 'a program file'), (b'\x7fELF', 'a program file'), (b'RIFF', 'an audio / video file')]
RAW_COLS = ['open', 'high', 'low', 'close', 'volume', 'trades', 'taker_buy_base']
TS_NAMES = ['ts', 'timestamp', 'time', 'open_time', 'bar_open_utc', 'datetime']


def _f(level, code, text):
    return {'level': level, 'code': code, 'text': text}


def inspect_csv(name, data):
    """Look at an uploaded file and say, in plain words, whether it can be used as 1-minute market data. Nothing is scored here.
    Returns {'ok': bool, 'kind': 'raw_bars' | 'features' | None, 'rows': int, 'findings': [{'level': 'error'|'warn'|'info', 'code', 'text'}]}"""
    import io
    import pandas as pd
    F, res = [], {'ok': False, 'kind': None, 'rows': 0, 'findings': None}
    res['findings'] = F
    if not data:
        F.append(_f('error', 'empty_file', 'The file is empty.'))
        return res
    for sig, desc in MAGIC:
        if data.startswith(sig):
            F.append(_f('error', 'not_csv', f'This is {desc}, not a CSV file of 1-minute market data. Nothing was read from it.'))
            return res
    try:
        txt = data.decode('utf-8-sig')
    except UnicodeDecodeError:
        F.append(_f('error', 'not_text', 'The file is not readable text, so it cannot be a CSV file.'))
        return res
    try:
        d = pd.read_csv(io.StringIO(txt))
    except Exception as e:                                           # parser errors: ragged rows, no columns ...
        F.append(_f('error', 'unreadable', f'The file could not be read as a table ({type(e).__name__}).'))
        return res
    d.columns = [str(c).strip().lower() for c in d.columns]
    res['rows'] = len(d)
    if len(d) == 0:
        F.append(_f('error', 'no_rows', 'The file has column names but no rows.'))
        return res
    tcol = next((c for c in TS_NAMES if c in d.columns), None)
    has_feats = all(f in d.columns for f in FEATS)
    has_raw = all(c in d.columns for c in RAW_COLS)
    if has_raw:
        res['kind'] = 'raw_bars'
        if 'large_count' not in d.columns and 'large_trade_count' not in d.columns:
            F.append(_f('info', 'no_large_trades', 'No large-trade column: large_trade_count is set to 0 (the same as at live start-up). Scores that depend on it are less informed.'))
    elif has_feats:
        res['kind'] = 'features'
    else:
        want_r = [c for c in RAW_COLS if c not in d.columns]
        want_f = [f for f in FEATS if f not in d.columns]
        F.append(_f('error', 'missing_columns', f'The file has neither the raw columns ({", ".join(RAW_COLS)}) nor the six feature columns. Missing for raw bars: {", ".join(want_r)}. Missing for features: {", ".join(want_f)}.'))
        return res
    if tcol is None:
        F.append(_f('error', 'no_time_column', f'No time column found (one of: {", ".join(TS_NAMES)}).'))
        return res
    ts = pd.to_datetime(d[tcol], errors='coerce', utc=True)
    nbad_t = int(ts.isna().sum())
    if nbad_t:
        F.append(_f('error', 'bad_timestamp', f'{nbad_t:,} time stamps cannot be read.'))
    t_ok = ts.dropna()
    if len(t_ok) > 1:
        dif = t_ok.diff().dropna().dt.total_seconds() / 60.0
        if (dif <= 0).any():
            F.append(_f('error', 'duplicate_or_out_of_order', f'{int((dif <= 0).sum()):,} rows are duplicates or go back in time. Sort the file by time and remove repeated minutes.'))
        miss = int(np.clip(dif[dif > 1] - 1, 0, None).sum())
        if miss:
            F.append(_f('warn', 'gaps', f'{miss:,} minute(s) are missing in {int((dif > 1).sum()):,} place(s). The engine does not invent data: it will warn at each gap.'))
    if res['kind'] == 'raw_bars':
        nbad = sum(1 for r in d[RAW_COLS].to_dict('records') if check_bar(r))
    else:
        A = d[FEATS].apply(pd.to_numeric, errors='coerce').to_numpy('float64')
        nbad = sum(1 for r in A if check_features(r))
    if nbad:
        F.append(_f('error' if nbad == len(d) else 'warn', 'bad_rows', f'{nbad:,} of {len(d):,} rows fail the data checks (empty, impossible or inconsistent values). They would be shown as NO DATA and not scored.'))
    need = WARMUP_MIN + (15 if res['kind'] == 'raw_bars' else 0)
    if len(d) - nbad < need:
        F.append(_f('error', 'too_few_bars', f'Only {len(d) - nbad:,} usable minutes. The engine needs at least {need:,} before it can start scoring (7 days = 10,080 is preferred).'))
    res['ok'] = not any(f['level'] == 'error' for f in F)
    if res['ok'] and not F:
        F.append(_f('info', 'ok', 'The file looks usable.'))
    return res
