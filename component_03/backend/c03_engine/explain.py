"""Plain-language reasons for a flagged bar. Model shares are exact (logistic stacker); feature reasons come from TreeSHAP of the XGBoost boosters (validated in notebook 45: top feature inside what the generator touched for 7 of 8 families);
the AE per-feature errors are shown as a secondary, lower-confidence line (notebook 45: H-E3 not met). Volume-burst-like patterns always carry the ambiguity caveat (notebook 40)."""
import numpy as np, xgboost as xgb

FEATURE_TEXT = {'log_returns': {'high': 'the price rose sharply within the minute', 'low': 'the price fell sharply within the minute'},
    'trade_flow_imbalance': {'high': 'aggressive buyers dominate the executed volume', 'low': 'aggressive sellers dominate the executed volume'},
    'hl_spread_ratio': {'high': 'the candle range is very wide (a price range, not the bid-ask spread)', 'low': 'the candle range is unusually narrow (a price range, not the bid-ask spread)'},
    'rush_order_density': {'high': 'a rush of many small trades per unit of volume', 'low': 'few, large trades per unit of volume'},
    'volume_acceleration': {'high': 'volume surged compared with the previous 15 minutes', 'low': 'volume dried up compared with the previous 15 minutes'},
    'large_trade_count': {'high': 'an unusual number of large trades', 'low': 'no large trades'}}
MODEL_TEXT = {'ae': 'the combination of features is unfamiliar to the novelty detector (it has never seen it in normal trading)', 'xgboost': 'the recent minutes match a known manipulation-like pattern',
              'transformer': 'the multi-minute sequence resembles a known manipulation-like pattern', 'tgnn': 'the order-book structure is unusual (not active yet)'}
VOLUME_CAVEAT = 'A volume surge on 1-minute bars can also be a natural spike; this is shown as unusual, not as proof.'
VOLUME_LIKE = {'volume_acceleration', 'large_trade_count', 'rush_order_density'}


def _groups(c):
    base = [n.split('__')[0] for n in c.FA_NAMES]; return {g: [i for i, n in enumerate(base) if n == g] for g in c.FEATS}


def reasons(engine, res, top_models=2, top_feats=2):
    """res = result dict of SymbolEngine.step (state != NORMAL). Returns a list of {type, text, weight} and a caveat list."""
    c, last, out, cav = engine.c, engine._last, [], []
    ct = res.get('contrib') or {}; tot = sum(abs(v) for v in ct.values() if v) or 1.0
    for k, v in sorted(((k, v) for k, v in ct.items() if k != 'shared' and v > 0), key=lambda kv: -kv[1])[:top_models]: out.append({'type': 'model', 'text': MODEL_TEXT[k], 'weight': round(v / tot, 3)})
    if last['Xa'] is not None:
        dm = xgb.DMatrix(last['Xa'], feature_names=c.FA_NAMES); con = np.mean([b.predict(dm, pred_contribs=True, iteration_range=(0, bi + 1))[0, :-1] for b, bi in engine.bun.xgb], axis=0)
        G = _groups(c); grp = np.array([con[G[f]].sum() for f in c.FEATS]); z6 = last['z6']; order = np.argsort(-grp); pos = grp[grp > 0].sum() or 1.0
        for j in order[:top_feats]:
            if grp[j] <= 0: continue
            f = c.FEATS[j]; out.append({'type': 'feature', 'text': FEATURE_TEXT[f]['high' if z6[j] >= 0 else 'low'], 'weight': round(float(grp[j] / pos), 3), 'feature': f})
        if c.FEATS[order[0]] in VOLUME_LIKE: cav.append(VOLUME_CAVEAT)
    if last['cols8'] is not None:
        e = last['cols8'][2:]; j = int(np.argmax(e)); out.append({'type': 'feature_secondary', 'text': 'least familiar feature for the novelty detector: ' + c.FEATS[j].replace('_', ' ') + ' (lower confidence)', 'weight': None, 'feature': c.FEATS[j]})
    return out, cav
