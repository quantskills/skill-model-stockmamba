from pathlib import Path
import numpy as np
import pandas as pd

def load_panel(path, require_label=True):
    p=Path(path); f=pd.read_parquet(p) if p.suffix.lower()=='.parquet' else pd.read_csv(p)
    req={'date','ticker'} | ({'label'} if require_label else set())
    missing=sorted(req-set(f.columns))
    if missing: raise ValueError(f'missing columns: {missing}')
    features=[c for c in f if c.startswith('feature_')]; market=[c for c in f if c.startswith('market_')]
    if not features or not market: raise ValueError('feature_* and market_* columns are required')
    if f[['date','ticker']].duplicated().any(): raise ValueError('duplicate date/ticker rows')
    if not np.isfinite(f[features+market].to_numpy(float)).all(): raise ValueError('non-finite features')
    if require_label and not np.isfinite(f['label'].to_numpy(float)).all(): raise ValueError('non-finite labels')
    return f.sort_values(['date','ticker']).reset_index(drop=True)

def daily_windows(frame, lookback, require_label=True):
    feature=[c for c in frame if c.startswith('feature_')]; market=[c for c in frame if c.startswith('market_')]
    groups={d:g for d,g in frame.groupby('date',sort=True)}; dates=sorted(groups)
    for i in range(lookback-1,len(dates)):
        parts=[groups[d].set_index('ticker') for d in dates[i-lookback+1:i+1]]; common=set(parts[-1].index)
        for g in parts[:-1]: common &= set(g.index)
        tickers=sorted(common)
        if len(tickers)<2: continue
        factors=np.stack([g.loc[tickers,feature].to_numpy('float32') for g in parts],axis=1)
        context=np.stack([g.loc[tickers,market].to_numpy('float32') for g in parts],axis=1)
        label=parts[-1].loc[tickers,'label'].to_numpy('float32') if require_label else None
        yield dates[i],np.nan_to_num(factors),np.nan_to_num(context),tickers,label
