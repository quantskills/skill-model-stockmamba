import pandas as pd, torch
from .data import load_panel,daily_windows
from .model import PaperAlignedStockMamba

def load_checkpoint(path):
 ck=torch.load(path,map_location='cpu',weights_only=False); c=ck['config']; m=PaperAlignedStockMamba(ck['factor_size'],ck['market_size'],d_model=c['d_model'],temporal_heads=c['temporal_heads'],stock_heads=c['stock_heads'],beta=c['beta'],dropout=c['dropout']); m.load_state_dict(ck['model_state']); m.eval(); return m,ck

def predict(frame,checkpoint):
 model,ck=load_checkpoint(checkpoint); rows=[]
 for d,f,m,t,_ in daily_windows(frame,ck['config']['lookback'],False):
  with torch.no_grad(): pred=model(torch.from_numpy(f),torch.from_numpy(m))['prediction'].numpy()
  rows.append(pd.DataFrame({'date':d,'ticker':t,'prediction':pred}))
 return pd.concat(rows,ignore_index=True) if rows else pd.DataFrame(columns=['date','ticker','prediction'])
