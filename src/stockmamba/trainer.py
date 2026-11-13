import json
from pathlib import Path
import numpy as np, torch
from .data import daily_windows
from .model import PaperAlignedStockMamba
from .losses import paper_rank_position_loss

def train(frame,cfg,outdir):
 out=Path(outdir); out.mkdir(parents=True,exist_ok=True); fs=[c for c in frame if c.startswith('feature_')]; ms=[c for c in frame if c.startswith('market_')]; tr=set(cfg['split']['train_dates']); va=set(cfg['split']['validation_dates']); mc=cfg['model']
 model=PaperAlignedStockMamba(len(fs),len(ms),d_model=mc['d_model'],temporal_heads=mc['temporal_heads'],stock_heads=mc['stock_heads'],beta=mc['beta'],dropout=mc['dropout']); opt=torch.optim.AdamW(model.parameters(),lr=cfg['training']['learning_rate'],weight_decay=cfg['training']['weight_decay']); best=-float('inf'); hist=[]
 for epoch in range(cfg['training']['epochs']):
  model.train(); losses=[]
  for d,f,m,t,y in daily_windows(frame,mc['lookback'],True):
   if d not in tr or len(t)<cfg['training']['min_stocks']: continue
   pred=model(torch.from_numpy(f),torch.from_numpy(m))['prediction']; valid=np.isfinite(y)
   if valid.sum()<cfg['training']['min_stocks']: continue
   loss,_=paper_rank_position_loss(pred[valid],torch.from_numpy(y[valid]),alpha=cfg['loss']['alpha'],gamma=cfg['loss']['gamma']); opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['training']['grad_clip']); opt.step(); losses.append(float(loss.detach()))
  model.eval(); scores=[]
  with torch.no_grad():
   for d,f,m,t,y in daily_windows(frame,mc['lookback'],True):
    if d not in va or len(t)<cfg['training']['min_stocks']: continue
    pred=model(torch.from_numpy(f),torch.from_numpy(m))['prediction'].numpy(); valid=np.isfinite(y)
    if valid.sum()>1 and np.std(pred[valid])>0 and np.std(y[valid])>0: scores.append(float(np.corrcoef(pred[valid],y[valid])[0,1]))
  metric=float(np.nanmean(scores)) if scores else float('nan'); hist.append({'epoch':epoch+1,'train_loss':float(np.mean(losses)) if losses else None,'valid_pearson':metric})
  if np.isfinite(metric) and metric>best:
   best=metric; torch.save({'model_state':model.state_dict(),'factor_size':len(fs),'market_size':len(ms),'feature_columns':fs,'market_columns':ms,'config':mc|cfg['loss'],'best_epoch':epoch+1,'best_valid_pearson':best},out/'checkpoint.pt')
 (out/'history.json').write_text(json.dumps(hist,indent=2)+'\n'); (out/'selection_manifest.json').write_text(json.dumps({'status':'selected','best_valid_pearson':best},indent=2)+'\n')
