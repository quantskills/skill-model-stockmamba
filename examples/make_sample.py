from pathlib import Path
import numpy as np, pandas as pd
rng=np.random.default_rng(3); rows=[]
for d in range(20240102,20240107):
    for t in range(12):
        x=rng.normal(size=4); m=rng.normal(size=3); rows.append([d,f"{t:06d}.SZ",*x,*m,float(x[0]*.2+rng.normal())])
cols=["date","ticker"]+[f"feature_{i}" for i in range(4)]+[f"market_{i}" for i in range(3)]+["label"]
f=pd.DataFrame(rows,columns=cols); Path("examples/generated").mkdir(parents=True,exist_ok=True); f.to_parquet("examples/generated/train.parquet",index=False); f.drop(columns="label").to_parquet("examples/generated/test.parquet",index=False)
