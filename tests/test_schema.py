import pandas as pd
from stockmamba.data import load_panel
def test_schema(tmp_path):
 f=pd.DataFrame({'date':[1,1],'ticker':['a','b'],'feature_1':[.1,.2],'market_1':[.2,.3],'label':[.1,.2]}); p=tmp_path/'x.parquet'; f.to_parquet(p,index=False); assert len(load_panel(p))==2
