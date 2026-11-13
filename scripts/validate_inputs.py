import argparse,json
from stockmamba.data import load_panel
p=argparse.ArgumentParser(); p.add_argument('--input',required=True); p.add_argument('--config',required=True); a=p.parse_args(); f=load_panel(a.input,True); print(json.dumps({'status':'valid','rows':len(f),'dates':int(f.date.nunique())}))
