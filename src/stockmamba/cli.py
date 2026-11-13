import argparse,json
from pathlib import Path
import yaml
from .data import load_panel
from .trainer import train
from .inference import predict

def main():
 p=argparse.ArgumentParser(); sub=p.add_subparsers(dest='command',required=True); a=sub.add_parser('train'); a.add_argument('--config',required=True); a.add_argument('--input',required=True); a.add_argument('--output-dir',required=True); b=sub.add_parser('infer'); b.add_argument('--checkpoint',required=True); b.add_argument('--input',required=True); b.add_argument('--output',required=True); x=p.parse_args()
 if x.command=='train': cfg=yaml.safe_load(Path(x.config).read_text()); train(load_panel(x.input,True),cfg,x.output_dir); print(json.dumps({'status':'complete'}))
 else: out=predict(load_panel(x.input,False),x.checkpoint); Path(x.output).parent.mkdir(parents=True,exist_ok=True); out.to_parquet(x.output,index=False); print(json.dumps({'status':'complete','rows':len(out)}))
