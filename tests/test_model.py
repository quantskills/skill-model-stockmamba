import torch
from stockmamba.model import PaperAlignedStockMamba
def test_forward():
 m=PaperAlignedStockMamba(4,3,d_model=16,temporal_heads=4,stock_heads=2,dropout=.1); o=m(torch.randn(10,4,4),torch.randn(10,4,3)); assert o['prediction'].shape==(10,)
