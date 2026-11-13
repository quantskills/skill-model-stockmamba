import torch
from stockmamba.losses import paper_rank_position_loss
def test_loss():
 x=torch.randn(40); y=torch.randn(40); loss,_=paper_rank_position_loss(x,y); assert loss.ndim==0
