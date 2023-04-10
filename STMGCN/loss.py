import torch.nn.functional as F
import torch

def target_distribution(q):

    p = q ** 2 / torch.sum(q, dim=0)
    p = p / torch.sum(p, dim=1, keepdim=True)
    return p

def kl_loss(q, p):
    return F.kl_div(q, p, reduction="batchmean")