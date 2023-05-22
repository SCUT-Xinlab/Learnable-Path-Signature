import torch
import torch.nn.functional as F
from torch import nn

class DropBlock_Ske(nn.Module):
    def __init__(self, block_size=7, num_point=11):
        super(DropBlock_Ske, self).__init__()
        self.keep_prob = 0.0
        self.num_point = num_point
        self.block_size = block_size


    def forward(self, input, keep_prob, A): # n,c,t,v
        self.keep_prob = keep_prob
        if not self.training or self.keep_prob == 1:
            return input
        n,c,t,v = input.size()

        input_abs = torch.mean(torch.mean(torch.abs(input),dim=2),dim=1).detach() 
        input_abs = input_abs/torch.sum(input_abs)*input_abs.numel()
        gamma = (1. - self.keep_prob) / (1+1.92)
        M_seed = torch.bernoulli(torch.clamp(input_abs * gamma, max=1.0)).to(device=input.device, dtype=input.dtype)
        M = torch.matmul(M_seed,A) 
        M[M>0.001] = 1.0
        M[M<0.5] = 0.0
        mask = (1-M).view(n,1,1,self.num_point)
        return input * mask * mask.numel() /mask.sum()
