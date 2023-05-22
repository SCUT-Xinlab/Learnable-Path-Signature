'''
Author      : now more
Contact     : lin.honghui@qq.com
LastEditors: Please set LastEditors
Description : 
LastEditTime: 2020-10-16 04:40:16
'''
from torch.nn import init
import torch.nn as nn
    
def weight_init(m):
    classname = m.__class__.__name__
    if classname.find("Linear")!=-1:
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv2d') != -1:
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.normal_(m.weight, 1.0, 0.1)
            nn.init.constant_(m.bias, 0.0)



def input_prepare(x):
    """Transform from NC_in to NCJT"""
    C_in = 3
    J = 11
    T = 39
    N = x.shape[0]
    x = x.view(N, J, C_in, T)   # NJCT
    x = x.permute(0, 2, 1, 3).contiguous() # NCJT
    return x