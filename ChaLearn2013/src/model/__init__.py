'''
Author      : now more
Contact     : lin.honghui@qq.com
LastEditors: Please set LastEditors
Description : 
LastEditTime: 2020-10-15 10:08:18
'''
import torch
from src.model import arch as ARCH
from copy import deepcopy

def build_model(cfg,pretrain_path=r""):
    cfg = deepcopy(cfg)
    arch_cfg = cfg['model']['arch']
    arch_type = arch_cfg.pop("type")
    if hasattr(ARCH,arch_type):
        model = getattr(ARCH,arch_type)(cfg)
    else:
        raise KeyError("`arch_type` not found. Got {}".format(arch_type))
    
    if pretrain_path:
        model_state_dict = model.state_dict()
        state_dict = torch.load(pretrain_path,map_location='cpu')['model']
        for key in state_dict.keys():
            if key in model_state_dict.keys() and state_dict[key].shape==model_state_dict[key].shape:
                model_state_dict[key] = state_dict[key]
        model.load_state_dict(model_state_dict)
    return model