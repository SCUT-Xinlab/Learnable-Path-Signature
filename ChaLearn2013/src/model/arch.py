'''
Author      : now more
Contact     : lin.honghui@qq.com
LastEditors: Please set LastEditors
Description : 
LastEditTime: 2020-10-15 10:22:31
'''
import torch
import torch.nn as nn
from copy import deepcopy
from torch.utils.tensorboard import SummaryWriter
from src.model import net as NET
from src.model import losses as LOSSES

class baseline(nn.Module):
    def __init__(self,cfg):
        super(baseline,self).__init__()
        self.cfg = deepcopy(cfg)
        cfg_net = self.cfg['model']['net']
        cfg_losses = self.cfg['model']['losses']
        self.net = self.build_net(cfg_net)
        self.losses = self.build_losses(deepcopy(cfg_losses))


        # for log
        log_dir = self.cfg['log_dir']
        self.writer = SummaryWriter(log_dir=log_dir)
        self.step = 1


    def build_losses(self,cfg_losses):
        loss_type = cfg_losses.pop("type")
        if hasattr(LOSSES,loss_type):
            loss = getattr(LOSSES,loss_type)(**cfg_losses)
            return loss
        else:
            raise KeyError("loss_type not found. Got {}".format(loss_type))
    
    def build_net(self,cfg_net):
        net_type = cfg_net.pop('type')
        if hasattr(NET,net_type):
            net = getattr(NET,net_type)(**cfg_net)
            return net
        else:
            raise KeyError("net_type not found. Got {}".format(net_type))

    def forward(self,inputs,targets=None):
        pred = self.net(inputs)

        # calu losses
        if self.training:
            loss = self.losses(pred,targets)
            self.writer.add_scalar("loss",loss.cpu().data.numpy(),self.step)
            loss = torch.unsqueeze(loss,0)

            self.step += 1
            return loss
        else:
            # inference
            return pred







