'''
Author      : now more
Contact     : lin.honghui@qq.com
LastEditors: Please set LastEditors
Description : 
LastEditTime: 2020-10-17 04:37:10
'''
import torch.optim.lr_scheduler as lr_schedulers
from torch.optim.lr_scheduler import *

class wrapper_lr_scheduler(object):
    def __init__(self,cfg_lr_scheduler,optimizer):
        cfg_lr_scheduler = cfg_lr_scheduler.copy()
        lr_scheduler_type = cfg_lr_scheduler.pop('type')
        if hasattr(lr_schedulers,lr_scheduler_type):
            self.lr = getattr(lr_schedulers,lr_scheduler_type)(optimizer,**cfg_lr_scheduler)
        else:
            raise KeyError("lr_scheduler not found. Got {}".format(lr_scheduler_type))
    def ITERATION_COMPLETED(self):
        if isinstance(self.lr,(CyclicLR,ExponentialLR)):
            self.lr.step()
        
    
    def EPOCH_COMPLETED(self):
        if isinstance(self.lr,(StepLR,MultiStepLR,CosineAnnealingWarmRestarts,CosineAnnealingLR)):
            self.lr.step()