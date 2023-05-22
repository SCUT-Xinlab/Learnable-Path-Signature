'''
Author      : now more
Contact     : lin.honghui@qq.com
LastEditors: Please set LastEditors
Description : 
LastEditTime: 2020-10-17 04:43:16
'''
import sys
sys.path.append('..')
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import torch.nn as nn
import torch
from argparse import ArgumentParser
import datetime
import os
from copy import deepcopy

from src.data.dataloader import build_dataloader
from configs import merage_from_arg,load_arg
from src.model import build_model
from src.solver import make_optimizer,wrapper_lr_scheduler
from src.engine import do_train, do_test
from src.utils import get_free_device_ids,find_lr



if __name__ == "__main__":
    # 若更新了load_arg函数，需要对应更新merage_from_arg()
    arg = vars(load_arg())
    # 待修改
    config_file = arg["CONFIG_FILE"]
    config_file = config_file.replace("../","").replace(".py","").replace('/','.')

    exec(r"from {} import config as cfg".format(config_file))
    # if arg['MODEL.LOAD_PATH'] != None: #优先级：arg传入命令 >model中存的cfg > config_file
    #     cfg = torch.load(arg['MODEL.LOAD_PATH'])['cfg']
    cfg = merage_from_arg(cfg,arg)
    print(cfg)
    cfg = deepcopy(cfg)

    # train_dataloader = build_dataloader(cfg['train_pipeline'])
    test_dataloader = build_dataloader(cfg['test_pipeline'])

    
    curr_time = datetime.datetime.now()
    time_str = datetime.datetime.strftime(curr_time,'%Y%m%d_')
    save_dir = os.path.join(cfg['save_dir'],time_str+cfg['tag'])
    log_dir = os.path.join(cfg['log_dir'],"log_"+time_str+cfg['tag'])
    cfg['save_dir'] = save_dir
    cfg['log_dir'] = log_dir
    # if not os.path.isdir(save_dir):
    #     os.makedirs(save_dir)
    # if not os.path.isdir(log_dir):
    #     os.makedirs(log_dir)
    print("Save_dir :",save_dir)
    print("Log_dir :", log_dir)

    model = build_model(cfg,pretrain_path=arg['load_path'])

    # optimizer = make_optimizer(cfg['optimizer'],model)
    # lr_scheduler = wrapper_lr_scheduler(cfg['lr_scheduler'],optimizer)

    if arg['device']: # 传入命令指定 device id
        free_device_ids = arg['device']
    else:
        free_device_ids = get_free_device_ids()

    max_num_devices = cfg['max_num_devices']
    if len(free_device_ids)>=max_num_devices:
        free_device_ids = free_device_ids[:max_num_devices]

    print("free_device_ids :",free_device_ids)
    master_device = free_device_ids[0]
    model.cuda(master_device)
    if cfg['multi_gpu']:
        model = nn.DataParallel(model,device_ids=free_device_ids).cuda(master_device)


    if cfg['enable_backends_cudnn_benchmark']:
        print("enable backends cudnn benchmark")
        torch.backends.cudnn.benchmark = True

    # if arg['find_lr'] == True: 
    #     import matplotlib
    #     import matplotlib.pyplot as plt
    #     matplotlib.use('Agg')
    #     cfg_find_lr = cfg['find_lr']
    #     log_lrs,log_losses = find_lr(model,optimizer,loss_fn=model.losses,dataloader=train_dataloader,**cfg_find_lr,
    #         inputs_transform=lambda x: x,outputs_transform=lambda x: x,targets_transform=lambda x: x,device=master_device)
    #     plt.plot(np.log10(log_lrs),log_losses)
    #     plt.savefig(os.path.join(cfg['save_dir'],"find_lr.jpg"))

    # else:
    #     do_train(cfg,model=model,train_loader=train_dataloader,val_loader=test_dataloader,optimizer=optimizer,
    #                 scheduler=lr_scheduler,metrics=None,device=free_device_ids)
    do_test(cfg,model=model,val_loader=test_dataloader,metrics=None,device=free_device_ids)
