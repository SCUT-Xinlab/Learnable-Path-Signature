'''
Author      : now more
Contact     : lin.honghui@qq.com
LastEditors: Please set LastEditors
Description : 
LastEditTime: 2020-10-18 22:24:19
'''
import numpy as np
config = dict(
    # Basic Config
    enable_backends_cudnn_benchmark = True,
    max_epochs = 500 + 1,
    log_period = 0.1,
    save_dir = r"../checkpoints/",
    save_period = 5,
    n_save = 2,
    log_dir = r"../log/",
    tag = "",          

    # Dataset
    train_pipeline = dict(
        dataloader = dict(batch_size = 64,num_workers = 2,drop_last = True,pin_memory=True,shuffle=True,),

        dataset = dict(type="dataset",root_path=r"/data/ChengJiale/GestureRecognition/ChaLearn2016/train_16_new.h5"),

        transforms = [
            dict(type="RandomShift",p=1,shift_limit=5,final_frame_nb=39),
            dict(type="ToTensor",),
            ],

    ),
    test_pipeline = dict(
        dataloader = dict(batch_size = 64,num_workers = 2,drop_last = False,pin_memory=True,shuffle=False,),
        dataset = dict(type="dataset",root_path=r"/data/ChengJiale/GestureRecognition/ChaLearn2016/test_16_new.h5"),
        transforms = [
            # dict(type="RandomShift",p=1,shift_limit=5,final_frame_nb=39),
            dict(type="ToTensor",),
            ],
    ),
    val_pipeline = dict(
        dataloader = dict(batch_size = 64,num_workers = 2,drop_last = False,pin_memory=True,shuffle=False,),
        dataset = dict(type="dataset",root_path=r"/data/ChengJiale/GestureRecognition/ChaLearn2016/val_16_new.h5"),
        transforms = [
            # dict(type="RandomShift",p=1,shift_limit=5,final_frame_nb=39),
            dict(type="ToTensor",),
            ],

    ),
    # Model
    model = dict(
        arch = dict(type="baseline"),
        net = dict(type="Proposed_new",path_length=3,spatial_length=3,temporal_length=3,joint_num=55,dropout_rate=0.6),
        losses = dict(type="CrossEntropyLoss"),
                
    ),


    multi_gpu = False,
    max_num_devices = 2, #自动获取空闲显卡，默认第一个为主卡


    # Solver
    # lr_scheduler = dict(type="ExponentialLR",gamma=0.99997), # cycle_momentum=False if optimizer==Adam
    # cycle_momentum=False if optimizer==Adam
    lr_scheduler = dict(type="CyclicLR",base_lr=1e-8,max_lr=1e-5,step_size_up=2800,mode='triangular',cycle_momentum=False),

    # optimizer = dict(type="Adam",lr=1e-3,weight_decay=1e-5),
    # optimizer = dict(type="Adam",lr=1e-3),
    optimizer = dict(type="Adam",lr=1e-4,weight_decay=1e-5),
    # warm_up = dict(length=2000,min_lr=1e-7,max_lr=3e-5,froze_num_lyers=0)
    
    find_lr = dict(init_value=1e-7,final_value=0.01,beta=0.98,),

)


if __name__ == "__main__":

    pass