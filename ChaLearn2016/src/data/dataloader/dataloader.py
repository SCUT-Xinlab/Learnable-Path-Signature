'''
Author      : now more
Contact     : lin.honghui@qq.com
LastEditors: Please set LastEditors
Description : 
LastEditTime: 2020-10-10 09:48:20
'''

from ..dataset import build_dataset
from ..transforms import build_transforms
from src.data.dataloader import sampler as SAMPLERS
from src.data.dataloader import collect_fn as COLLECT_FN


from torch.utils.data import DataLoader
from copy import deepcopy



def build_dataloader(cfg_data_pipeline):
    cfg_data_pipeline = deepcopy(cfg_data_pipeline)
    cfg_dataset = cfg_data_pipeline.pop('dataset')
    cfg_transforms = cfg_data_pipeline.pop('transforms')
    cfg_dataloader = cfg_data_pipeline.pop('dataloader')

    transforms = build_transforms(cfg_transforms)
    dataset = build_dataset(cfg_dataset,transforms)

    # 自定义 sampler
    if 'sampler' in cfg_dataloader:
        cfg_sample = cfg_dataloader.pop('sampler')
        sample_type = cfg_sample.pop('type')
        if hasattr(SAMPLERS,sample_type):
            sampler = getattr(SAMPLERS,sample_type)(**cfg_sample)
            cfg_dataloader['sampler'] = sampler
        else:
            raise ValueError("\'type\' of dataloader.sampler is not defined. Got {}".format(sample_type))

    # 自定义 collate_fn 
    if "collate_fn" in cfg_dataloader:
        cfg_collate_fn = cfg_dataloader.pop("collate_fn")
        collate_fn_type = cfg_collate_fn.pop("type")
        if hasattr(COLLECT_FN,collate_fn_type):
            collate_fn = getattr(COLLECT_FN,collate_fn_type)
            cfg_dataloader['collate_fn'] = collate_fn
        else:
            raise ValueError("\'type\' of dataloader.collate_fn is not defined. Got {}".format(collate_fn_type))
    dataloader = DataLoader(dataset,**cfg_dataloader)
    return dataloader


