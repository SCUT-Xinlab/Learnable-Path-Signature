from src.data.dataset import dataset as DATASETS
from copy import deepcopy

def build_dataset(cfg_dataset,transforms):
    '''
    Description:
    '''
    cfg_dataset = deepcopy(cfg_dataset)
    dataset_type = cfg_dataset.pop("type")
    dataset_kwags = cfg_dataset
    
    if hasattr(DATASETS,dataset_type):
        dataset = getattr(DATASETS,dataset_type)(**dataset_kwags,transforms=transforms)
    else:
        raise ValueError("\'type\' of dataset is not defined. Got {}".format(dataset_type))
    return dataset