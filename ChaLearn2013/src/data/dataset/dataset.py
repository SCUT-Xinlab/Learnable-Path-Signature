'''
Author      : now more
Contact     : lin.honghui@qq.com
LastEditors: Please set LastEditors
Description : 
LastEditTime: 2020-10-17 09:30:00
'''
from  torch.utils.data import Dataset
import os
import numpy as np
import h5py
from copy import deepcopy
from collections import defaultdict
import torch

def load_h5py(path):
    with h5py.File(path,"r") as hf:
        x = np.array(hf.get('matrix'))
        x = np.transpose(x,(0,1,3,2)).reshape(x.shape[0],-1)
        y = np.array(hf.get('label'))
        return x,y

def sel_temp_point(batch_xs, sel_temp_points_id=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], sel_points_id=[i for i in range(20)], final_frame_nb=39):
	batch_xs_temp = np.array([[]]*batch_xs.shape[0])
	for i in sel_temp_points_id:
		p_id = sel_points_id.index(i)
		batch_xs_temp = np.append(batch_xs_temp, \
			batch_xs[:,p_id*(final_frame_nb*3):(p_id+1)*(final_frame_nb*3)], axis=1)
	return batch_xs_temp


class balance_dataset(Dataset):
    '''
    根据类别数目平衡采样，每次采样 sample_per_classes*num_classes 个样本构成 mini-batch
    '''
    def __init__(self,root_path,samples_per_classes=4,len_dataset=100,transforms=None):
        self.samples_per_classes = samples_per_classes
        self.len_dataset = len_dataset
        self.transforms = transforms
        data_arr,label_arr = load_h5py(root_path)
        data_arr = sel_temp_point(data_arr)
        self.data_arr = data_arr
        self.label_arr = np.argmax(label_arr,axis=1)

        self.instance_dict = defaultdict(list)
        for i in range(len(self.label_arr)):
            class_id = self.label_arr[i]
            self.instance_dict[class_id].append(i)
        self.class_ids = list(self.instance_dict.keys())
        
        # self.step = 0

    def __len__(self):
        return self.len_dataset
    
    def __getitem__(self,idx):
        # idx = np.random.choice(self.class_ids,size=1)[0] # 0-19 选一个id
        # print(idx)
        sampler_indices = np.random.choice(self.instance_dict[idx],size=self.samples_per_classes)
        sampler_indices = np.array(sampler_indices).reshape(-1)
        batch_raw_data,batch_label = self.data_arr[sampler_indices],self.label_arr[sampler_indices]
        # print(self.step,batch_label)
        # self.step += 1

        batch_sampler = [{"raw_data":batch_raw_data[i][None,:],"label":batch_label[i]} for i in range(len(batch_label))]
        if self.transforms:
            batch_sampler = [self.transforms(sample) for sample in batch_sampler]
        
        batch_raw_data,batch_label = list(),list()
        for i in range(len(batch_sampler)):
            batch_raw_data.append(batch_sampler[i]['raw_data'].unsqueeze(0))
            batch_label.append(batch_sampler[i]['label'])
        batch_raw_data = torch.cat(batch_raw_data,dim=0).float()
        batch_label = torch.tensor(batch_label)

        return batch_raw_data,batch_label


class dataset(Dataset):
    def __init__(self, root_path,transforms=None):
        data_arr,label_arr = load_h5py(root_path)
        data_arr = sel_temp_point(data_arr)
        self.data_arr = data_arr
        self.label_arr = np.argmax(label_arr,axis=1)
        self.transforms = transforms
    
    def __len__(self):
        return len(self.data_arr)


    def __getitem__(self,idx):
        raw_data,label = (self.data_arr[idx,:])[None,:],self.label_arr[idx]
        sample = {'raw_data':raw_data,'label':label}
        if self.transforms:
            sample = self.transforms(sample)
        raw_data,label = sample['raw_data'],sample['label']
        
        return raw_data.float(),label



if __name__ == "__main__":
    # trainset = dataset(root_path=r"C:\Users\now more\Desktop\chalearn2013\data\train.h5",transforms=None)
    balance_trainset = balance_dataset(root_path=r"/home/LinHonghui/Project/chalearn2013/data/train.h5",samples_per_classes=4)
    # print(trainset[0][0].shape,trainset[0][1])
    sample = balance_trainset[0]
    print(sample[0].shape)
    # import pdb; pdb.set_trace()