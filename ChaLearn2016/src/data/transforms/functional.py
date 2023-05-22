'''
Author      : now more
Contact     : lin.honghui@qq.com
LastEditors: Please set LastEditors
Description : 
LastEditTime: 2020-10-15 10:13:54
'''
import numpy as np
import torch
import cv2 as cv
from PIL import Image,ImageOps,ImageEnhance,ImageFilter
import warnings
import numbers
import collections


def _is_tensor_image(image):
    '''
    Description:  Return whether image is torch.tensor and the number of dimensions of image.
    Reutrn : True or False.
    '''
    return torch.is_tensor(image) and image.ndimension()==3

def _is_numpy_image(image):
    '''
    Description: Return whether image is np.ndarray and the number of dimensions of image
    Return: True or False.
    '''
    return isinstance(image,np.ndarray) and (image.ndim in {2,3} )

def _is_numpy(landmarks):
    '''
    Description: Return whether landmarks is np.ndarray.
    Return: True or False
    '''
    return isinstance(landmarks,np.ndarray)


def to_tensor(sample):
    '''
    Description: Convert sample.values() to Tensor.
    Args (type): sample : {image:ndarray,target:int}
    Return: Converted sample
    '''
    # image,target = sample['image'],sample['target']
    raw_data,label = sample['raw_data'],sample['label']


    raw_data = torch.from_numpy(raw_data)
    # label = torch.from_numpy(label)
    sample['raw_data'] = raw_data
    return sample

        

def randomshift(sample,shift_limit,final_frame_nb=39):
    src_samples = sample['raw_data']
    num, dim = src_samples.shape # 1,1287
    dst_samples = np.zeros(src_samples.shape)
    trans_nb_list = np.random.randint(-shift_limit,shift_limit,num)

    for k in range(num):
        trans_nb = trans_nb_list[k]
        if trans_nb==0:
            dst_samples[k,:] = src_samples[k,:]
        elif trans_nb<0:
            src_arr = src_samples[k,:]
            dst_arr_l = np.zeros(dim)
            trans_nb = -trans_nb
            for i in range(dim//final_frame_nb):
                # left shift.
                dst_arr_l[i*final_frame_nb:(i+1)*final_frame_nb-trans_nb] = \
                    src_arr[i*final_frame_nb+trans_nb:(i+1)*final_frame_nb]
                dst_arr_l[(i+1)*final_frame_nb-trans_nb:(i+1)*final_frame_nb] = \
                    src_arr[(i+1)*final_frame_nb-1]
            dst_samples[k,:] = dst_arr_l
        else:
            src_arr = src_samples[k,:]
            dst_arr_r = np.zeros(dim)
            for i in range(dim//final_frame_nb):
                # right shift.
                dst_arr_r[i*final_frame_nb+trans_nb:(i+1)*final_frame_nb] = \
                    src_arr[i*final_frame_nb:(i+1)*final_frame_nb-trans_nb]
                dst_arr_r[i*final_frame_nb:i*final_frame_nb+trans_nb] = \
                    src_arr[i*final_frame_nb]
            dst_samples[k,:] = dst_arr_r
    sample['raw_data'] = dst_samples
    return sample

def gaussnoise(sample,scale):
    raw_data = sample['raw_data']
    noise_mat = np.random.normal(scale=scale,size=raw_data.shape)
    sample['raw_data'] = raw_data + noise_mat
    return sample


# def randomrotation(sample,angles):
#     def rxf(a):
#         x = np.array([1, 0, 0, 0,
#                     0, np.cos(a), np.sin(a), 0,
#                     0, -np.sin(a), np.cos(a), 0,
#                     0, 0, 0, 1])
#         return x.reshape(4,4)

#     def ryf(b):
#         y = np.array([np.cos(b), 0, -np.sin(b), 0,
#                     0, 1, 0, 0,
#                     np.sin(b), 0, np.cos(b), 0,
#                     0, 0, 0, 1])
#         return y.reshape(4,4)

#     def rzf(c):
#         z = np.array([np.cos(c), np.sin(c), 0, 0,
#                     -np.sin(c), np.cos(c), 0, 0,
#                     0, 0, 1, 0,
#                     0, 0, 0, 1])
#         return z.reshape(4,4)

#     raw_data = sample['raw_data']
#     x_angles = np.random.uniform(-1,1)*r_angle[0]
#     y_angles = np.random.uniform(-1,1)*r_angle[1]
#     z_angles = np.random.uniform(-1,1)*r_angle[2]
#     angles[:,0] = x_angles[:,0]
#     angles[:,1] = y_angles[:,0]
#     angles[:,2] = z_angles[:,0]