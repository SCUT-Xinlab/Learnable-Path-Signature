
import torch
import numbers
import warnings
import types
import cv2 as cv
import numpy as np
import random
from . import functional as F
import math






class RandomChoice(object):
    """
    Apply transformations randomly picked from a list with a given probability
    Args:
        transforms: a list of transformations
        p: probability
    """
    def __init__(self,p,transforms):
        self.p = p
        self.transforms = transforms
    def __call__(self,sample):
        if len(self.transforms) < 1:
            raise TypeError("transforms(list) should at least have one transformation")
        for t in self.transforms:
            if np.random.uniform(0,1) < self.p:
                sample = t(sample)
        return sample

    def __repr__(self):
        return self.__class__.__name__+"(p={})".format(self.p)


class Compose(object):
    '''
    Description: Compose several transforms together
    Args (type): 
        transforms (list): list of transforms
        sample (ndarray or dict):
    return: 
        sample (ndarray or dict)
    '''
    def __init__(self,transforms):
        self.transforms = transforms
    def __call__(self,sample):
        for t in self.transforms:
            sample = t(sample)
        return sample
    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string

class ToTensor(object):
    '''
    Description: Convert ndarray in sample to Tensors.
    Args (type): 
        sample (ndarray or dict)
    return: 
        Converted sample.
    '''
    def __call__(self,sample):
        return F.to_tensor(sample)
    def __repr__(self):
        return self.__class__.__name__ + "()"




class Lambda(object):
    '''
    Description: Apply a user-defined lambda as a transform.
    Args (type): lambd (function): Lambda/function to be used for transform.
    Return: 
    '''
    def __init__(self, lambd):
        assert isinstance(lambd, types.LambdaType)
        self.lambd = lambd

    def __call__(self, img):
        return self.lambd(img)

    def __repr__(self):
        return self.__class__.__name__ + '()'



class RandomShift(object):
    def __init__(self,p=1,shift_limit=5,final_frame_nb=39):
        # shift_limit : 最大偏移量
        # final_frame_nb : 视频长度
        assert (shift_limit>=0) and (shift_limit<=final_frame_nb/5)
        self.p = p
        self.shift_limit = shift_limit
        self.final_frame_nb = final_frame_nb

    def __call__(self,sample):
        if np.random.random() <= self.p:
            return F.randomshift(sample,self.shift_limit,self.final_frame_nb)
        else:
            return sample
    def __repr__(self):
        return self.__class__.__name__ + "(p={},shift_limit={},final_frame_nb={})".format(self.p,self.shift_limit,self.final_frame_nb)

class GaussNoise(object):
    def __init__(self,p=1,scale=0.001):
        self.scale = scale
        self.p = p
    
    def __call__(self,sample):
        if np.random.random() <= self.p:
            return F.gaussnoise(sample,self.scale)
        else:
            return sample
    def __repr__(self):
        return self.__class__.__name__ + "(p={},scale={})".format(self.p,self.scale)

# class RandomRotation(object):
#     def __init__(self,angles=[np.pi/36, np.pi/18, np.pi/36]):
#         self.angles = angles
    
#     def __call__(self,sample):
#         pass

#     def __repr__(self):
#         return self.__class__.__name__ + "(angles={})".format(str(tuple(self.angles)))