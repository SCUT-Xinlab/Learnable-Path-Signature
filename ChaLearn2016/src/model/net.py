'''
Author      : now more
Contact     : lin.honghui@qq.com
LastEditors: Please set LastEditors
Description : 
LastEditTime: 2020-10-19 00:15:18
'''
import torch
import torch.nn as nn
from copy import deepcopy
from src.model.utils import weight_init,input_prepare
from src.model.layers import *
from torch.utils.tensorboard import SummaryWriter
from src.model_MSG3D.msg3d import MSG3D_Model, MSG3D_Model_2layer
from src.model_2sAGCN.agcn import AGCN_Model, AGCN_Model_2Layer
from src.model_ShiftGCN.shift_gcn import ShiftGCN_Model
from src.model_DecoupleGCN.decouple_gcn import DecoupleGCN_Model
from src.model_STGCN.st_gcn_aaai18 import ST_GCN_18, ST_GCN_18_2LAYER
from src.model_DSTA.dstanet import DSTANet
from src.model_SLGCN.decouple_gcn_attn import SLGCN_Model
import numpy as np

chalearn13_hand_crafted_spatial_path = np.array([
    [1, 0, 1], 
    [0, 1, 2],
    [1, 2, 1],
    [4, 3, 1], 
    [5, 4, 3],
    [6, 5, 4],
    [5, 6, 5],
    [8, 7, 1],
    [9, 8, 7],
    [10, 9, 8],
    [9, 10, 9]
]).astype(np.float32)


chalearn16_hand_crafted_spatial_path = np.array([
    [1, 0, 1],
    [0, 1, 10], 
    [1, 2, 3],
    [2, 3, 4],
    [3, 4, 5],
    [4, 5, 4],
    [1, 6, 7],
    [6, 7, 8],
    [7, 8, 9],
    [8, 9, 8],
    [1, 10, 11],
    [10, 11, 12],
    [11, 12, 13],
    [12, 13, 12],
    [1, 14, 15],
    [14, 15, 16],
    [15, 16, 17],
    [16, 17, 16],
    [1, 18, 19],
    [18, 19, 20],
    [19, 20, 21],
    [20, 21, 20]
]).astype(np.float32)


chalearn13_hand_crafted_adjacent_matrix = np.array([
    [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0],
    [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
])


chalearn16_hand_crafted_adjacent_matrix = np.array([
    [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], #-1
    [1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0], #0
    [0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], #1
    [0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], #2
    [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], #3
    [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], #4
    [0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], #5
    [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], #6
    [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], #7
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], #8
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], #9
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], #10
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0], #11
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0], #12
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0], #13
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0], #14
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0], #15
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0], #16
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0], #17
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0], #18
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1], #19
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1] #20
])


chalearn16_hand_crafted_bone_matrix = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], #-1
    [-1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], #0
    [0, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], #1
    [0, 0, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], #2
    [0, 0, 0,- 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], #3
    [0, 0, 0, 0, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], #4
    [0, -1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], #5
    [0, 0, 0, 0, 0, 0, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], #6
    [0, 0, 0, 0, 0, 0, 0, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], #7
    [0, 0, 0, 0, 0, 0, 0, 0, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], #8
    [0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], #9
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], #10
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], #11
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0], #12
    [0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0], #13
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 1, 0, 0, 0, 0, 0, 0], #14
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 1, 0, 0, 0, 0, 0], #15
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 1, 0, 0, 0, 0], #16
    [0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0], #17
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 1, 0, 0], #18
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 1, 0], #19
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 1] #20
])


chalearn16_hand_crafted_decoupled_adjacent_matrix = np.array([
    [
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], #-1
        [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0], #0
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], #1
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], #2
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], #3
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], #4
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], #5
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], #6
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], #7
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], #8
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], #9
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], #10
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0], #11
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], #12
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], #13
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], #14
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], #15
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], #16
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0], #17
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0], #18
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], #19
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] #20
    ],
    [
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], #-1
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], #0
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], #1
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], #2
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], #3
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], #4
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], #5
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], #6
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], #7
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], #8
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], #9
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], #10
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], #11
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0], #12
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0], #13
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], #14
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], #15
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], #16
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0], #17
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0], #18
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0], #19
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1] #20
    ],
    [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], #-1
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], #0
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], #1
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], #2
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], #3
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], #4
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], #5
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], #6
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], #7
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], #8
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], #9
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], #10
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], #11
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], #12
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], #13
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0], #14
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], #15
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], #16
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], #17
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0], #18
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0], #19
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0] #20
    ]
])


chalearn16_hand_crafted_adjacent_matrix_new = np.array([
#   0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4
    [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], #0
    [1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], #1
    [1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], #2
    [0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], #3
    [0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], #4
    [1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], #5
    [1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], #6
    [0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], #7
    [0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], #8
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], #9
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], #10
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], #11
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], #12
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], #13
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], #14
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], #15
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], #16
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], #17
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], #18
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], #19
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], #20
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], #21
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], #22
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], #23
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], #24
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], #25
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], #26
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], #27
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], #28
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], #29
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], #30
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], #31
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], #32
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], #33
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0], #34
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], #35
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], #36
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], #37
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], #38
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], #39
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], #40
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], #41
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], #42
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], #43
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], #44
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0], #45
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0], #46
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0], #47
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0], #48
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0], #49
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0], #50
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0], #51
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0], #52
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1], #53
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1] #54
])


class PROPOSED(nn.Module):
    def __init__(self, with_bn=True, bn_before_actfn=True, joint_num=44, spatial_ps=False,dropout_rate=0.5):
        super(PROPOSED, self).__init__()

        self.C_in = 2
        self.J = joint_num
        self.T = 39
        self.C_out = 48 * joint_num * 9
        self.cls_num = 249

        self.with_bn = with_bn
        self.bn_before_actfn = bn_before_actfn
        self.spatial_ps = spatial_ps

        self.increase_dimension_early = True
        self.spatial = False
        self.temporal = True
        self.conv = True

        self.non_local = True
        self.double = False

        self.nonlocal_spatial = True
        self.nonlocal_temporal = False
        self.nonlocal_concat = False
        self.nonlocal_self = True

        self.path_length = 5

        self.stem = STEM(spatial=self.spatial, temporal=self.temporal, conv=self.conv, C_in=self.C_in, C_out=32, with_bn=self.with_bn, \
            bn_before_actfn=self.bn_before_actfn, temporal_length=7, spatial_length=3)

        self.offset_1 = ConvOffset2D_nonlocal3(32, spatial=self.nonlocal_spatial, temporal=self.nonlocal_temporal, \
            concat=self.nonlocal_concat, if_self=self.nonlocal_self, path_length=self.path_length)
        self.offset_2 = ConvOffset2D_nonlocal3(32, spatial=self.nonlocal_spatial, temporal=self.nonlocal_temporal, \
            concat=self.nonlocal_concat, if_self=self.nonlocal_self, path_length=self.path_length)
        self.offset_3 = ConvOffset2D_nonlocal3(32, spatial=self.nonlocal_spatial, temporal=self.nonlocal_temporal, \
            concat=self.nonlocal_concat, if_self=self.nonlocal_self, path_length=self.path_length)
        self.offset_4 = ConvOffset2D_nonlocal3(32, spatial=self.nonlocal_spatial, temporal=self.nonlocal_temporal, \
            concat=self.nonlocal_concat, if_self=self.nonlocal_self, path_length=self.path_length)

        self.sig1 = SigModuleParallel_cheng(32, 3, 32, 3, win_size=self.path_length, use_bottleneck=True, spatial_ps=self.spatial_ps)

        self.nonlocal_fusion = nn.Sequential()
        self.nonlocal_fusion.add_module("conv", nn.Conv2d(32 * 4, 32, kernel_size=(1, 1), padding=(0, 0)))
        if self.with_bn and self.bn_before_actfn:
            self.nonlocal_fusion.add_module("bn", nn.BatchNorm2d(32))
        self.nonlocal_fusion.add_module("relu", nn.ReLU())
        if self.with_bn and not self.bn_before_actfn:
            self.nonlocal_fusion.add_module("bn", nn.BatchNorm2d(32))

        self.sig_fusion = nn.Sequential()
        self.sig_fusion.add_module("conv", nn.Conv2d(32 * 4, 32, kernel_size=(1, 1), padding=(0, 0)))
        if self.with_bn and self.bn_before_actfn:
            self.sig_fusion.add_module("bn", nn.BatchNorm2d(32))
        self.sig_fusion.add_module("relu", nn.ReLU())
        if self.with_bn and not self.bn_before_actfn:
            self.sig_fusion.add_module("bn", nn.BatchNorm2d(32))

        self.conv3 = nn.Sequential()
        self.conv3.add_module("conv", nn.Conv2d(32, 32, kernel_size=(1, 3), padding=(0, 1)))
        if self.with_bn and self.bn_before_actfn:
            self.conv3.add_module("bn", nn.BatchNorm2d(32))
        self.conv3.add_module("relu", nn.ReLU())
        if self.with_bn and not self.bn_before_actfn:
            self.conv3.add_module("bn", nn.BatchNorm2d(32))

        self.pool3 = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 2))

        self.conv_fused = nn.Sequential()
        self.conv_fused.add_module("conv", nn.Conv2d(32 + 32, 48, kernel_size=(1, 3), padding=(0, 1)))
        if self.with_bn and self.bn_before_actfn:
            self.conv_fused.add_module("bn", nn.BatchNorm2d(48))
        self.conv_fused.add_module("relu", nn.ReLU())
        if self.with_bn and not self.bn_before_actfn:
            self.conv_fused.add_module("bn", nn.BatchNorm2d(48))
        
        self.pool_fused = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 2))

        self.fc1 = nn.Linear(self.C_out, 256)
        self.dp1 = nn.Dropout(p=dropout_rate)
        self.fc2 = nn.Linear(256, self.cls_num)

        self.apply(weight_init)


    def forward(self, x):
        x_in = input_prepare(x)
        # x_in.shape = (256, 3, 11, 39)

        x_in = self.stem(x_in)

        x_in = self.conv3(x_in)
        x_in = self.pool3(x_in)

        x_nonlocal_1, x_in_off_1, offset_1 = self.offset_1(x_in)
        x_nonlocal_2, x_in_off_2, offset_2 = self.offset_2(x_in)
        x_nonlocal_3, x_in_off_3, offset_3 = self.offset_3(x_in)
        x_nonlocal_4, x_in_off_4, offset_4 = self.offset_4(x_in)

        x_nonlocal = torch.cat([x_nonlocal_1, x_nonlocal_2, x_nonlocal_3, x_nonlocal_4], dim=0)
        x_in_off = torch.cat([x_in_off_1, x_in_off_2, x_in_off_3, x_in_off_4], dim=0)
        offset = torch.cat([offset_1, offset_2, offset_3, offset_4], dim=0)

        x_in_ps = self.sig1(x_in_off)

        B, C, J, T = x_in.shape
        x_in_ps = torch.reshape(x_in_ps, (4, B, 32, J, T)).permute((1, 0, 2, 3, 4)).reshape((B, -1, J, T))
        x_in_ps = self.sig_fusion(x_in_ps)

        x_nonlocal = torch.reshape(x_nonlocal, (4, B, 32, J, T)).permute((1, 0, 2, 3, 4)).reshape((B, -1, J, T))
        x_nonlocal = self.nonlocal_fusion(x_nonlocal)

        x_in = x_in + x_nonlocal
        x_in = torch.cat([x_in, x_in_ps], 1)

        x_4 = self.conv_fused(x_in)
        x_4 = self.pool_fused(x_4)
        x_4 = x_4.view(x_4.shape[0], -1)

        x = self.fc1(x_4)
        x = self.dp1(x)
        x = self.fc2(x)
        
        # offset = torch.Tensor.reshape(offset, (4, B, 32, 11, 39, 3)).float().mean(dim=0).squeeze()
        # return x, offset
        return x


class PROPOSED_ORIGIN(nn.Module):
    def __init__(self, with_bn=True, bn_before_actfn=True, joint_num=44, spatial_ps=False,dropout_rate=0.5):
        super(PROPOSED_ORIGIN, self).__init__()

        self.C_in = 2
        self.J = joint_num
        self.T = 39
        self.C_out = 48 * joint_num * 9
        self.cls_num = 249

        self.with_bn = with_bn
        self.bn_before_actfn = bn_before_actfn
        self.spatial_ps = spatial_ps

        self.increase_dimension_early = True
        self.spatial = True
        self.temporal = True
        self.conv = True

        self.non_local = True
        self.double = False

        self.nonlocal_spatial = True
        self.nonlocal_temporal = False
        self.nonlocal_concat = False
        self.nonlocal_self = True

        self.stem = nn.Sequential()
        self.stem.add_module("conv", nn.Conv2d(2, 32, kernel_size=1, padding=0))
        if self.with_bn and self.bn_before_actfn:
            self.stem.add_module("bn", nn.BatchNorm2d(32))
        self.stem.add_module("relu", nn.ReLU())
        if self.with_bn and not self.bn_before_actfn:
            self.stem.add_module("bn", nn.BatchNorm2d(32))

        self.conv3 = nn.Sequential()
        self.conv3.add_module("conv", nn.Conv2d(32, 32, kernel_size=(1, 3), padding=(0, 1)))
        if self.with_bn and self.bn_before_actfn:
            self.conv3.add_module("bn", nn.BatchNorm2d(32))
        self.conv3.add_module("relu", nn.ReLU())
        if self.with_bn and not self.bn_before_actfn:
            self.conv3.add_module("bn", nn.BatchNorm2d(32))

        self.pool3 = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 2))

        self.conv_fused = nn.Sequential()
        self.conv_fused.add_module("conv", nn.Conv2d(32, 48, kernel_size=(1, 3), padding=(0, 1)))
        if self.with_bn and self.bn_before_actfn:
            self.conv_fused.add_module("bn", nn.BatchNorm2d(48))
        self.conv_fused.add_module("relu", nn.ReLU())
        if self.with_bn and not self.bn_before_actfn:
            self.conv_fused.add_module("bn", nn.BatchNorm2d(48))
        
        self.pool_fused = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 2))

        self.fc1 = nn.Linear(self.C_out, 256)
        self.dp1 = nn.Dropout(p=dropout_rate)
        self.fc2 = nn.Linear(256, self.cls_num)

        self.apply(weight_init)


    def forward(self, x):
        x_in = input_prepare(x)
        # x_in.shape = (256, 3, 11, 39)

        x_in = self.stem(x_in)

        x_in = self.conv3(x_in)
        x_in = self.pool3(x_in)

        x_4 = self.conv_fused(x_in)
        x_4 = self.pool_fused(x_4)
        x_4 = x_4.view(x_4.shape[0], -1)

        x = self.fc1(x_4)
        x = self.dp1(x)
        x = self.fc2(x)
        
        # offset = torch.Tensor.reshape(offset, (4, B, 32, 11, 39, 3)).float().mean(dim=0).squeeze()
        # return x, offset
        return x


class PROPOSED_STPSM(nn.Module):
    def __init__(self, with_bn=True, bn_before_actfn=True, joint_num=44, spatial_ps=False,dropout_rate=0.5):
        super(PROPOSED_STPSM, self).__init__()

        self.C_in = 2
        self.J = joint_num
        self.T = 39
        self.C_out = 48 * joint_num * 9
        self.cls_num = 249

        self.with_bn = with_bn
        self.bn_before_actfn = bn_before_actfn
        self.spatial_ps = spatial_ps

        self.increase_dimension_early = True
        self.spatial = False
        self.temporal = True
        self.conv = True

        self.non_local = True
        self.double = False

        self.nonlocal_spatial = True
        self.nonlocal_temporal = False
        self.nonlocal_concat = False
        self.nonlocal_self = True

        self.stem = STEM(spatial=self.spatial, temporal=self.temporal, conv=self.conv, C_in=self.C_in, C_out=32, with_bn=self.with_bn, \
            bn_before_actfn=self.bn_before_actfn, temporal_length=7, spatial_length=3)

        self.conv3 = nn.Sequential()
        self.conv3.add_module("conv", nn.Conv2d(32, 32, kernel_size=(1, 3), padding=(0, 1)))
        if self.with_bn and self.bn_before_actfn:
            self.conv3.add_module("bn", nn.BatchNorm2d(32))
        self.conv3.add_module("relu", nn.ReLU())
        if self.with_bn and not self.bn_before_actfn:
            self.conv3.add_module("bn", nn.BatchNorm2d(32))

        self.pool3 = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 2))

        self.conv_fused = nn.Sequential()
        self.conv_fused.add_module("conv", nn.Conv2d(32, 48, kernel_size=(1, 3), padding=(0, 1)))
        if self.with_bn and self.bn_before_actfn:
            self.conv_fused.add_module("bn", nn.BatchNorm2d(48))
        self.conv_fused.add_module("relu", nn.ReLU())
        if self.with_bn and not self.bn_before_actfn:
            self.conv_fused.add_module("bn", nn.BatchNorm2d(48))
        
        self.pool_fused = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 2))

        self.fc1 = nn.Linear(self.C_out, 256)
        self.dp1 = nn.Dropout(p=dropout_rate)
        self.fc2 = nn.Linear(256, self.cls_num)

        self.apply(weight_init)


    def forward(self, x):
        x_in = input_prepare(x)
        # x_in.shape = (256, 3, 11, 39)

        x_in = self.stem(x_in)

        x_in = self.conv3(x_in)
        x_in = self.pool3(x_in)

        x_4 = self.conv_fused(x_in)
        x_4 = self.pool_fused(x_4)
        x_4 = x_4.view(x_4.shape[0], -1)

        x = self.fc1(x_4)
        x = self.dp1(x)
        x = self.fc2(x)
        
        # offset = torch.Tensor.reshape(offset, (4, B, 32, 11, 39, 3)).float().mean(dim=0).squeeze()
        # return x, offset
        return x


class PROPOSED_LSTPSM(nn.Module):
    def __init__(self, with_bn=True, bn_before_actfn=True, joint_num=44, spatial_ps=False,dropout_rate=0.5):
        super(PROPOSED_LSTPSM, self).__init__()

        self.C_in = 2
        self.J = joint_num
        self.T = 39
        self.C_out = 48 * joint_num * 9
        self.cls_num = 249

        self.with_bn = with_bn
        self.bn_before_actfn = bn_before_actfn
        self.spatial_ps = spatial_ps

        self.increase_dimension_early = True
        self.spatial = False
        self.temporal = True
        self.conv = True

        self.non_local = True
        self.double = False

        self.nonlocal_spatial = True
        self.nonlocal_temporal = False
        self.nonlocal_concat = False
        self.nonlocal_self = True

        self.stem = L_STEM(C_in=self.C_in, C_out=32, with_bn=self.with_bn, bn_before_actfn=self.bn_before_actfn, temporal_length=7, spatial_length=3)

        self.conv3 = nn.Sequential()
        self.conv3.add_module("conv", nn.Conv2d(32, 32, kernel_size=(1, 3), padding=(0, 1)))
        if self.with_bn and self.bn_before_actfn:
            self.conv3.add_module("bn", nn.BatchNorm2d(32))
        self.conv3.add_module("relu", nn.ReLU())
        if self.with_bn and not self.bn_before_actfn:
            self.conv3.add_module("bn", nn.BatchNorm2d(32))

        self.pool3 = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 2))

        self.conv_fused = nn.Sequential()
        self.conv_fused.add_module("conv", nn.Conv2d(32, 48, kernel_size=(1, 3), padding=(0, 1)))
        if self.with_bn and self.bn_before_actfn:
            self.conv_fused.add_module("bn", nn.BatchNorm2d(48))
        self.conv_fused.add_module("relu", nn.ReLU())
        if self.with_bn and not self.bn_before_actfn:
            self.conv_fused.add_module("bn", nn.BatchNorm2d(48))
        
        self.pool_fused = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 2))

        self.fc1 = nn.Linear(self.C_out, 256)
        self.dp1 = nn.Dropout(p=dropout_rate)
        self.fc2 = nn.Linear(256, self.cls_num)

        self.apply(weight_init)


    def forward(self, x):
        x_in = input_prepare(x)
        # x_in.shape = (256, 3, 11, 39)

        x_in = self.stem(x_in)

        x_in = self.conv3(x_in)
        x_in = self.pool3(x_in)

        x_4 = self.conv_fused(x_in)
        x_4 = self.pool_fused(x_4)
        x_4 = x_4.view(x_4.shape[0], -1)

        x = self.fc1(x_4)
        x = self.dp1(x)
        x = self.fc2(x)
        
        # offset = torch.Tensor.reshape(offset, (4, B, 32, 11, 39, 3)).float().mean(dim=0).squeeze()
        # return x, offset
        return x


class PROPOSED_LSTPSM_MULTIHEAD(nn.Module):
    def __init__(self, with_bn=True, bn_before_actfn=True, joint_num=44, spatial_ps=False,dropout_rate=0.5):
        super(PROPOSED_LSTPSM_MULTIHEAD, self).__init__()

        self.C_in = 2
        self.J = joint_num
        self.T = 39
        self.C_out = 48 * joint_num * 9
        self.cls_num = 249

        self.with_bn = with_bn
        self.bn_before_actfn = bn_before_actfn
        self.spatial_ps = spatial_ps

        self.increase_dimension_early = True
        self.spatial = False
        self.temporal = True
        self.conv = True

        self.non_local = True
        self.double = False

        self.nonlocal_spatial = True
        self.nonlocal_temporal = False
        self.nonlocal_concat = False
        self.nonlocal_self = True

        self.stem_1 = L_STEM(C_in=self.C_in, C_out=32, with_bn=self.with_bn, bn_before_actfn=self.bn_before_actfn, temporal_length=7, spatial_length=3)
        self.stem_2 = L_STEM(C_in=self.C_in, C_out=32, with_bn=self.with_bn, bn_before_actfn=self.bn_before_actfn, temporal_length=7, spatial_length=3)
        self.stem_3 = L_STEM(C_in=self.C_in, C_out=32, with_bn=self.with_bn, bn_before_actfn=self.bn_before_actfn, temporal_length=7, spatial_length=3)
        self.stem_4 = L_STEM(C_in=self.C_in, C_out=32, with_bn=self.with_bn, bn_before_actfn=self.bn_before_actfn, temporal_length=7, spatial_length=3)

        self.stem_fusion = nn.Sequential()
        self.stem_fusion.add_module("conv", nn.Conv2d(32 * 4, 32, kernel_size=1, padding=0))
        if self.with_bn and self.bn_before_actfn:
            self.stem_fusion.add_module("bn", nn.BatchNorm2d(32))
        self.stem_fusion.add_module("relu", nn.ReLU())
        if self.with_bn and not self.bn_before_actfn:
            self.stem_fusion.add_module("bn", nn.BatchNorm2d(32))

        self.conv3 = nn.Sequential()
        self.conv3.add_module("conv", nn.Conv2d(32, 32, kernel_size=(1, 3), padding=(0, 1)))
        if self.with_bn and self.bn_before_actfn:
            self.conv3.add_module("bn", nn.BatchNorm2d(32))
        self.conv3.add_module("relu", nn.ReLU())
        if self.with_bn and not self.bn_before_actfn:
            self.conv3.add_module("bn", nn.BatchNorm2d(32))

        self.pool3 = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 2))

        self.conv_fused = nn.Sequential()
        self.conv_fused.add_module("conv", nn.Conv2d(32, 48, kernel_size=(1, 3), padding=(0, 1)))
        if self.with_bn and self.bn_before_actfn:
            self.conv_fused.add_module("bn", nn.BatchNorm2d(48))
        self.conv_fused.add_module("relu", nn.ReLU())
        if self.with_bn and not self.bn_before_actfn:
            self.conv_fused.add_module("bn", nn.BatchNorm2d(48))
        
        self.pool_fused = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 2))

        self.fc1 = nn.Linear(self.C_out, 256)
        self.dp1 = nn.Dropout(p=dropout_rate)
        self.fc2 = nn.Linear(256, self.cls_num)

        self.apply(weight_init)


    def forward(self, x):
        x_in = input_prepare(x)
        # x_in.shape = (256, 3, 11, 39)

        x_in_1 = self.stem_1(x_in)
        x_in_2 = self.stem_2(x_in)
        x_in_3 = self.stem_3(x_in)
        x_in_4 = self.stem_4(x_in)

        # x_in = torch.cat([x_in_1, x_in_2], dim=1)
        x_in = torch.cat([x_in_1, x_in_2, x_in_3, x_in_4], dim=1)
        x_in = self.stem_fusion(x_in)

        x_in = self.conv3(x_in)
        x_in = self.pool3(x_in)

        x_4 = self.conv_fused(x_in)
        x_4 = self.pool_fused(x_4)
        x_4 = x_4.view(x_4.shape[0], -1)

        x = self.fc1(x_4)
        x = self.dp1(x)
        x = self.fc2(x)
        
        # offset = torch.Tensor.reshape(offset, (4, B, 32, 11, 39, 3)).float().mean(dim=0).squeeze()
        # return x, offset
        return x


class PROPOSED_LPSM(nn.Module):
    def __init__(self, with_bn=True, bn_before_actfn=True, joint_num=44, spatial_ps=False,dropout_rate=0.5):
        super(PROPOSED_LPSM, self).__init__()

        self.C_in = 2
        self.J = joint_num
        self.T = 39
        self.C_out = 48 * joint_num * 9
        self.cls_num = 249

        self.with_bn = with_bn
        self.bn_before_actfn = bn_before_actfn
        self.spatial_ps = spatial_ps

        self.increase_dimension_early = True
        self.spatial = True
        self.temporal = True
        self.conv = True

        self.non_local = True
        self.double = False

        self.nonlocal_spatial = True
        self.nonlocal_temporal = False
        self.nonlocal_concat = False
        self.nonlocal_self = True

        self.path_length = 5

        self.stem = nn.Sequential()
        self.stem.add_module("conv", nn.Conv2d(2, 32, kernel_size=1, padding=0))
        if self.with_bn and self.bn_before_actfn:
            self.stem.add_module("bn", nn.BatchNorm2d(32))
        self.stem.add_module("relu", nn.ReLU())
        if self.with_bn and not self.bn_before_actfn:
            self.stem.add_module("bn", nn.BatchNorm2d(32))

        self.offset_1 = ConvOffset2D_nonlocal3(32, spatial=self.nonlocal_spatial, temporal=self.nonlocal_temporal, \
            concat=self.nonlocal_concat, if_self=self.nonlocal_self, path_length=self.path_length)
        self.offset_2 = ConvOffset2D_nonlocal3(32, spatial=self.nonlocal_spatial, temporal=self.nonlocal_temporal, \
            concat=self.nonlocal_concat, if_self=self.nonlocal_self, path_length=self.path_length)
        self.offset_3 = ConvOffset2D_nonlocal3(32, spatial=self.nonlocal_spatial, temporal=self.nonlocal_temporal, \
            concat=self.nonlocal_concat, if_self=self.nonlocal_self, path_length=self.path_length)
        self.offset_4 = ConvOffset2D_nonlocal3(32, spatial=self.nonlocal_spatial, temporal=self.nonlocal_temporal, \
            concat=self.nonlocal_concat, if_self=self.nonlocal_self, path_length=self.path_length)

        self.sig1 = SigModuleParallel_cheng(32, 3, 32, 3, win_size=self.path_length, use_bottleneck=True, spatial_ps=self.spatial_ps)

        self.nonlocal_fusion = nn.Sequential()
        self.nonlocal_fusion.add_module("conv", nn.Conv2d(32 * 4, 32, kernel_size=(1, 1), padding=(0, 0)))
        if self.with_bn and self.bn_before_actfn:
            self.nonlocal_fusion.add_module("bn", nn.BatchNorm2d(32))
        self.nonlocal_fusion.add_module("relu", nn.ReLU())
        if self.with_bn and not self.bn_before_actfn:
            self.nonlocal_fusion.add_module("bn", nn.BatchNorm2d(32))

        self.sig_fusion = nn.Sequential()
        self.sig_fusion.add_module("conv", nn.Conv2d(32 * 4, 32, kernel_size=(1, 1), padding=(0, 0)))
        if self.with_bn and self.bn_before_actfn:
            self.sig_fusion.add_module("bn", nn.BatchNorm2d(32))
        self.sig_fusion.add_module("relu", nn.ReLU())
        if self.with_bn and not self.bn_before_actfn:
            self.sig_fusion.add_module("bn", nn.BatchNorm2d(32))

        self.conv3 = nn.Sequential()
        self.conv3.add_module("conv", nn.Conv2d(32, 32, kernel_size=(1, 3), padding=(0, 1)))
        if self.with_bn and self.bn_before_actfn:
            self.conv3.add_module("bn", nn.BatchNorm2d(32))
        self.conv3.add_module("relu", nn.ReLU())
        if self.with_bn and not self.bn_before_actfn:
            self.conv3.add_module("bn", nn.BatchNorm2d(32))

        self.pool3 = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 2))

        self.conv_fused = nn.Sequential()
        self.conv_fused.add_module("conv", nn.Conv2d(32 + 32, 48, kernel_size=(1, 3), padding=(0, 1)))
        if self.with_bn and self.bn_before_actfn:
            self.conv_fused.add_module("bn", nn.BatchNorm2d(48))
        self.conv_fused.add_module("relu", nn.ReLU())
        if self.with_bn and not self.bn_before_actfn:
            self.conv_fused.add_module("bn", nn.BatchNorm2d(48))
        
        self.pool_fused = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 2))

        self.fc1 = nn.Linear(self.C_out, 256)
        self.dp1 = nn.Dropout(p=dropout_rate)
        self.fc2 = nn.Linear(256, self.cls_num)

        self.apply(weight_init)


    def forward(self, x):
        x_in = input_prepare(x)
        # x_in.shape = (256, 3, 11, 39)

        x_in = self.stem(x_in)

        x_in = self.conv3(x_in)
        x_in = self.pool3(x_in)

        x_nonlocal_1, x_in_off_1, offset_1 = self.offset_1(x_in)
        x_nonlocal_2, x_in_off_2, offset_2 = self.offset_2(x_in)
        x_nonlocal_3, x_in_off_3, offset_3 = self.offset_3(x_in)
        x_nonlocal_4, x_in_off_4, offset_4 = self.offset_4(x_in)

        x_nonlocal = torch.cat([x_nonlocal_1, x_nonlocal_2, x_nonlocal_3, x_nonlocal_4], dim=0)
        x_in_off = torch.cat([x_in_off_1, x_in_off_2, x_in_off_3, x_in_off_4], dim=0)
        offset = torch.cat([offset_1, offset_2, offset_3, offset_4], dim=0)

        x_in_ps = self.sig1(x_in_off)

        B, C, J, T = x_in.shape
        x_in_ps = torch.reshape(x_in_ps, (4, B, 32, J, T)).permute((1, 0, 2, 3, 4)).reshape((B, -1, J, T))
        x_in_ps = self.sig_fusion(x_in_ps)

        x_nonlocal = torch.reshape(x_nonlocal, (4, B, 32, J, T)).permute((1, 0, 2, 3, 4)).reshape((B, -1, J, T))
        x_nonlocal = self.nonlocal_fusion(x_nonlocal)

        x_in = x_in + x_nonlocal
        x_in = torch.cat([x_in, x_in_ps], 1)

        x_4 = self.conv_fused(x_in)
        x_4 = self.pool_fused(x_4)
        x_4 = x_4.view(x_4.shape[0], -1)

        x = self.fc1(x_4)
        x = self.dp1(x)
        x = self.fc2(x)
        
        # offset = torch.Tensor.reshape(offset, (4, B, 32, 11, 39, 3)).float().mean(dim=0).squeeze()
        # return x, offset
        return x


class STGCN(nn.Module):
    def __init__(self, with_bn=True, bn_before_actfn=True, joint_num=11, dropout_rate=0.5):
        super(STGCN, self).__init__()
        # graph = np.concatenate((chalearn16_hand_crafted_decoupled_adjacent_matrix, np.zeros((3, 22, 22))), axis=2)
        # graph = np.concatenate((graph, np.concatenate((np.zeros((3, 22, 22)), \
        # chalearn16_hand_crafted_decoupled_adjacent_matrix), axis=2)), axis=1)
        graph = np.concatenate((chalearn16_hand_crafted_adjacent_matrix, np.zeros((22, 22))), axis=1)
        graph = np.concatenate((graph, np.concatenate((np.zeros((22, 22)), \
        chalearn16_hand_crafted_adjacent_matrix), axis=1)), axis=0)
        graph = np.expand_dims(graph, axis=0)

        A = torch.from_numpy(graph.astype(np.float32))
        self.backbone_1 = ST_GCN_18(num_class=249, in_channels=2, graph_cfg=A)

    def forward(self, x):
        x_in = input_prepare(x)
        x_in = torch.Tensor.permute(x_in, (0, 1, 3, 2)).unsqueeze(dim=4)
        # N, C, T, J, 1

        output_1 = torch.softmax(self.backbone_1(x_in), dim=1)

        return output_1


class STGCN_ST_new(nn.Module):
    def __init__(self, joint_num=55, dropout_rate=0.5):
        super(STGCN_ST_new, self).__init__()
        # graph = np.concatenate((chalearn16_hand_crafted_decoupled_adjacent_matrix, np.zeros((3, 22, 22))), axis=2)
        # graph = np.concatenate((graph, np.concatenate((np.zeros((3, 22, 22)), \
        # chalearn16_hand_crafted_decoupled_adjacent_matrix), axis=2)), axis=1)
        # graph = np.concatenate((chalearn16_hand_crafted_adjacent_matrix, np.zeros((22, 22))), axis=1)
        # graph = np.concatenate((graph, np.concatenate((np.zeros((22, 22)), \
        # chalearn16_hand_crafted_adjacent_matrix), axis=1)), axis=0)
        graph = chalearn16_hand_crafted_adjacent_matrix_new
        graph = np.expand_dims(graph, axis=0)

        self.stem = STEM_v2(C_in=2, C_out=32, spatial_length=3, temporal_length=11)

        A = torch.from_numpy(graph.astype(np.float32))
        self.backbone_1 = ST_GCN_18(num_class=249, in_channels=32, graph_cfg=A)

    def forward(self, x):
        x_in = input_prepare(x)

        x_in = self.stem(x_in)
        x_in = torch.Tensor.permute(x_in, (0, 1, 3, 2)).unsqueeze(dim=4)
        # N, C, T, J, 1

        output_1 = torch.softmax(self.backbone_1(x_in), dim=1)

        return output_1


class STGCN_2LAYER(nn.Module):
    def __init__(self, with_bn=True, bn_before_actfn=True, joint_num=11, dropout_rate=0.5):
        super(STGCN_2LAYER, self).__init__()
        graph = np.concatenate((chalearn16_hand_crafted_decoupled_adjacent_matrix, np.zeros((3, 22, 22))), axis=2)
        graph = np.concatenate((graph, np.concatenate((np.zeros((3, 22, 22)), \
        chalearn16_hand_crafted_decoupled_adjacent_matrix), axis=2)), axis=1)
        A = torch.from_numpy(graph.astype(np.float32))
        self.backbone_1 = ST_GCN_18_2LAYER(num_class=249, in_channels=2, graph_cfg=A)

    def forward(self, x):
        x_in = input_prepare(x)
        x_in = torch.Tensor.permute(x_in, (0, 1, 3, 2)).unsqueeze(dim=4)
        # N, C, T, J, 1

        output_1 = torch.softmax(self.backbone_1(x_in), dim=1)

        return output_1


class STGCN_STPSM(nn.Module):
    def __init__(self, with_bn=True, bn_before_actfn=True, joint_num=11, dropout_rate=0.5):
        super(STGCN_STPSM, self).__init__()

        self.spatial = True
        self.temporal = True
        self.conv = True

        self.stem = STEM(spatial=self.spatial, temporal=self.temporal, conv=self.conv, C_in=2, C_out=32, with_bn=with_bn, \
                         bn_before_actfn=bn_before_actfn)

        graph = np.concatenate((chalearn16_hand_crafted_decoupled_adjacent_matrix, np.zeros((3, 22, 22))), axis=2)
        graph = np.concatenate((graph, np.concatenate((np.zeros((3, 22, 22)), \
        chalearn16_hand_crafted_decoupled_adjacent_matrix), axis=2)), axis=1)
        A = torch.from_numpy(graph.astype(np.float32))
        self.backbone_1 = ST_GCN_18(num_class=249, in_channels=32, graph_cfg=A)

    def forward(self, x):
        x_in = input_prepare(x)
        x_in = self.stem(x_in)

        x_in = torch.Tensor.permute(x_in, (0, 1, 3, 2)).unsqueeze(dim=4)
        # N, C, T, J, 1

        output_1 = torch.softmax(self.backbone_1(x_in), dim=1)

        return output_1


class STGCN_ST(nn.Module):
    def __init__(self, with_bn=True, bn_before_actfn=True, joint_num=11, dropout_rate=0.5):
        super(STGCN_ST, self).__init__()

        self.spatial = True
        self.temporal = True
        self.conv = True

        self.stem = STEM_v2(spatial=self.spatial, temporal=self.temporal, conv=self.conv, C_in=2, C_out=32, with_bn=with_bn, \
                         bn_before_actfn=bn_before_actfn)

        # graph = np.concatenate((chalearn16_hand_crafted_decoupled_adjacent_matrix, np.zeros((3, 22, 22))), axis=2)
        # graph = np.concatenate((graph, np.concatenate((np.zeros((3, 22, 22)), \
        # chalearn16_hand_crafted_decoupled_adjacent_matrix), axis=2)), axis=1)
        graph = np.concatenate((chalearn16_hand_crafted_adjacent_matrix, np.zeros((22, 22))), axis=1)
        graph = np.concatenate((graph, np.concatenate((np.zeros((22, 22)), \
        chalearn16_hand_crafted_adjacent_matrix), axis=1)), axis=0)
        graph = np.expand_dims(graph, axis=0)

        A = torch.from_numpy(graph.astype(np.float32))
        # self.backbone_1 = ST_GCN_18_2LAYER(num_class=249, in_channels=32, graph_cfg=A)
        self.backbone_1 = ST_GCN_18(num_class=249, in_channels=32, graph_cfg=A)

    def forward(self, x):
        x_in = input_prepare(x)
        x_in = self.stem(x_in)

        x_in = torch.Tensor.permute(x_in, (0, 1, 3, 2)).unsqueeze(dim=4)
        # N, C, T, J, 1

        output_1 = torch.softmax(self.backbone_1(x_in), dim=1)

        return output_1


class STGCN_L(nn.Module):
    def __init__(self, with_bn=True, bn_before_actfn=True, joint_num=11, dropout_rate=0.5):
        super(STGCN_L, self).__init__()

        self.nonlocal_spatial = True
        self.nonlocal_temporal = False
        self.nonlocal_concat = False
        self.nonlocal_self = True

        self.with_bn = with_bn
        self.bn_before_actfn = bn_before_actfn
        self.spatial_ps = False

        self.path_length = 5

        self.stem = nn.Sequential()
        self.stem.add_module("conv", nn.Conv2d(2, 32, kernel_size=1, padding=0))
        self.stem.add_module("relu", nn.ReLU())
        self.stem.add_module("bn", nn.BatchNorm2d(32))

        self.offset_1 = ConvOffset2D_nonlocal3_v2(32, spatial=self.nonlocal_spatial, temporal=self.nonlocal_temporal, \
            concat=self.nonlocal_concat, if_self=self.nonlocal_self, path_length=self.path_length)
        self.offset_2 = ConvOffset2D_nonlocal3_v2(32, spatial=self.nonlocal_spatial, temporal=self.nonlocal_temporal, \
            concat=self.nonlocal_concat, if_self=self.nonlocal_self, path_length=self.path_length)
        self.offset_3 = ConvOffset2D_nonlocal3_v2(32, spatial=self.nonlocal_spatial, temporal=self.nonlocal_temporal, \
            concat=self.nonlocal_concat, if_self=self.nonlocal_self, path_length=self.path_length)
        self.offset_4 = ConvOffset2D_nonlocal3_v2(32, spatial=self.nonlocal_spatial, temporal=self.nonlocal_temporal, \
            concat=self.nonlocal_concat, if_self=self.nonlocal_self, path_length=self.path_length)

        self.sig1 = SigModuleParallel_cheng_v2(32, 3, 32, 3, win_size=self.path_length, use_bottleneck=True, spatial_ps=self.spatial_ps)

        self.nonlocal_fusion = nn.Sequential()
        self.nonlocal_fusion.add_module("conv", nn.Conv2d(32 * 4, 32, kernel_size=(1, 1), padding=(0, 0)))
        if self.with_bn and self.bn_before_actfn:
            self.nonlocal_fusion.add_module("bn", nn.BatchNorm2d(32))
        self.nonlocal_fusion.add_module("relu", nn.ReLU())
        if self.with_bn and not self.bn_before_actfn:
            self.nonlocal_fusion.add_module("bn", nn.BatchNorm2d(32))

        self.sig_fusion = nn.Sequential()
        self.sig_fusion.add_module("conv", nn.Conv2d(32 * 4, 32, kernel_size=(1, 1), padding=(0, 0)))
        if self.with_bn and self.bn_before_actfn:
            self.sig_fusion.add_module("bn", nn.BatchNorm2d(32))
        self.sig_fusion.add_module("relu", nn.ReLU())
        if self.with_bn and not self.bn_before_actfn:
            self.sig_fusion.add_module("bn", nn.BatchNorm2d(32))

        graph = np.concatenate((chalearn16_hand_crafted_decoupled_adjacent_matrix, np.zeros((3, 22, 22))), axis=2)
        graph = np.concatenate((graph, np.concatenate((np.zeros((3, 22, 22)), \
        chalearn16_hand_crafted_decoupled_adjacent_matrix), axis=2)), axis=1)
        A = torch.from_numpy(graph.astype(np.float32))
        # self.backbone_1 = ST_GCN_18_2LAYER(num_class=249, in_channels=64, graph_cfg=A)
        self.backbone_1 = ST_GCN_18(num_class=249, in_channels=64, graph_cfg=A)

    def forward(self, x):
        x_in = input_prepare(x)
        x_in = self.stem(x_in)

        x_nonlocal_1, x_in_off_1, offset_1 = self.offset_1(x_in)
        x_nonlocal_2, x_in_off_2, offset_2 = self.offset_2(x_in)
        x_nonlocal_3, x_in_off_3, offset_3 = self.offset_3(x_in)
        x_nonlocal_4, x_in_off_4, offset_4 = self.offset_4(x_in)

        x_nonlocal = torch.cat([x_nonlocal_1, x_nonlocal_2, x_nonlocal_3, x_nonlocal_4], dim=0)
        x_in_off = torch.cat([x_in_off_1, x_in_off_2, x_in_off_3, x_in_off_4], dim=0)
        offset = torch.cat([offset_1, offset_2, offset_3, offset_4], dim=0)

        x_in_ps = self.sig1(x_in_off)

        B, C, J, T = x_in.shape
        x_in_ps = torch.reshape(x_in_ps, (4, B, 32, J, T)).permute((1, 0, 2, 3, 4)).reshape((B, -1, J, T))
        x_in_ps = self.sig_fusion(x_in_ps)

        x_nonlocal = torch.reshape(x_nonlocal, (4, B, 32, J, T)).permute((1, 0, 2, 3, 4)).reshape((B, -1, J, T))
        x_nonlocal = self.nonlocal_fusion(x_nonlocal)

        x_in = x_in + x_nonlocal
        x_in = torch.cat([x_in, x_in_ps], 1)

        x_in = torch.Tensor.permute(x_in, (0, 1, 3, 2)).unsqueeze(dim=4)
        # N, C, T, J, 1

        output_1 = torch.softmax(self.backbone_1(x_in), dim=1)

        return output_1


class MSG3D(nn.Module):
    def __init__(self, joint_num=44, dropout_rate=0.5):
        super(MSG3D, self).__init__()
        # graph = np.concatenate((chalearn16_hand_crafted_adjacent_matrix, np.zeros((22, 22))), axis=1)
        # graph = np.concatenate((graph, np.concatenate((np.zeros((22, 22)), \
        # chalearn16_hand_crafted_adjacent_matrix), axis=1)), axis=0)
        graph = chalearn16_hand_crafted_adjacent_matrix_new

        self.backbone_1 = MSG3D_Model(num_class=249, num_point=55, num_person=1, num_gcn_scales=3, \
            num_g3d_scales=5, graph=graph, in_channels=2)

    def forward(self, x):
        x_in = input_prepare(x)
        x_in = torch.Tensor.permute(x_in, (0, 1, 3, 2)).unsqueeze(dim=4)
        # N, C, T, J, 1

        output = torch.softmax(self.backbone_1(x_in), dim=1)
        return output


class MSG3D_ST_new(nn.Module):
    def __init__(self, joint_num=44, dropout_rate=0.5):
        super(MSG3D_ST_new, self).__init__()
        # graph = np.concatenate((chalearn16_hand_crafted_adjacent_matrix, np.zeros((22, 22))), axis=1)
        # graph = np.concatenate((graph, np.concatenate((np.zeros((22, 22)), \
        # chalearn16_hand_crafted_adjacent_matrix), axis=1)), axis=0)
        graph = chalearn16_hand_crafted_adjacent_matrix_new

        self.stem = STEM_v2(C_in=2, C_out=32, spatial_length=3, temporal_length=11)

        self.backbone_1 = MSG3D_Model(num_class=249, num_point=55, num_person=1, num_gcn_scales=3, \
            num_g3d_scales=5, graph=graph, in_channels=32)

    def forward(self, x):
        x_in = input_prepare(x)
        x_in = self.stem(x_in)

        x_in = torch.Tensor.permute(x_in, (0, 1, 3, 2)).unsqueeze(dim=4)
        # N, C, T, J, 1

        output = torch.softmax(self.backbone_1(x_in), dim=1)
        return output


class MSG3D_2LAYER(nn.Module):
    def __init__(self, with_bn=True, bn_before_actfn=True, joint_num=44, dropout_rate=0.5):
        super(MSG3D_2LAYER, self).__init__()
        graph = np.concatenate((chalearn16_hand_crafted_adjacent_matrix, np.zeros((22, 22))), axis=1)
        graph = np.concatenate((graph, np.concatenate((np.zeros((22, 22)), \
        chalearn16_hand_crafted_adjacent_matrix), axis=1)), axis=0)

        self.backbone_1 = MSG3D_Model_2layer(num_class=249, num_point=22, num_person=2, num_gcn_scales=3, \
            num_g3d_scales=5, graph=graph, in_channels=2)

    def forward(self, x):
        x_in = input_prepare(x)
        x_in = torch.Tensor.permute(x_in, (0, 1, 3, 2)).unsqueeze(dim=4)
        # N, C, T, J, 1

        output = torch.softmax(self.backbone_1(x_in), dim=1)
        return output


class MSG3D_ST(nn.Module):
    def __init__(self, with_bn=True, bn_before_actfn=True, joint_num=44, dropout_rate=0.5):
        super(MSG3D_ST, self).__init__()

        self.spatial = True
        self.temporal = True
        self.conv = True

        self.stem = STEM_v2(spatial=self.spatial, temporal=self.temporal, conv=self.conv, C_in=2, C_out=32, with_bn=with_bn, \
                         bn_before_actfn=bn_before_actfn)

        graph = np.concatenate((chalearn16_hand_crafted_adjacent_matrix, np.zeros((22, 22))), axis=1)
        graph = np.concatenate((graph, np.concatenate((np.zeros((22, 22)), \
        chalearn16_hand_crafted_adjacent_matrix), axis=1)), axis=0)

        # self.backbone_1 = MSG3D_Model_2layer(num_class=249, num_point=22, num_person=2, num_gcn_scales=3, \
        #     num_g3d_scales=5, graph=graph, in_channels=32)
        self.backbone_1 = MSG3D_Model(num_class=249, num_point=22, num_person=2, num_gcn_scales=3, \
            num_g3d_scales=5, graph=graph, in_channels=32)

    def forward(self, x):
        x_in = input_prepare(x)
        x_in = self.stem(x_in)
        x_in = torch.Tensor.permute(x_in, (0, 1, 3, 2)).unsqueeze(dim=4)
        # N, C, T, J, 1

        output = torch.softmax(self.backbone_1(x_in), dim=1)
        return output


class MSG3D_L(nn.Module):
    def __init__(self, with_bn=True, bn_before_actfn=True, joint_num=44, dropout_rate=0.5):
        super(MSG3D_L, self).__init__()

        self.spatial = True
        self.temporal = True
        self.conv = True


        self.nonlocal_spatial = True
        self.nonlocal_temporal = False
        self.nonlocal_concat = False
        self.nonlocal_self = True

        self.with_bn = with_bn
        self.bn_before_actfn = bn_before_actfn
        self.spatial_ps = False

        self.path_length = 5

        self.stem = nn.Sequential()
        self.stem.add_module("conv", nn.Conv2d(2, 32, kernel_size=1, padding=0))
        self.stem.add_module("relu", nn.ReLU())
        self.stem.add_module("bn", nn.BatchNorm2d(32))

        self.offset_1 = ConvOffset2D_nonlocal3_v2(32, spatial=self.nonlocal_spatial, temporal=self.nonlocal_temporal, \
            concat=self.nonlocal_concat, if_self=self.nonlocal_self, path_length=self.path_length)
        self.offset_2 = ConvOffset2D_nonlocal3_v2(32, spatial=self.nonlocal_spatial, temporal=self.nonlocal_temporal, \
            concat=self.nonlocal_concat, if_self=self.nonlocal_self, path_length=self.path_length)
        self.offset_3 = ConvOffset2D_nonlocal3_v2(32, spatial=self.nonlocal_spatial, temporal=self.nonlocal_temporal, \
            concat=self.nonlocal_concat, if_self=self.nonlocal_self, path_length=self.path_length)
        self.offset_4 = ConvOffset2D_nonlocal3_v2(32, spatial=self.nonlocal_spatial, temporal=self.nonlocal_temporal, \
            concat=self.nonlocal_concat, if_self=self.nonlocal_self, path_length=self.path_length)

        self.sig1 = SigModuleParallel_cheng_v2(32, 3, 32, 3, win_size=self.path_length, use_bottleneck=True, spatial_ps=self.spatial_ps)

        self.nonlocal_fusion = nn.Sequential()
        self.nonlocal_fusion.add_module("conv", nn.Conv2d(32 * 4, 32, kernel_size=(1, 1), padding=(0, 0)))
        if self.with_bn and self.bn_before_actfn:
            self.nonlocal_fusion.add_module("bn", nn.BatchNorm2d(32))
        self.nonlocal_fusion.add_module("relu", nn.ReLU())
        if self.with_bn and not self.bn_before_actfn:
            self.nonlocal_fusion.add_module("bn", nn.BatchNorm2d(32))

        self.sig_fusion = nn.Sequential()
        self.sig_fusion.add_module("conv", nn.Conv2d(32 * 4, 32, kernel_size=(1, 1), padding=(0, 0)))
        if self.with_bn and self.bn_before_actfn:
            self.sig_fusion.add_module("bn", nn.BatchNorm2d(32))
        self.sig_fusion.add_module("relu", nn.ReLU())
        if self.with_bn and not self.bn_before_actfn:
            self.sig_fusion.add_module("bn", nn.BatchNorm2d(32))

        graph = np.concatenate((chalearn16_hand_crafted_adjacent_matrix, np.zeros((22, 22))), axis=1)
        graph = np.concatenate((graph, np.concatenate((np.zeros((22, 22)), \
        chalearn16_hand_crafted_adjacent_matrix), axis=1)), axis=0)

        # self.backbone_1 = MSG3D_Model_2layer(num_class=249, num_point=22, num_person=2, num_gcn_scales=3, \
        #     num_g3d_scales=5, graph=graph, in_channels=64)
        self.backbone_1 = MSG3D_Model(num_class=249, num_point=22, num_person=2, num_gcn_scales=3, \
            num_g3d_scales=5, graph=graph, in_channels=64)

    def forward(self, x):
        x_in = input_prepare(x)
        x_in = self.stem(x_in)

        x_nonlocal_1, x_in_off_1, offset_1 = self.offset_1(x_in)
        x_nonlocal_2, x_in_off_2, offset_2 = self.offset_2(x_in)
        x_nonlocal_3, x_in_off_3, offset_3 = self.offset_3(x_in)
        x_nonlocal_4, x_in_off_4, offset_4 = self.offset_4(x_in)

        x_nonlocal = torch.cat([x_nonlocal_1, x_nonlocal_2, x_nonlocal_3, x_nonlocal_4], dim=0)
        x_in_off = torch.cat([x_in_off_1, x_in_off_2, x_in_off_3, x_in_off_4], dim=0)
        offset = torch.cat([offset_1, offset_2, offset_3, offset_4], dim=0)

        x_in_ps = self.sig1(x_in_off)

        B, C, J, T = x_in.shape
        x_in_ps = torch.reshape(x_in_ps, (4, B, 32, J, T)).permute((1, 0, 2, 3, 4)).reshape((B, -1, J, T))
        x_in_ps = self.sig_fusion(x_in_ps)

        x_nonlocal = torch.reshape(x_nonlocal, (4, B, 32, J, T)).permute((1, 0, 2, 3, 4)).reshape((B, -1, J, T))
        x_nonlocal = self.nonlocal_fusion(x_nonlocal)

        x_in = x_in + x_nonlocal
        x_in = torch.cat([x_in, x_in_ps], 1)

        x_in = torch.Tensor.permute(x_in, (0, 1, 3, 2)).unsqueeze(dim=4)
        # N, C, T, J, 1

        output = torch.softmax(self.backbone_1(x_in), dim=1)
        return output


class AGCN(nn.Module):
    def __init__(self, with_bn=True, bn_before_actfn=True, joint_num=44, dropout_rate=0.5):
        super(AGCN, self).__init__()

        graph = np.concatenate((chalearn16_hand_crafted_adjacent_matrix, np.zeros((22, 22))), axis=1)
        graph = np.concatenate((graph, np.concatenate((np.zeros((22, 22)), \
        chalearn16_hand_crafted_adjacent_matrix), axis=1)), axis=0)    

        self.backbone_1 = AGCN_Model(num_class=249, num_point=22, num_person=2, \
            graph=graph, in_channels=2)

    def forward(self, x):
        x_in = input_prepare(x)
        x_in = torch.Tensor.permute(x_in, (0, 1, 3, 2)).unsqueeze(dim=4)
        # N, C, T, J, 1

        output = torch.softmax(self.backbone_1(x_in), dim=1)
        return output


class AGCN_2LAYER(nn.Module):
    def __init__(self, with_bn=True, bn_before_actfn=True, joint_num=44, dropout_rate=0.5):
        super(AGCN_2LAYER, self).__init__()

        graph = np.concatenate((chalearn16_hand_crafted_adjacent_matrix, np.zeros((22, 22))), axis=1)
        graph = np.concatenate((graph, np.concatenate((np.zeros((22, 22)), \
        chalearn16_hand_crafted_adjacent_matrix), axis=1)), axis=0)    

        self.backbone_1 = AGCN_Model(num_class=249, num_point=22, num_person=2, \
            graph=graph, in_channels=2)

    def forward(self, x):
        x_in = input_prepare(x)
        x_in = torch.Tensor.permute(x_in, (0, 1, 3, 2)).unsqueeze(dim=4)
        # N, C, T, J, 1

        output = torch.softmax(self.backbone_1(x_in), dim=1)
        return output


class AGCN_ST(nn.Module):
    def __init__(self, with_bn=True, bn_before_actfn=True, joint_num=44, dropout_rate=0.5):
        super(AGCN_ST, self).__init__()

        self.spatial = True
        self.temporal = True
        self.conv = True

        self.stem = STEM(spatial=self.spatial, temporal=self.temporal, conv=self.conv, C_in=2, C_out=32, with_bn=with_bn, \
                         bn_before_actfn=bn_before_actfn)

        graph = np.concatenate((chalearn16_hand_crafted_adjacent_matrix, np.zeros((22, 22))), axis=1)
        graph = np.concatenate((graph, np.concatenate((np.zeros((22, 22)), \
        chalearn16_hand_crafted_adjacent_matrix), axis=1)), axis=0)    

        self.backbone_1 = AGCN_Model(num_class=249, num_point=22, num_person=2, \
            graph=graph, in_channels=32)

    def forward(self, x):
        x_in = input_prepare(x)
        x_in = self.stem(x_in)

        x_in = torch.Tensor.permute(x_in, (0, 1, 3, 2)).unsqueeze(dim=4)
        # N, C, T, J, 1

        output = torch.softmax(self.backbone_1(x_in), dim=1)
        return output


class AGCN_L(nn.Module):
    def __init__(self, with_bn=True, bn_before_actfn=True, joint_num=44, dropout_rate=0.5):
        super(AGCN_L, self).__init__()

        self.nonlocal_spatial = True
        self.nonlocal_temporal = False
        self.nonlocal_concat = False
        self.nonlocal_self = True

        self.with_bn = with_bn
        self.bn_before_actfn = bn_before_actfn
        self.spatial_ps = False

        self.path_length = 5

        self.stem = nn.Sequential()
        self.stem.add_module("conv", nn.Conv2d(2, 32, kernel_size=1, padding=0))
        self.stem.add_module("relu", nn.ReLU())
        self.stem.add_module("bn", nn.BatchNorm2d(32))

        self.offset_1 = ConvOffset2D_nonlocal3(32, spatial=self.nonlocal_spatial, temporal=self.nonlocal_temporal, \
            concat=self.nonlocal_concat, if_self=self.nonlocal_self, path_length=self.path_length)
        self.offset_2 = ConvOffset2D_nonlocal3(32, spatial=self.nonlocal_spatial, temporal=self.nonlocal_temporal, \
            concat=self.nonlocal_concat, if_self=self.nonlocal_self, path_length=self.path_length)
        self.offset_3 = ConvOffset2D_nonlocal3(32, spatial=self.nonlocal_spatial, temporal=self.nonlocal_temporal, \
            concat=self.nonlocal_concat, if_self=self.nonlocal_self, path_length=self.path_length)
        self.offset_4 = ConvOffset2D_nonlocal3(32, spatial=self.nonlocal_spatial, temporal=self.nonlocal_temporal, \
            concat=self.nonlocal_concat, if_self=self.nonlocal_self, path_length=self.path_length)

        self.sig1 = SigModuleParallel_cheng(32, 3, 32, 3, win_size=self.path_length, use_bottleneck=True, spatial_ps=self.spatial_ps)

        self.nonlocal_fusion = nn.Sequential()
        self.nonlocal_fusion.add_module("conv", nn.Conv2d(32 * 4, 32, kernel_size=(1, 1), padding=(0, 0)))
        if self.with_bn and self.bn_before_actfn:
            self.nonlocal_fusion.add_module("bn", nn.BatchNorm2d(32))
        self.nonlocal_fusion.add_module("relu", nn.ReLU())
        if self.with_bn and not self.bn_before_actfn:
            self.nonlocal_fusion.add_module("bn", nn.BatchNorm2d(32))

        self.sig_fusion = nn.Sequential()
        self.sig_fusion.add_module("conv", nn.Conv2d(32 * 4, 32, kernel_size=(1, 1), padding=(0, 0)))
        if self.with_bn and self.bn_before_actfn:
            self.sig_fusion.add_module("bn", nn.BatchNorm2d(32))
        self.sig_fusion.add_module("relu", nn.ReLU())
        if self.with_bn and not self.bn_before_actfn:
            self.sig_fusion.add_module("bn", nn.BatchNorm2d(32))

        graph = np.concatenate((chalearn16_hand_crafted_adjacent_matrix, np.zeros((22, 22))), axis=1)
        graph = np.concatenate((graph, np.concatenate((np.zeros((22, 22)), \
        chalearn16_hand_crafted_adjacent_matrix), axis=1)), axis=0)    

        self.backbone_1 = AGCN_Model(num_class=249, num_point=22, num_person=2, \
            graph=graph, in_channels=64)

    def forward(self, x):
        x_in = input_prepare(x)
        x_in = self.stem(x_in)

        x_nonlocal_1, x_in_off_1, offset_1 = self.offset_1(x_in)
        x_nonlocal_2, x_in_off_2, offset_2 = self.offset_2(x_in)
        x_nonlocal_3, x_in_off_3, offset_3 = self.offset_3(x_in)
        x_nonlocal_4, x_in_off_4, offset_4 = self.offset_4(x_in)

        x_nonlocal = torch.cat([x_nonlocal_1, x_nonlocal_2, x_nonlocal_3, x_nonlocal_4], dim=0)
        x_in_off = torch.cat([x_in_off_1, x_in_off_2, x_in_off_3, x_in_off_4], dim=0)
        offset = torch.cat([offset_1, offset_2, offset_3, offset_4], dim=0)

        x_in_ps = self.sig1(x_in_off)

        B, C, J, T = x_in.shape
        x_in_ps = torch.reshape(x_in_ps, (4, B, 32, J, T)).permute((1, 0, 2, 3, 4)).reshape((B, -1, J, T))
        x_in_ps = self.sig_fusion(x_in_ps)

        x_nonlocal = torch.reshape(x_nonlocal, (4, B, 32, J, T)).permute((1, 0, 2, 3, 4)).reshape((B, -1, J, T))
        x_nonlocal = self.nonlocal_fusion(x_nonlocal)

        x_in = x_in + x_nonlocal
        x_in = torch.cat([x_in, x_in_ps], 1)

        x_in = torch.Tensor.permute(x_in, (0, 1, 3, 2)).unsqueeze(dim=4)
        # N, C, T, J, 1

        output = torch.softmax(self.backbone_1(x_in), dim=1)
        return output


class ShiftGCN(nn.Module):
    def __init__(self, joint_num=55, dropout_rate=0.5):
        super(ShiftGCN, self).__init__()

        # graph = np.concatenate((chalearn16_hand_crafted_adjacent_matrix, np.zeros((22, 22))), axis=1)
        # graph = np.concatenate((graph, np.concatenate((np.zeros((22, 22)), \
        # chalearn16_hand_crafted_adjacent_matrix), axis=1)), axis=0)
        graph = chalearn16_hand_crafted_adjacent_matrix_new

        self.padding = nn.ReplicationPad2d((1, 0, 0, 0))
        self.backbone_1 = ShiftGCN_Model(num_class=249, num_point=55, num_person=1, \
            graph=graph, in_channels=2)


    def forward(self, x):
        x_in = input_prepare(x)
        x_in = self.padding(x_in)
        x_in = torch.Tensor.permute(x_in, (0, 1, 3, 2)).unsqueeze(dim=4)
        # N, C, T, J, 1

        output_1 = torch.softmax(self.backbone_1(x_in), dim=1)

        return output_1


class ShiftGCN_ST_new(nn.Module):
    def __init__(self, joint_num=55, dropout_rate=0.5):
        super(ShiftGCN_ST_new, self).__init__()

        # graph = np.concatenate((chalearn16_hand_crafted_adjacent_matrix, np.zeros((22, 22))), axis=1)
        # graph = np.concatenate((graph, np.concatenate((np.zeros((22, 22)), \
        # chalearn16_hand_crafted_adjacent_matrix), axis=1)), axis=0)
        graph = chalearn16_hand_crafted_adjacent_matrix_new

        self.stem = STEM_v2(C_in=2, C_out=32, spatial_length=3, temporal_length=11)

        self.padding = nn.ReplicationPad2d((1, 0, 0, 0))
        self.backbone_1 = ShiftGCN_Model(num_class=249, num_point=55, num_person=1, \
            graph=graph, in_channels=32)


    def forward(self, x):
        x_in = input_prepare(x)
        x_in = self.stem(x_in)
        x_in = self.padding(x_in)
        x_in = torch.Tensor.permute(x_in, (0, 1, 3, 2)).unsqueeze(dim=4)
        # N, C, T, J, 1

        output_1 = torch.softmax(self.backbone_1(x_in), dim=1)

        return output_1


class ShiftGCN_ST(nn.Module):
    def __init__(self, with_bn=True, bn_before_actfn=True, joint_num=11, dropout_rate=0.5):
        super(ShiftGCN_ST, self).__init__()

        self.spatial = True
        self.temporal = True
        self.conv = True

        self.stem = STEM_v2(spatial=self.spatial, temporal=self.temporal, conv=self.conv, C_in=2, C_out=32, with_bn=with_bn, \
                         bn_before_actfn=bn_before_actfn)

        graph = np.concatenate((chalearn16_hand_crafted_adjacent_matrix, np.zeros((22, 22))), axis=1)
        graph = np.concatenate((graph, np.concatenate((np.zeros((22, 22)), \
        chalearn16_hand_crafted_adjacent_matrix), axis=1)), axis=0)

        self.padding = nn.ReplicationPad2d((1, 0, 0, 0))
        self.backbone_1 = ShiftGCN_Model(num_class=249, num_point=44, num_person=1, \
            graph=graph, in_channels=32)


    def forward(self, x):
        x_in = input_prepare(x)
        x_in = self.stem(x_in)
        x_in = self.padding(x_in)
        # import pdb
        # pdb.set_trace()
        x_in = torch.Tensor.permute(x_in, (0, 1, 3, 2)).unsqueeze(dim=4)
        # N, C, T, J, 1

        output_1 = torch.softmax(self.backbone_1(x_in), dim=1)

        return output_1


class ShiftGCN_L(nn.Module):
    def __init__(self, with_bn=True, bn_before_actfn=True, joint_num=11, dropout_rate=0.5):
        super(ShiftGCN_L, self).__init__()

        self.spatial = True
        self.temporal = True
        self.conv = True


        self.nonlocal_spatial = True
        self.nonlocal_temporal = False
        self.nonlocal_concat = False
        self.nonlocal_self = True

        self.with_bn = with_bn
        self.bn_before_actfn = bn_before_actfn
        self.spatial_ps = False

        self.path_length = 5

        self.stem = nn.Sequential()
        self.stem.add_module("conv", nn.Conv2d(2, 32, kernel_size=1, padding=0))
        self.stem.add_module("relu", nn.ReLU())
        self.stem.add_module("bn", nn.BatchNorm2d(32))

        self.offset_1 = ConvOffset2D_nonlocal3_v2(32, spatial=self.nonlocal_spatial, temporal=self.nonlocal_temporal, \
            concat=self.nonlocal_concat, if_self=self.nonlocal_self, path_length=self.path_length)
        self.offset_2 = ConvOffset2D_nonlocal3_v2(32, spatial=self.nonlocal_spatial, temporal=self.nonlocal_temporal, \
            concat=self.nonlocal_concat, if_self=self.nonlocal_self, path_length=self.path_length)
        self.offset_3 = ConvOffset2D_nonlocal3_v2(32, spatial=self.nonlocal_spatial, temporal=self.nonlocal_temporal, \
            concat=self.nonlocal_concat, if_self=self.nonlocal_self, path_length=self.path_length)
        self.offset_4 = ConvOffset2D_nonlocal3_v2(32, spatial=self.nonlocal_spatial, temporal=self.nonlocal_temporal, \
            concat=self.nonlocal_concat, if_self=self.nonlocal_self, path_length=self.path_length)

        self.sig1 = SigModuleParallel_cheng_v2(32, 3, 32, 3, win_size=self.path_length, use_bottleneck=True, spatial_ps=self.spatial_ps)

        self.nonlocal_fusion = nn.Sequential()
        self.nonlocal_fusion.add_module("conv", nn.Conv2d(32 * 4, 32, kernel_size=(1, 1), padding=(0, 0)))
        if self.with_bn and self.bn_before_actfn:
            self.nonlocal_fusion.add_module("bn", nn.BatchNorm2d(32))
        self.nonlocal_fusion.add_module("relu", nn.ReLU())
        if self.with_bn and not self.bn_before_actfn:
            self.nonlocal_fusion.add_module("bn", nn.BatchNorm2d(32))

        self.sig_fusion = nn.Sequential()
        self.sig_fusion.add_module("conv", nn.Conv2d(32 * 4, 32, kernel_size=(1, 1), padding=(0, 0)))
        if self.with_bn and self.bn_before_actfn:
            self.sig_fusion.add_module("bn", nn.BatchNorm2d(32))
        self.sig_fusion.add_module("relu", nn.ReLU())
        if self.with_bn and not self.bn_before_actfn:
            self.sig_fusion.add_module("bn", nn.BatchNorm2d(32))

        graph = np.concatenate((chalearn16_hand_crafted_adjacent_matrix, np.zeros((22, 22))), axis=1)
        graph = np.concatenate((graph, np.concatenate((np.zeros((22, 22)), \
        chalearn16_hand_crafted_adjacent_matrix), axis=1)), axis=0)

        self.padding = nn.ReplicationPad2d((1, 0, 0, 0))
        self.backbone_1 = ShiftGCN_Model(num_class=249, num_point=44, num_person=1, \
            graph=graph, in_channels=64)


    def forward(self, x):
        x_in = input_prepare(x)
        x_in = self.stem(x_in)

        x_nonlocal_1, x_in_off_1, offset_1 = self.offset_1(x_in)
        x_nonlocal_2, x_in_off_2, offset_2 = self.offset_2(x_in)
        x_nonlocal_3, x_in_off_3, offset_3 = self.offset_3(x_in)
        x_nonlocal_4, x_in_off_4, offset_4 = self.offset_4(x_in)

        x_nonlocal = torch.cat([x_nonlocal_1, x_nonlocal_2, x_nonlocal_3, x_nonlocal_4], dim=0)
        x_in_off = torch.cat([x_in_off_1, x_in_off_2, x_in_off_3, x_in_off_4], dim=0)
        offset = torch.cat([offset_1, offset_2, offset_3, offset_4], dim=0)

        x_in_ps = self.sig1(x_in_off)

        B, C, J, T = x_in.shape
        x_in_ps = torch.reshape(x_in_ps, (4, B, 32, J, T)).permute((1, 0, 2, 3, 4)).reshape((B, -1, J, T))
        x_in_ps = self.sig_fusion(x_in_ps)

        x_nonlocal = torch.reshape(x_nonlocal, (4, B, 32, J, T)).permute((1, 0, 2, 3, 4)).reshape((B, -1, J, T))
        x_nonlocal = self.nonlocal_fusion(x_nonlocal)

        x_in = x_in + x_nonlocal
        x_in = torch.cat([x_in, x_in_ps], 1)
        x_in = self.padding(x_in)
        x_in = torch.Tensor.permute(x_in, (0, 1, 3, 2)).unsqueeze(dim=4)
        # N, C, T, J, 1

        output_1 = torch.softmax(self.backbone_1(x_in), dim=1)

        return output_1


class SLGCN_ST_new(nn.Module):
    def __init__(self, joint_num=55, dropout_rate=0.5):
        super(SLGCN_ST_new, self).__init__()
        # graph = np.concatenate((chalearn16_hand_crafted_adjacent_matrix, np.zeros((22, 22))), axis=1)
        # graph = np.concatenate((graph, np.concatenate((np.zeros((22, 22)), \
        # chalearn16_hand_crafted_adjacent_matrix), axis=1)), axis=0)
        # graph = np.expand_dims(graph, axis=0).repeat(3, axis=0)
        graph = np.expand_dims(chalearn16_hand_crafted_adjacent_matrix_new, axis=0).repeat(3, axis=0)

        self.stem = STEM_v2(C_in=2, C_out=32, spatial_length=3, temporal_length=11)

        # import pdb
        # pdb.set_trace()
        # print(graph.shape)
        # A = torch.from_numpy(graph.astype(np.float32))
        self.backbone_1 = SLGCN_Model(num_class=249, num_point=55, num_person=1, \
            graph=graph, in_channels=32, groups=8)

    def forward(self, x):
        x_in = input_prepare(x)
        x_in = self.stem(x_in)
        x_in = torch.Tensor.permute(x_in, (0, 1, 3, 2)).unsqueeze(dim=4)
        # N, C, T, J, 1

        output_1 = torch.softmax(self.backbone_1(x_in), dim=1)

        return output_1


class Proposed_new(nn.Module):
    def __init__(self, path_length, spatial_length, temporal_length, joint_num=55, dropout_rate=0.5):
        super(Proposed_new, self).__init__()

        self.C_in = 2
        self.J = joint_num
        self.T = 39
        self.C_out = 48 * joint_num * 9
        self.cls_num = 249
        self.path_length = path_length
        self.spatial_length = spatial_length
        self.temporal_length = temporal_length

        self.stem = STEM_v2(C_in=2, C_out=32, spatial_length=self.spatial_length, temporal_length=self.temporal_length)

        self.conv1 = nn.Sequential()
        self.conv1.add_module("conv", nn.Conv2d(32, 32, kernel_size=(1, 3), padding=(0, 1)))
        self.conv1.add_module("bn", nn.BatchNorm2d(32))
        self.conv1.add_module("relu", nn.ReLU())

        self.pool1 = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 2))

        self.offset_1 = ConvOffset2D_nonlocal3_v2(32, path_length=self.path_length)
        self.offset_2 = ConvOffset2D_nonlocal3_v2(32, path_length=self.path_length)
        self.offset_3 = ConvOffset2D_nonlocal3_v2(32, path_length=self.path_length)
        self.offset_4 = ConvOffset2D_nonlocal3_v2(32, path_length=self.path_length)

        self.sig1 = SigModuleParallel_cheng_v2(32, 3, 32, 3, win_size=self.path_length)

        self.nonlocal_fusion = nn.Sequential()
        self.nonlocal_fusion.add_module("conv", nn.Conv2d(32 * 4, 32, kernel_size=(1, 1), padding=(0, 0)))
        self.nonlocal_fusion.add_module("bn", nn.BatchNorm2d(32))
        self.nonlocal_fusion.add_module("relu", nn.ReLU())

        self.sig_fusion = nn.Sequential()
        self.sig_fusion.add_module("conv", nn.Conv2d(32 * 4, 32, kernel_size=(1, 1), padding=(0, 0)))
        self.sig_fusion.add_module("bn", nn.BatchNorm2d(32))
        self.sig_fusion.add_module("relu", nn.ReLU())

        self.conv2 = nn.Sequential()
        self.conv2.add_module("conv", nn.Conv2d(32 * 2, 48, kernel_size=(1, 3), padding=(0, 1)))
        self.conv2.add_module("bn", nn.BatchNorm2d(48))
        self.conv2.add_module("relu", nn.ReLU())
        
        self.pool2 = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 2))

        self.fc1 = nn.Linear(self.C_out, 256)
        self.dp1 = nn.Dropout(p=dropout_rate)
        self.fc2 = nn.Linear(256, self.cls_num)

        self.apply(weight_init)


    def forward(self, x):
        x_in = input_prepare(x)
        # x_in.shape = (256, 3, 11, 39)

        x_in = self.stem(x_in)

        x_in = self.conv1(x_in)
        x_in = self.pool1(x_in)

        x_nonlocal_1, x_in_off_1, offset_1 = self.offset_1(x_in)
        x_nonlocal_2, x_in_off_2, offset_2 = self.offset_2(x_in)
        x_nonlocal_3, x_in_off_3, offset_3 = self.offset_3(x_in)
        x_nonlocal_4, x_in_off_4, offset_4 = self.offset_4(x_in)

        x_nonlocal = torch.cat([x_nonlocal_1, x_nonlocal_2, x_nonlocal_3, x_nonlocal_4], dim=0)
        x_in_off = torch.cat([x_in_off_1, x_in_off_2, x_in_off_3, x_in_off_4], dim=0)
        # offset = torch.cat([offset_1, offset_2, offset_3, offset_4], dim=0)

        x_in_ps = self.sig1(x_in_off)

        B, C, J, T = x_in.shape
        x_in_ps = torch.reshape(x_in_ps, (4, B, 32, J, T)).permute((1, 0, 2, 3, 4)).reshape((B, -1, J, T))
        x_in_ps = self.sig_fusion(x_in_ps)

        x_nonlocal = torch.reshape(x_nonlocal, (4, B, 32, J, T)).permute((1, 0, 2, 3, 4)).reshape((B, -1, J, T))
        x_nonlocal = self.nonlocal_fusion(x_nonlocal)

        x_in = x_in + x_nonlocal
        x_in = torch.cat([x_in, x_in_ps], 1)

        x_in = self.conv2(x_in)
        x_in = self.pool2(x_in)
        x_in = x_in.view(x_in.shape[0], -1)

        x_in = self.fc1(x_in)
        x_in = self.dp1(x_in)
        x_in = self.fc2(x_in)
        
        # offset = torch.Tensor.reshape(offset, (4, B, 32, 11, 39, 3)).float().mean(dim=0).squeeze()
        # return x, offset
        return x_in


class Proposed_new_origin_ps(nn.Module):
    def __init__(self, path_length=3, spatial_length=3, temporal_length=3, joint_num=44, dropout_rate=0.6):
        super(Proposed_new_origin_ps, self).__init__()

        self.C_in = 3
        self.J = joint_num
        self.T = 39
        self.C_out = 48 * joint_num * 9
        self.cls_num = 20
        self.path_length = path_length
        self.spatial_length = spatial_length
        self.temporal_length = temporal_length

        self.stem = STEM_v2_origin_ps(C_in=self.C_in, C_out=32, spatial_length=self.spatial_length, temporal_length=self.temporal_length)

        self.conv1 = nn.Sequential()
        self.conv1.add_module("conv", nn.Conv2d(32, 32, kernel_size=(1, 3), padding=(0, 1)))
        self.conv1.add_module("bn", nn.BatchNorm2d(32))
        self.conv1.add_module("relu", nn.ReLU())

        self.pool1 = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 2))

        self.offset_1 = ConvOffset2D_nonlocal3_v2(32, path_length=self.path_length)
        self.offset_2 = ConvOffset2D_nonlocal3_v2(32, path_length=self.path_length)
        self.offset_3 = ConvOffset2D_nonlocal3_v2(32, path_length=self.path_length)
        self.offset_4 = ConvOffset2D_nonlocal3_v2(32, path_length=self.path_length)

        self.sig1 = SigModuleParallel_cheng_v2_origin_ps(32, 32, 32, 3, win_size=self.path_length)

        self.nonlocal_fusion = nn.Sequential()
        self.nonlocal_fusion.add_module("conv", nn.Conv2d(32 * 4, 32, kernel_size=(1, 1), padding=(0, 0)))
        self.nonlocal_fusion.add_module("bn", nn.BatchNorm2d(32))
        self.nonlocal_fusion.add_module("relu", nn.ReLU())

        self.sig_fusion = nn.Sequential()
        self.sig_fusion.add_module("conv", nn.Conv2d(37059 * 4, 32, kernel_size=(1, 1), padding=(0, 0)))
        self.sig_fusion.add_module("bn", nn.BatchNorm2d(32))
        self.sig_fusion.add_module("relu", nn.ReLU())

        self.conv2 = nn.Sequential()
        self.conv2.add_module("conv", nn.Conv2d(32 * 2, 48, kernel_size=(1, 3), padding=(0, 1)))
        self.conv2.add_module("bn", nn.BatchNorm2d(48))
        self.conv2.add_module("relu", nn.ReLU())
        
        self.pool2 = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 2))

        self.fc1 = nn.Linear(self.C_out, 256)
        self.dp1 = nn.Dropout(p=dropout_rate)
        self.fc2 = nn.Linear(256, self.cls_num)

        self.apply(weight_init)


    def forward(self, x):
        x_in = input_prepare(x)
        # x_in.shape = (256, 3, 11, 39)

        x_in = self.stem(x_in)

        x_in = self.conv1(x_in)
        x_in = self.pool1(x_in)

        x_nonlocal_1, x_in_off_1, offset_1 = self.offset_1(x_in)
        x_nonlocal_2, x_in_off_2, offset_2 = self.offset_2(x_in)
        x_nonlocal_3, x_in_off_3, offset_3 = self.offset_3(x_in)
        x_nonlocal_4, x_in_off_4, offset_4 = self.offset_4(x_in)

        x_nonlocal = torch.cat([x_nonlocal_1, x_nonlocal_2, x_nonlocal_3, x_nonlocal_4], dim=0)
        x_in_off = torch.cat([x_in_off_1, x_in_off_2, x_in_off_3, x_in_off_4], dim=0)
        # offset = torch.cat([offset_1, offset_2, offset_3, offset_4], dim=0)

        x_in_ps = self.sig1(x_in_off)

        B, C, J, T = x_in.shape
        x_in_ps = torch.reshape(x_in_ps, (4, B, 37059, J, T)).permute((1, 0, 2, 3, 4)).reshape((B, -1, J, T))
        x_in_ps = self.sig_fusion(x_in_ps)

        x_nonlocal = torch.reshape(x_nonlocal, (4, B, 32, J, T)).permute((1, 0, 2, 3, 4)).reshape((B, -1, J, T))
        x_nonlocal = self.nonlocal_fusion(x_nonlocal)

        x_in = x_in + x_nonlocal
        x_in = torch.cat([x_in, x_in_ps], 1)

        x_in = self.conv2(x_in)
        x_in = self.pool2(x_in)
        x_in = x_in.view(x_in.shape[0], -1)

        x_in = self.fc1(x_in)
        x_in = self.dp1(x_in)
        x_in = self.fc2(x_in)
        
        # offset = torch.Tensor.reshape(offset, (4, B, 32, 11, 39, 3)).float().mean(dim=0).squeeze()
        # return x, offset
        return x_in


class Proposed_new_part_w3(nn.Module):
    def __init__(self, path_length, spatial_length, temporal_length, joint_num=55, dropout_rate=0.5):
        super(Proposed_new_part_w3, self).__init__()

        self.C_in = 3
        self.J = joint_num
        self.T = 39
        # self.C_out = 48 * joint_num * 9
        self.C_out = 48 * 9
        self.cls_num = 249
        self.path_length = path_length
        self.spatial_length = spatial_length
        self.temporal_length = temporal_length

        self.stem = STEM_v2(C_in=self.C_in, C_out=32, spatial_length=self.spatial_length, temporal_length=self.temporal_length)

        self.conv1 = nn.Sequential()
        self.conv1.add_module("conv", nn.Conv2d(32, 32, kernel_size=(1, 3), padding=(0, 1)))
        self.conv1.add_module("bn", nn.BatchNorm2d(32))
        self.conv1.add_module("relu", nn.ReLU())

        self.pool1 = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 2))

        self.offset_1 = ConvOffset2D_nonlocal3_v2(32, path_length=self.path_length)
        self.offset_2 = ConvOffset2D_nonlocal3_v2(32, path_length=self.path_length)
        self.offset_3 = ConvOffset2D_nonlocal3_v2(32, path_length=self.path_length)
        self.offset_4 = ConvOffset2D_nonlocal3_v2(32, path_length=self.path_length)

        self.sig1 = SigModuleParallel_cheng_v2(32, 3, 32, 3, win_size=self.path_length)

        self.nonlocal_fusion = nn.Sequential()
        self.nonlocal_fusion.add_module("conv", nn.Conv2d(32 * 4, 32, kernel_size=(1, 1), padding=(0, 0)))
        self.nonlocal_fusion.add_module("bn", nn.BatchNorm2d(32))
        self.nonlocal_fusion.add_module("relu", nn.ReLU())

        self.sig_fusion = nn.Sequential()
        self.sig_fusion.add_module("conv", nn.Conv2d(32 * 4, 32, kernel_size=(1, 1), padding=(0, 0)))
        self.sig_fusion.add_module("bn", nn.BatchNorm2d(32))
        self.sig_fusion.add_module("relu", nn.ReLU())

        self.conv2 = nn.Sequential()
        self.conv2.add_module("conv", nn.Conv2d(32 * 2, 48, kernel_size=(1, 3), padding=(0, 1)))
        self.conv2.add_module("bn", nn.BatchNorm2d(48))
        self.conv2.add_module("relu", nn.ReLU())
        
        self.pool2 = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 2))

        self.fc1 = nn.Linear(int(self.C_out * 13), 256)
        self.dp1 = nn.Dropout(p=dropout_rate)
        self.out1 = nn.Linear(256, self.cls_num)
        self.weight1 = nn.Linear(256 + 13, 1)

        self.fc2 = nn.Linear(int(self.C_out * 21), 256)
        self.dp2 = nn.Dropout(p=dropout_rate)
        self.out2 = nn.Linear(256, self.cls_num)
        self.weight2 = nn.Linear(256 + 21, 1)

        self.fc3 = nn.Linear(int(self.C_out * 21), 256)
        self.dp3 = nn.Dropout(p=dropout_rate)
        self.out3 = nn.Linear(256, self.cls_num)
        self.weight3 = nn.Linear(256 + 21, 1)

        # self.weight = nn.Parameter(torch.FloatTensor(3))

        self.apply(weight_init)


    def forward(self, x):
        x_in = input_prepare(x)
        xp = x_in[:, 2:, :, :].mean(dim=3).permute(0, 2, 1).squeeze()

        # x_in.shape = (256, 3, 11, 39)

        x_in = self.stem(x_in)

        x_in = self.conv1(x_in)
        x_in = self.pool1(x_in)

        x_nonlocal_1, x_in_off_1, offset_1 = self.offset_1(x_in)
        x_nonlocal_2, x_in_off_2, offset_2 = self.offset_2(x_in)
        x_nonlocal_3, x_in_off_3, offset_3 = self.offset_3(x_in)
        x_nonlocal_4, x_in_off_4, offset_4 = self.offset_4(x_in)

        x_nonlocal = torch.cat([x_nonlocal_1, x_nonlocal_2, x_nonlocal_3, x_nonlocal_4], dim=0)
        x_in_off = torch.cat([x_in_off_1, x_in_off_2, x_in_off_3, x_in_off_4], dim=0)
        # offset = torch.cat([offset_1, offset_2, offset_3, offset_4], dim=0)

        x_in_ps = self.sig1(x_in_off)

        B, C, J, T = x_in.shape
        x_in_ps = torch.reshape(x_in_ps, (4, B, 32, J, T)).permute((1, 0, 2, 3, 4)).reshape((B, -1, J, T))
        x_in_ps = self.sig_fusion(x_in_ps)

        x_nonlocal = torch.reshape(x_nonlocal, (4, B, 32, J, T)).permute((1, 0, 2, 3, 4)).reshape((B, -1, J, T))
        x_nonlocal = self.nonlocal_fusion(x_nonlocal)

        x_in = x_in + x_nonlocal
        x_in = torch.cat([x_in, x_in_ps], 1)

        x_in = self.conv2(x_in)
        x_in = self.pool2(x_in)
        # x_in = x_in.view(x_in.shape[0], -1)
        # x_in = x_in.permute(0, 2, 1, 3).reshape(B, J, -1)
        x_in1 = x_in[:, :, :13, :].reshape(B, -1)
        x_in2 = x_in[:, :, 13:34, :].reshape(B, -1)
        x_in3 = x_in[:, :, 34:, :].reshape(B, -1)

        x_in1 = self.fc1(x_in1)
        x_in1 = self.dp1(x_in1)
        weight1 = self.weight1(torch.cat((x_in1, xp[:, :13]), dim=1))
        x_in1 = self.out1(x_in1)

        x_in2 = self.fc2(x_in2)
        x_in2 = self.dp2(x_in2)
        weight2 = self.weight2(torch.cat((x_in2, xp[:, 13:34]), dim=1))
        x_in2 = self.out2(x_in2)

        x_in3 = self.fc3(x_in3)
        x_in3 = self.dp3(x_in3)
        weight3 = self.weight3(torch.cat((x_in3, xp[:, 34:]), dim=1))
        x_in3 = self.out3(x_in3)

        x_in = torch.cat((x_in1.unsqueeze(dim=1), x_in2.unsqueeze(dim=1), x_in3.unsqueeze(dim=1)), dim=1)
        weight = torch.cat((weight1, weight2, weight3), dim=1)
        weight = torch.Tensor.softmax(weight, dim=1)
        # weight = torch.Tensor.repeat(weight.unsqueeze(dim=0), (B, 1))
        # pdb.set_trace()
        x_in = torch.einsum('bn, bnc->bc', weight, x_in)
        # xp1 = torch.Tensor.mean(xp[:, :13, :], dim=1)
        # xp2 = torch.Tensor.mean(xp[:, :13, :], dim=1)
        # xp3 = torch.Tensor.mean(xp[:, :13, :], dim=1)

        # x_in = (x_in1 * xp1 + x_in2 * xp2 + x_in3 * xp3) / 3
        # x_in = x_in.mean(dim=1)
        
        # offset = torch.Tensor.reshape(offset, (4, B, 32, 11, 39, 3)).float().mean(dim=0).squeeze()
        # return x, offset
        return x_in


class Proposed_new_part_w3_woxp(nn.Module):
    def __init__(self, path_length, spatial_length, temporal_length, joint_num=55, dropout_rate=0.5):
        super(Proposed_new_part_w3_woxp, self).__init__()

        self.C_in = 3
        self.J = joint_num
        self.T = 39
        # self.C_out = 48 * joint_num * 9
        self.C_out = 48 * 9
        self.cls_num = 249
        self.path_length = path_length
        self.spatial_length = spatial_length
        self.temporal_length = temporal_length

        self.stem = STEM_v2(C_in=self.C_in, C_out=32, spatial_length=self.spatial_length, temporal_length=self.temporal_length)

        self.conv1 = nn.Sequential()
        self.conv1.add_module("conv", nn.Conv2d(32, 32, kernel_size=(1, 3), padding=(0, 1)))
        self.conv1.add_module("bn", nn.BatchNorm2d(32))
        self.conv1.add_module("relu", nn.ReLU())

        self.pool1 = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 2))

        self.offset_1 = ConvOffset2D_nonlocal3_v2(32, path_length=self.path_length)
        self.offset_2 = ConvOffset2D_nonlocal3_v2(32, path_length=self.path_length)
        self.offset_3 = ConvOffset2D_nonlocal3_v2(32, path_length=self.path_length)
        self.offset_4 = ConvOffset2D_nonlocal3_v2(32, path_length=self.path_length)

        self.sig1 = SigModuleParallel_cheng_v2(32, 3, 32, 3, win_size=self.path_length)

        self.nonlocal_fusion = nn.Sequential()
        self.nonlocal_fusion.add_module("conv", nn.Conv2d(32 * 4, 32, kernel_size=(1, 1), padding=(0, 0)))
        self.nonlocal_fusion.add_module("bn", nn.BatchNorm2d(32))
        self.nonlocal_fusion.add_module("relu", nn.ReLU())

        self.sig_fusion = nn.Sequential()
        self.sig_fusion.add_module("conv", nn.Conv2d(32 * 4, 32, kernel_size=(1, 1), padding=(0, 0)))
        self.sig_fusion.add_module("bn", nn.BatchNorm2d(32))
        self.sig_fusion.add_module("relu", nn.ReLU())

        self.conv2 = nn.Sequential()
        self.conv2.add_module("conv", nn.Conv2d(32 * 2, 48, kernel_size=(1, 3), padding=(0, 1)))
        self.conv2.add_module("bn", nn.BatchNorm2d(48))
        self.conv2.add_module("relu", nn.ReLU())
        
        self.pool2 = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 2))

        self.fc1 = nn.Linear(int(self.C_out * 13), 256)
        self.dp1 = nn.Dropout(p=dropout_rate)
        self.out1 = nn.Linear(256, self.cls_num)
        self.weight1 = nn.Linear(256, 1)

        self.fc2 = nn.Linear(int(self.C_out * 21), 256)
        self.dp2 = nn.Dropout(p=dropout_rate)
        self.out2 = nn.Linear(256, self.cls_num)
        self.weight2 = nn.Linear(256, 1)

        self.fc3 = nn.Linear(int(self.C_out * 21), 256)
        self.dp3 = nn.Dropout(p=dropout_rate)
        self.out3 = nn.Linear(256, self.cls_num)
        self.weight3 = nn.Linear(256, 1)

        # self.weight = nn.Parameter(torch.FloatTensor(3))

        self.apply(weight_init)


    def forward(self, x):
        x_in = input_prepare(x)
        # xp = x_in[:, 2:, :, :].mean(dim=3).permute(0, 2, 1).squeeze()

        # x_in.shape = (256, 3, 11, 39)

        x_in = self.stem(x_in)

        x_in = self.conv1(x_in)
        x_in = self.pool1(x_in)

        x_nonlocal_1, x_in_off_1, offset_1 = self.offset_1(x_in)
        x_nonlocal_2, x_in_off_2, offset_2 = self.offset_2(x_in)
        x_nonlocal_3, x_in_off_3, offset_3 = self.offset_3(x_in)
        x_nonlocal_4, x_in_off_4, offset_4 = self.offset_4(x_in)

        x_nonlocal = torch.cat([x_nonlocal_1, x_nonlocal_2, x_nonlocal_3, x_nonlocal_4], dim=0)
        x_in_off = torch.cat([x_in_off_1, x_in_off_2, x_in_off_3, x_in_off_4], dim=0)
        # offset = torch.cat([offset_1, offset_2, offset_3, offset_4], dim=0)

        x_in_ps = self.sig1(x_in_off)

        B, C, J, T = x_in.shape
        x_in_ps = torch.reshape(x_in_ps, (4, B, 32, J, T)).permute((1, 0, 2, 3, 4)).reshape((B, -1, J, T))
        x_in_ps = self.sig_fusion(x_in_ps)

        x_nonlocal = torch.reshape(x_nonlocal, (4, B, 32, J, T)).permute((1, 0, 2, 3, 4)).reshape((B, -1, J, T))
        x_nonlocal = self.nonlocal_fusion(x_nonlocal)

        x_in = x_in + x_nonlocal
        x_in = torch.cat([x_in, x_in_ps], 1)

        x_in = self.conv2(x_in)
        x_in = self.pool2(x_in)
        # x_in = x_in.view(x_in.shape[0], -1)
        # x_in = x_in.permute(0, 2, 1, 3).reshape(B, J, -1)
        x_in1 = x_in[:, :, :13, :].reshape(B, -1)
        x_in2 = x_in[:, :, 13:34, :].reshape(B, -1)
        x_in3 = x_in[:, :, 34:, :].reshape(B, -1)

        x_in1 = self.fc1(x_in1)
        x_in1 = self.dp1(x_in1)
        # weight1 = self.weight1(torch.cat((x_in1, xp[:, :13]), dim=1))
        weight1 = self.weight1(x_in1)
        x_in1 = self.out1(x_in1)

        x_in2 = self.fc2(x_in2)
        x_in2 = self.dp2(x_in2)
        weight2 = self.weight2(x_in2)
        x_in2 = self.out2(x_in2)

        x_in3 = self.fc3(x_in3)
        x_in3 = self.dp3(x_in3)
        weight3 = self.weight3(x_in3)
        x_in3 = self.out3(x_in3)

        x_in = torch.cat((x_in1.unsqueeze(dim=1), x_in2.unsqueeze(dim=1), x_in3.unsqueeze(dim=1)), dim=1)
        weight = torch.cat((weight1, weight2, weight3), dim=1)
        weight = torch.Tensor.softmax(weight, dim=1)
        # weight = torch.Tensor.repeat(weight.unsqueeze(dim=0), (B, 1))
        # pdb.set_trace()
        x_in = torch.einsum('bn, bnc->bc', weight, x_in)
        # xp1 = torch.Tensor.mean(xp[:, :13, :], dim=1)
        # xp2 = torch.Tensor.mean(xp[:, :13, :], dim=1)
        # xp3 = torch.Tensor.mean(xp[:, :13, :], dim=1)

        # x_in = (x_in1 * xp1 + x_in2 * xp2 + x_in3 * xp3) / 3
        # x_in = x_in.mean(dim=1)
        
        # offset = torch.Tensor.reshape(offset, (4, B, 32, 11, 39, 3)).float().mean(dim=0).squeeze()
        # return x, offset
        return x_in



if __name__ == "__main__":
    # x = torch.randn(1,12)
    pass
