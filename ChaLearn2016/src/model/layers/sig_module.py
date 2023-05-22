'''
Author      : now more
Contact     : lin.honghui@qq.com
LastEditors: Please set LastEditors
Description : 
LastEditTime: 2020-10-16 02:29:44
'''
import torch
import torch.nn as nn
import signatory
import numpy as np
from src.model.layers.non_local import *
import pdb

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


chalearn16_hand_crafted_spatial_path_3 = np.array([
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

chalearn16_hand_crafted_spatial_path_5 = np.array([
    [10, 1, 0, 1, 10],
    [1, 0, 1, 10, 11], 
    [0, 1, 2, 3, 4],
    [1, 2, 3, 4, 5],
    [2, 3, 4, 5, 4],
    [3, 4, 5, 4, 3],
    [0, 1, 6, 7, 8],
    [1, 6, 7, 8, 9],
    [6, 7, 8, 9, 8],
    [7, 8, 9, 8, 7],
    [0, 1, 10, 11, 12],
    [1, 10, 11, 12, 13],
    [10, 11, 12, 13, 12],
    [11, 12, 13, 12, 11],
    [0, 1, 14, 15, 16],
    [1, 14, 15, 16, 17],
    [14, 15, 16, 17, 16],
    [15, 16, 17, 16, 15],
    [0, 1, 18, 19, 20],
    [17, 18, 19, 20, 19],
    [18, 19, 20, 21, 20],
    [19, 20, 21, 20, 19]
]).astype(np.float32)

chalearn16_hand_crafted_spatial_path_7 = np.array([
    [11, 10, 1, 0, 1, 10, 11],
    [10, 1, 0, 1, 10, 11, 12], 
    [1, 0, 1, 2, 3, 4, 5],
    [0, 1, 2, 3, 4, 5, 4],
    [1, 2, 3, 4, 5, 4, 3],
    [2, 3, 4, 5, 4, 3, 2],
    [1, 0, 1, 6, 7, 8, 9],
    [0, 1, 6, 7, 8, 9, 8],
    [1, 6, 7, 8, 9, 8, 7],
    [6, 7, 8, 9, 8, 7, 6],
    [1, 0, 1, 10, 11, 12, 13],
    [0, 1, 10, 11, 12, 13, 12],
    [1, 10, 11, 12, 13, 12, 11],
    [10, 11, 12, 13, 12, 11, 10],
    [1, 0, 1, 14, 15, 16, 17],
    [0, 1, 14, 15, 16, 17, 16],
    [1, 14, 15, 16, 17, 16, 15],
    [14, 15, 16, 17, 16, 15, 14],
    [1, 0, 1, 18, 19, 20, 21],
    [0, 1, 18, 19, 20, 19, 18],
    [1, 18, 19, 20, 21, 20, 19],
    [18, 19, 20, 21, 20, 19, 18]
]).astype(np.float32)

chalearn16_hand_crafted_spatial_path_9 = np.array([
    [12, 11, 10, 1, 0, 1, 10, 11, 12],
    [11, 10, 1, 0, 1, 10, 11, 12, 13], 
    [2, 1, 0, 1, 2, 3, 4, 5, 4],
    [1, 0, 1, 2, 3, 4, 5, 4, 3],
    [0, 1, 2, 3, 4, 5, 4, 3, 2],
    [1, 2, 3, 4, 5, 4, 3, 2, 1],
    [6, 1, 0, 1, 6, 7, 8, 9, 8],
    [1, 0, 1, 6, 7, 8, 9, 8, 7],
    [0, 1, 6, 7, 8, 9, 8, 7, 6],
    [1, 6, 7, 8, 9, 8, 7, 6, 1],
    [10, 1, 0, 1, 10, 11, 12, 13, 12],
    [1, 0, 1, 10, 11, 12, 13, 12, 11],
    [0, 1, 10, 11, 12, 13, 12, 11, 10],
    [1, 10, 11, 12, 13, 12, 11, 10, 1],
    [14, 1, 0, 1, 14, 15, 16, 17, 16],
    [1, 0, 1, 14, 15, 16, 17, 16, 15],
    [0, 1, 14, 15, 16, 17, 16, 15, 14],
    [1, 14, 15, 16, 17, 16, 15, 14, 1],
    [18, 1, 0, 1, 18, 19, 20, 21, 20],
    [1, 0, 1, 18, 19, 20, 19, 18, 1],
    [0, 1, 18, 19, 20, 21, 20, 19, 18],
    [1, 18, 19, 20, 21, 20, 19, 18, 1]
]).astype(np.float32)

chalearn16_hand_crafted_spatial_path_7_new = np.array([
    [11, 9, 7, 1, 6, 8, 10],
    [3, 2, 4, 2, 1, 6, 8], 
    [2, 3, 5, 3, 1, 7, 9],
    [5, 3, 2, 4, 1, 7, 9],
    [4, 2, 3, 5, 1, 6, 8],
    [7, 13, 12, 6, 8, 10, 14],
    [6, 12, 13, 7, 9, 11, 35],
    [7, 1, 6, 8, 10, 14, 23],
    [6, 1, 7, 9, 11, 35, 44],
    [12, 6, 8, 10, 14, 23, 26],
    [13, 7, 9, 11, 35, 44, 47],
    [9, 7, 13, 12, 6, 1, 2],
    [8, 6, 12, 13, 7, 1, 3],
    [6, 8, 10, 14, 23, 26, 18],
    [26, 23, 14, 15, 16, 17, 18],
    [23, 14, 15, 16, 17, 18, 22],
    [14, 15, 16, 17, 18, 22, 19],
    [15, 16, 17, 18, 22, 19, 14],
    [26, 23, 14, 19, 20, 21, 22],
    [23, 14, 19, 20, 21, 22, 18],
    [14, 19, 20, 21, 22, 18, 15],
    [19, 20, 21, 22, 18, 15, 14],
    [22, 19, 14, 23, 24, 25, 26],
    [19, 14, 23, 24, 25, 26, 18],
    [14, 23, 24, 25, 26, 18, 15],
    [23, 24, 25, 26, 18, 15, 14],
    [26, 23, 14, 27, 28, 29, 30],
    [23, 14, 27, 28, 29, 30, 18],
    [14, 27, 28, 29, 30, 18, 15],
    [27, 28, 29, 30, 18, 15, 14],
    [26, 23, 14, 31, 32, 33, 34],
    [23, 14, 31, 32, 33, 34, 18],
    [14, 31, 32, 33, 34, 18, 15],
    [31, 32, 33, 34, 18, 15, 14],
    [7, 9, 11, 35, 44, 47, 39]
]).astype(np.float32) - 1

chalearn16_hand_crafted_spatial_path_7_new = np.concatenate((chalearn16_hand_crafted_spatial_path_7_new, chalearn16_hand_crafted_spatial_path_7_new[-21:-1] + 21), axis=0)

chalearn16_hand_crafted_spatial_path_5_new = chalearn16_hand_crafted_spatial_path_7_new[:, 1:-1]

chalearn16_hand_crafted_spatial_path_3_new = chalearn16_hand_crafted_spatial_path_5_new[:, 1:-1]


class SigNet(nn.Module):
    def __init__(self, in_channels, out_dimension, sig_depth, include_time=True):
        super(SigNet, self).__init__()
        self.augment = signatory.Augment(in_channels=in_channels,
                                         layer_sizes=(),
                                         kernel_size=1,
                                         include_original=True,
                                         include_time=include_time)
        self.signature = signatory.Signature(depth=sig_depth)

        if include_time:
            # +1 because signatory.Augment is used to add time as well
            sig_channels = signatory.signature_channels(channels=in_channels + 1,
                                                        depth=sig_depth)
        else:
            sig_channels = signatory.signature_channels(channels=in_channels,
                                                        depth=sig_depth)
        # pdb.set_trace()
        self.linear = torch.nn.Linear(sig_channels,
                                      out_dimension)

    def forward(self, inp):
        # inp is a three dimensional tensor of shape (batch, stream, in_channels)
        x = self.augment(inp)
        # pdb.set_trace()

        if x.size(1) <= 1:
            raise RuntimeError("Given an input with too short a stream to take the"
                               " signature")
        # x in a three dimensional tensor of shape (batch, stream, in_channels + 1),
        # as time has been added as a value
        y = self.signature(x, basepoint=True)
        # pdb.set_trace()

        # y is a two dimensional tensor of shape (batch, terms), corresponding to
        # the terms of the signature
        # pdb.set_trace()
        z = self.linear(y)
        # z is a two dimensional tensor of shape (batch, out_dimension)

        # pdb.set_trace()

        return z


class SigNet_origin_ps(nn.Module):
    def __init__(self, in_channels, out_dimension, sig_depth, include_time=True):
        super(SigNet_origin_ps, self).__init__()
        self.augment = signatory.Augment(in_channels=in_channels,
                                         layer_sizes=(),
                                         kernel_size=1,
                                         include_original=True,
                                         include_time=include_time)
        self.signature = signatory.Signature(depth=sig_depth)

        if include_time:
            # +1 because signatory.Augment is used to add time as well
            sig_channels = signatory.signature_channels(channels=in_channels + 1,
                                                        depth=sig_depth)
        else:
            sig_channels = signatory.signature_channels(channels=in_channels,
                                                        depth=sig_depth)
        # pdb.set_trace()
        # self.linear = torch.nn.Linear(sig_channels,
                                    #   out_dimension)

    def forward(self, inp):
        # inp is a three dimensional tensor of shape (batch, stream, in_channels)
        x = self.augment(inp)
        # pdb.set_trace()

        if x.size(1) <= 1:
            raise RuntimeError("Given an input with too short a stream to take the"
                               " signature")
        # x in a three dimensional tensor of shape (batch, stream, in_channels + 1),
        # as time has been added as a value
        y = self.signature(x, basepoint=True)
        # pdb.set_trace()

        # y is a two dimensional tensor of shape (batch, terms), corresponding to
        # the terms of the signature
        # pdb.set_trace()
        # z = self.linear(y)
        # z is a two dimensional tensor of shape (batch, out_dimension)

        # pdb.set_trace()

        return y


class SigModule(nn.Module):
    def __init__(self, in_channels, sig_in_channels, out_dimension, sig_depth, \
        win_size=5, use_bottleneck=False):

        super(SigModule, self).__init__()

        self.in_channels = in_channels
        self.sig_in_channels = sig_in_channels
        self.out_dimension = out_dimension
        self.sig_depth = sig_depth
        self.win_size = win_size
        self.use_bottleneck = use_bottleneck

        if self.use_bottleneck:
            self.bottleneck = nn.Conv2d(self.in_channels, self.sig_in_channels, kernel_size=(1, 1))
            self.relu = nn.ReLU()
            self.bn = nn.BatchNorm2d(self.sig_in_channels)

        self.sig = SigNet(self.sig_in_channels, self.out_dimension, self.sig_depth)


    def forward(self, x):

        # start = time.time()

        if self.use_bottleneck:
            x = self.bn(self.relu(self.bottleneck(x)))

        N, C, J, T = x.shape
        delta_t = (self.win_size - 1) // 2

        floor_t = 0
        ceil_t = T

        self.res = torch.zeros([N, self.out_dimension, J, T], dtype=torch.float, requires_grad=True).to(x.device)

        for i in range(J):
            for j in range(T):

                start_t = np.clip(j - delta_t, floor_t, ceil_t)
                end_t = np.clip(j + delta_t + 1, floor_t, ceil_t)

                stream_cur = x[:, :, i, start_t: end_t]
                stream_cur = stream_cur.permute(0, 2, 1)    # NCT to NTC

                feat_cur = self.sig(stream_cur)

                self.res[:, :, i, j] = feat_cur

        return self.res


class SigModuleParallel(nn.Module):
    def __init__(self, in_channels, sig_in_channels, out_dimension, sig_depth, \
        win_size=5, use_bottleneck=False, specific_path=None, spatial_ps=False):

        """
            - inputs:
                specific_path: list(tuple). Specify the path. [(y1, x1), (y2, x2), ...]
        """

        super(SigModuleParallel, self).__init__()

        self.in_channels = in_channels
        self.sig_in_channels = sig_in_channels
        self.out_dimension = out_dimension
        self.sig_depth = sig_depth
        self.win_size = win_size
        self.use_bottleneck = use_bottleneck
        self.specific_path = specific_path
        self.spatial_ps = spatial_ps

        # pdb.set_trace()

        assert self.win_size % 2 == 1

        if self.use_bottleneck:
            self.bottleneck = nn.Conv2d(self.in_channels, self.sig_in_channels, kernel_size=(1, 1))
            self.relu = nn.ReLU()
            self.bn = nn.BatchNorm2d(self.sig_in_channels)


        self.delta_t = (self.win_size - 1) // 2

        if self.specific_path is None:
            self.pad = nn.ReplicationPad2d((self.delta_t, self.delta_t, 0, 0))
        else:
            self.pad = nn.ReplicationPad2d((self.delta_t, self.delta_t, self.delta_t, self.delta_t))
        # pdb.set_trace()
        self.sig = SigNet(self.sig_in_channels, self.out_dimension, self.sig_depth, include_time=False if self.spatial_ps else True)


    def forward(self, x):

        # start = time.time()

        if self.use_bottleneck:
            x = self.bn(self.relu(self.bottleneck(x)))

        N, C, J, T = x.shape

        x = self.pad(x)

        if self.specific_path is None:
            self.input = torch.zeros([J*T*N, self.win_size, C], dtype=torch.float, requires_grad=True).to(x.device)
        else:
            self.input = torch.zeros([J*T*N, len(self.specific_path), C], dtype=torch.float, requires_grad=True).to(x.device)

        self.res = torch.zeros([N, self.out_dimension, J, T], dtype=torch.float, requires_grad=True).to(x.device)


        if self.specific_path is None:
            for i in range(J):
                for j in range(T):
                    self.input[(i*T+j)*N:(i*T+j+1)*N, :, :] = x[:, :, i, j:j+self.win_size].permute(0, 2, 1)
        else:
            for i in range(self.delta_t, self.delta_t+J):
                for j in range(self.delta_t, self.delta_t+T):
                    for k, coords in enumerate(self.specific_path):
                        self.input[((i-self.delta_t)*T+(j-self.delta_t))*N:((i-self.delta_t)*T+(j-self.delta_t)+1)*N, k, :] = \
                            x[:, :, i+coords[0]-self.delta_t, j+coords[1]-self.delta_t]


        feat_cur = self.sig(self.input)

        for i in range(J):
            for j in range(T):
                self.res[:, :, i, j] = feat_cur[(i*T+j)*N:(i*T+j+1)*N, :]

        return self.res


class SigModuleParallel_cheng(nn.Module):
    def __init__(self, in_channels, sig_in_channels, out_dimension, sig_depth, \
        win_size=5, use_bottleneck=False, specific_path=None, spatial_ps=False):

        """
            - inputs:
                specific_path: list(tuple). Specify the path. [(y1, x1), (y2, x2), ...]
        """

        super(SigModuleParallel_cheng, self).__init__()

        self.in_channels = in_channels
        self.sig_in_channels = sig_in_channels
        self.out_dimension = out_dimension
        self.sig_depth = sig_depth
        self.win_size = win_size
        self.use_bottleneck = use_bottleneck
        self.specific_path = specific_path
        self.spatial_ps = spatial_ps

        assert self.win_size % 2 == 1

        if self.use_bottleneck:
            self.bottleneck = nn.Conv2d(self.in_channels, self.sig_in_channels, kernel_size=(1, 1))
            self.relu = nn.ReLU()
            self.bn = nn.BatchNorm2d(self.sig_in_channels)


        self.delta_t = (self.win_size - 1) // 2

        if self.specific_path is None:
            self.pad = nn.ReplicationPad2d((self.delta_t, self.delta_t, 0, 0))
        else:
            self.pad = nn.ReplicationPad2d((self.delta_t, self.delta_t, self.delta_t, self.delta_t))
        self.sig = SigNet(self.sig_in_channels, self.out_dimension, self.sig_depth, include_time=False if self.spatial_ps else True)


    def forward(self, x):

        N, C, J, T, L = x.shape

        if self.use_bottleneck:
            x = torch.Tensor.permute(x, (0, 4, 1, 2, 3)).reshape((N * L, C, J, T))
            x = self.bn(self.relu(self.bottleneck(x)))
            x = torch.Tensor.reshape(x, (N, L, x.shape[1], J, T)).permute((0, 2, 3, 4, 1))

        if self.specific_path is None:
            self.input = torch.zeros([J*T*N, self.win_size, self.sig_in_channels], dtype=torch.float, requires_grad=True).to(x.device)
        else:
            self.input = torch.zeros([J*T*N, len(self.specific_path), C], dtype=torch.float, requires_grad=True).to(x.device)

        self.res = torch.zeros([N, self.out_dimension, J, T], dtype=torch.float, requires_grad=True).to(x.device)


        if self.specific_path is None:
            for i in range(J):
                for j in range(T):
                    self.input[(i * T + j) * N:(i * T  + j + 1) * N, :, :] = x[:, :, i, j, :].permute(0, 2, 1)
        else:
            for i in range(self.delta_t, self.delta_t+J):
                for j in range(self.delta_t, self.delta_t+T):
                    for k, coords in enumerate(self.specific_path):
                        self.input[((i-self.delta_t)*T+(j-self.delta_t))*N:((i-self.delta_t)*T+(j-self.delta_t)+1)*N, k, :] = \
                            x[:, :, i+coords[0]-self.delta_t, j+coords[1]-self.delta_t]


        feat_cur = self.sig(self.input)
        for i in range(J):
            for j in range(T):

                self.res[:, :, i, j] = feat_cur[(i*T+j)*N:(i*T+j+1)*N, :]
        return self.res


class SigModuleParallel_cheng_v2(nn.Module):
    def __init__(self, in_channels, sig_in_channels, out_dimension, sig_depth, \
        win_size=5):

        """
            - inputs:
                specific_path: list(tuple). Specify the path. [(y1, x1), (y2, x2), ...]
        """

        super(SigModuleParallel_cheng_v2, self).__init__()

        self.in_channels = in_channels
        self.sig_in_channels = sig_in_channels
        self.out_dimension = out_dimension
        self.sig_depth = sig_depth
        self.win_size = win_size

        assert self.win_size % 2 == 1

        # if self.use_bottleneck:
        self.bottleneck = nn.Conv2d(self.in_channels, self.sig_in_channels, kernel_size=(1, 1))
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(self.sig_in_channels)


        self.delta_t = (self.win_size - 1) // 2

        # if self.specific_path is None:
        self.pad = nn.ReplicationPad2d((self.delta_t, self.delta_t, 0, 0))
        # else:
            # self.pad = nn.ReplicationPad2d((self.delta_t, self.delta_t, self.delta_t, self.delta_t))
        # self.sig = SigNet(self.sig_in_channels, self.out_dimension, self.sig_depth, include_time=False if self.spatial_ps else True)
        self.sig = SigNet(self.sig_in_channels, self.out_dimension, self.sig_depth, include_time=True)


    def forward(self, x):

        N, C, J, T, L = x.shape

        # if self.use_bottleneck:
        x = torch.Tensor.permute(x, (0, 4, 1, 2, 3)).reshape((N * L, C, J, T))
        x = self.bn(self.relu(self.bottleneck(x)))
        # x = torch.Tensor.reshape(x, (N, L, x.shape[1], J, T)).permute((0, 2, 3, 4, 1)) 
        x = torch.Tensor.reshape(x, (N, L, x.shape[1], J, T)).permute((0, 3, 4, 1, 2)).reshape((-1, L, x.shape[1])) # Before the 2nd reshape: B * J * T * L * C
        # pdb.set_trace()
        x = self.sig(x)
        x = x.reshape((N, J, T, C)).permute((0, 3, 1, 2))

        # if self.specific_path is None:
        # self.input = torch.zeros([J*T*N, self.win_size, self.sig_in_channels], dtype=torch.float, requires_grad=True).to(x.device)
        # else:
        #     self.input = torch.zeros([J*T*N, len(self.specific_path), C], dtype=torch.float, requires_grad=True).to(x.device)

        # self.res = torch.zeros([N, self.out_dimension, J, T], dtype=torch.float, requires_grad=True).to(x.device)


        # if self.specific_path is None:
        # for i in range(J):
        #     for j in range(T):
        #         self.input[(i * T + j) * N:(i * T  + j + 1) * N, :, :] = x[:, :, i, j, :].permute(0, 2, 1)
        # else:
        #     for i in range(self.delta_t, self.delta_t+J):
        #         for j in range(self.delta_t, self.delta_t+T):
        #             for k, coords in enumerate(self.specific_path):
        #                 self.input[((i-self.delta_t)*T+(j-self.delta_t))*N:((i-self.delta_t)*T+(j-self.delta_t)+1)*N, k, :] = \
        #                     x[:, :, i+coords[0]-self.delta_t, j+coords[1]-self.delta_t]


        # feat_cur = self.sig(self.input)
        # for i in range(J):
        #     for j in range(T):

        #         self.res[:, :, i, j] = feat_cur[(i*T+j)*N:(i*T+j+1)*N, :]
        return x


class SigModuleParallel_cheng_v2_origin_ps(nn.Module):
    def __init__(self, in_channels, sig_in_channels, out_dimension, sig_depth, \
        win_size=5):

        """
            - inputs:
                specific_path: list(tuple). Specify the path. [(y1, x1), (y2, x2), ...]
        """

        super(SigModuleParallel_cheng_v2_origin_ps, self).__init__()

        self.in_channels = in_channels
        self.sig_in_channels = sig_in_channels
        self.out_dimension = out_dimension
        self.sig_depth = sig_depth
        self.win_size = win_size

        assert self.win_size % 2 == 1

        # if self.use_bottleneck:
        # self.bottleneck = nn.Conv2d(self.in_channels, self.sig_in_channels, kernel_size=(1, 1))
        # self.relu = nn.ReLU()
        # self.bn = nn.BatchNorm2d(self.sig_in_channels)


        self.delta_t = (self.win_size - 1) // 2

        # if self.specific_path is None:
        self.pad = nn.ReplicationPad2d((self.delta_t, self.delta_t, 0, 0))
        # else:
            # self.pad = nn.ReplicationPad2d((self.delta_t, self.delta_t, self.delta_t, self.delta_t))
        # self.sig = SigNet(self.sig_in_channels, self.out_dimension, self.sig_depth, include_time=False if self.spatial_ps else True)
        self.sig = SigNet_origin_ps(self.sig_in_channels, self.out_dimension, self.sig_depth, include_time=True)


    def forward(self, x):

        N, C, J, T, L = x.shape

        # if self.use_bottleneck:
        x = torch.Tensor.permute(x, (0, 4, 1, 2, 3)).reshape((N * L, C, J, T))
        # x = self.bn(self.relu(self.bottleneck(x)))
        # x = torch.Tensor.reshape(x, (N, L, x.shape[1], J, T)).permute((0, 2, 3, 4, 1)) 
        x = torch.Tensor.reshape(x, (N, L, x.shape[1], J, T)).permute((0, 3, 4, 1, 2)).reshape((-1, L, x.shape[1])) # Before the 2nd reshape: B * J * T * L * C
        # pdb.set_trace()
        x = self.sig(x)
        x = x.reshape((N, J, T, 37059)).permute((0, 3, 1, 2))

        # if self.specific_path is None:
        # self.input = torch.zeros([J*T*N, self.win_size, self.sig_in_channels], dtype=torch.float, requires_grad=True).to(x.device)
        # else:
        #     self.input = torch.zeros([J*T*N, len(self.specific_path), C], dtype=torch.float, requires_grad=True).to(x.device)

        # self.res = torch.zeros([N, self.out_dimension, J, T], dtype=torch.float, requires_grad=True).to(x.device)


        # if self.specific_path is None:
        # for i in range(J):
        #     for j in range(T):
        #         self.input[(i * T + j) * N:(i * T  + j + 1) * N, :, :] = x[:, :, i, j, :].permute(0, 2, 1)
        # else:
        #     for i in range(self.delta_t, self.delta_t+J):
        #         for j in range(self.delta_t, self.delta_t+T):
        #             for k, coords in enumerate(self.specific_path):
        #                 self.input[((i-self.delta_t)*T+(j-self.delta_t))*N:((i-self.delta_t)*T+(j-self.delta_t)+1)*N, k, :] = \
        #                     x[:, :, i+coords[0]-self.delta_t, j+coords[1]-self.delta_t]


        # feat_cur = self.sig(self.input)
        # for i in range(J):
        #     for j in range(T):

        #         self.res[:, :, i, j] = feat_cur[(i*T+j)*N:(i*T+j+1)*N, :]
        return x


class STEM(nn.Module):
    def __init__(self, spatial, temporal, conv, C_in, C_out, with_bn=True, bn_before_actfn=False, temporal_length=7, spatial_length=5):
        super(STEM, self).__init__()
        self.spatial = spatial
        self.temporal = temporal
        self.conv = conv

        self.with_bn = with_bn
        self.bn_before_actfn = bn_before_actfn

        self.C_in = C_in
        self.C_out = C_out

        self.spatial_length = spatial_length
        self.spatial_padding = int((spatial_length - 1) // 2)
        self.temporal_length = temporal_length
        self.temporal_padding = int((temporal_length - 1) // 2)

        if self.spatial or self.temporal:
            if self.conv:
                self.fusion = nn.Sequential()
                self.fusion.add_module("conv", nn.Conv2d(int(C_out * 2), C_out, kernel_size=(1, 1), padding=(0, 0)))
                if self.with_bn and self.bn_before_actfn:
                    self.fusion.add_module("bn", nn.BatchNorm2d(C_out))
                self.fusion.add_module("relu", nn.ReLU())
                if self.with_bn and not self.bn_before_actfn:
                    self.fusion.add_module("bn", nn.BatchNorm2d(C_out))

            if self.spatial and self.temporal:
                self.ps_fusion = nn.Sequential()
                self.ps_fusion.add_module("conv", nn.Conv2d(int(C_out * 2), C_out, kernel_size=(1, 1), padding=(0, 0)))
                if self.with_bn and self.bn_before_actfn:
                    self.ps_fusion.add_module("bn", nn.BatchNorm2d(C_out))
                self.ps_fusion.add_module("relu", nn.ReLU())
                if self.with_bn and not self.bn_before_actfn:
                    self.ps_fusion.add_module("bn", nn.BatchNorm2d(C_out))

            if self.spatial:
                self.spatial_sig = SigNet(C_in, C_out, sig_depth=3, include_time=True)
            if self.temporal:
                self.temporal_sig = SigNet(C_in, C_out, sig_depth=3, include_time=True)
        
        if self.conv:
            self.raw = nn.Sequential()
            self.raw.add_module("conv", nn.Conv2d(C_in, C_out, kernel_size=(1, 3), padding=(0, 1)))
            if self.with_bn and self.bn_before_actfn:
                self.raw.add_module("bn", nn.BatchNorm2d(C_out))
            self.raw.add_module("relu", nn.ReLU())
            if self.with_bn and not self.bn_before_actfn:
                self.raw.add_module("bn", nn.BatchNorm2d(C_out))
        
        if self.spatial_length == 3:
            chalearn16_hand_crafted_spatial_path = chalearn16_hand_crafted_spatial_path_3
        elif self.spatial_length == 5:
            chalearn16_hand_crafted_spatial_path = chalearn16_hand_crafted_spatial_path_5
        elif self.spatial_length == 7:
            chalearn16_hand_crafted_spatial_path = chalearn16_hand_crafted_spatial_path_7
        elif self.spatial_length == 9:
            chalearn16_hand_crafted_spatial_path = chalearn16_hand_crafted_spatial_path_9
        else:
            raise RuntimeError

        self.hand_crafted_path = np.concatenate((chalearn16_hand_crafted_spatial_path, chalearn16_hand_crafted_spatial_path + 22), axis=0)

    
    def forward(self, x):
        B, C, J, T = x.shape
        if self.conv:
            x_conv = self.raw(x)
        
        if not self.spatial and not self.temporal:
            return x_conv

        if self.spatial:
            spatial_path = torch.zeros((int(B * J * T), self.spatial_length, C), dtype=x.dtype, device=x.device)
            for i in range(J):
                # spatial_path[B * T * i: B * T * (i + 1), :, :] = torch.index_select(x, dim=2, \
                #     index=torch.from_numpy(chalearn16_hand_crafted_spatial_path[i, :].ravel()).long().to(x.device)).permute((0, 3, 2, 1)).reshape((B * T, 3, C))
                spatial_path[B * T * i: B * T * (i + 1), :, :] = torch.index_select(x, dim=2, \
                    index=torch.from_numpy(self.hand_crafted_path[i, :].ravel()).long().to(x.device)).permute((0, 3, 2, 1)).reshape((B * T, self.spatial_length, C))
            spatial_path = spatial_path.contiguous()
            
            spatial_sig = self.spatial_sig(spatial_path)
            spatial_sig = torch.Tensor.reshape(spatial_sig, (J, B, T, self.C_out)).permute((1, 3, 0, 2))

        if self.temporal:
            temporal_path = torch.zeros((int(B * J * T), self.temporal_length, C), dtype=x.dtype, device=x.device)
            x_ps = torch.nn.functional.pad(x, (self.temporal_padding, self.temporal_padding, 0, 0), 'replicate')
            for i in range(T):
                temporal_path[(B * J) * i:(B * J) * (i + 1), :, :] = x_ps[:, :, :, i:i + self.temporal_length].permute((0, 2, 3, 1)).reshape((B * J, self.temporal_length, C))
            temporal_path = temporal_path.contiguous()
            
            temporal_sig = self.temporal_sig(temporal_path)
            temporal_sig = torch.Tensor.reshape(temporal_sig, (T, B, J, self.C_out)).permute((1, 3, 2, 0))
        
        
        if self.spatial and self.temporal:
            sig = torch.cat((spatial_sig, temporal_sig), dim=1)
            sig = self.ps_fusion(sig)

        elif self.spatial:
            sig = spatial_sig
        elif self.temporal:
            sig = temporal_sig
        if not self.conv:
            return sig

        x_out = torch.cat((x_conv, sig), dim=1)
        x_out = self.fusion(x_out)

        return x_out


class STEM_v2(nn.Module):
    def __init__(self, C_in, C_out, temporal_length=7, spatial_length=5):
        super(STEM_v2, self).__init__()
        # self.spatial = spatial
        # self.temporal = temporal
        # self.conv = conv

        # self.with_bn = with_bn
        # self.bn_before_actfn = bn_before_actfn

        self.C_in = C_in
        self.C_out = C_out

        self.spatial_length = spatial_length
        self.spatial_padding = int((spatial_length - 1) // 2)
        self.temporal_length = temporal_length
        self.temporal_padding = int((temporal_length - 1) // 2)

        self.fusion = nn.Sequential()
        self.fusion.add_module("conv", nn.Conv2d(int(C_out * 2), C_out, kernel_size=(1, 1), padding=(0, 0)))
        self.fusion.add_module("bn", nn.BatchNorm2d(C_out))
        self.fusion.add_module("relu", nn.ReLU())

        self.ps_fusion = nn.Sequential()
        self.ps_fusion.add_module("conv", nn.Conv2d(int(C_out * 2), C_out, kernel_size=(1, 1), padding=(0, 0)))
        self.ps_fusion.add_module("bn", nn.BatchNorm2d(C_out))
        self.ps_fusion.add_module("relu", nn.ReLU())


        self.spatial_sig = SigNet(C_in, C_out, sig_depth=3, include_time=True)
        self.temporal_sig = SigNet(C_in, C_out, sig_depth=3, include_time=True)
        
        self.raw = nn.Sequential()
        self.raw.add_module("conv", nn.Conv2d(C_in, C_out, kernel_size=(1, 3), padding=(0, 1)))
        self.raw.add_module("bn", nn.BatchNorm2d(C_out))
        self.raw.add_module("relu", nn.ReLU())
        
        if self.spatial_length == 3:
            chalearn16_hand_crafted_spatial_path = chalearn16_hand_crafted_spatial_path_3_new
        elif self.spatial_length == 5:
            chalearn16_hand_crafted_spatial_path = chalearn16_hand_crafted_spatial_path_5_new
        elif self.spatial_length == 7:
            chalearn16_hand_crafted_spatial_path = chalearn16_hand_crafted_spatial_path_7_new
        # elif self.spatial_length == 9:
        #     chalearn16_hand_crafted_spatial_path = chalearn16_hand_crafted_spatial_path_9_new
        else:
            raise RuntimeError

        # self.hand_crafted_path = np.concatenate((chalearn16_hand_crafted_spatial_path, chalearn16_hand_crafted_spatial_path + 22), axis=0)
        self.hand_crafted_path = chalearn16_hand_crafted_spatial_path
        self.hand_crafted_path = torch.from_numpy(self.hand_crafted_path).long().unsqueeze(dim=0).repeat((C_in, 1, 1))

        self.unfold = torch.nn.Unfold(kernel_size=(1, self.temporal_length),
                                       padding=(0, 0),
                                       stride=(1, 1))
    
    def forward(self, x):
        B, C, J, T = x.shape
        x_conv = self.raw(x)

        # spatial_path = torch.zeros((int(B * J * T), self.spatial_length, C), dtype=x.dtype, device=x.device)
        # for i in range(J):
        #     # spatial_path[B * T * i: B * T * (i + 1), :, :] = torch.index_select(x, dim=2, \
        #     #     index=torch.from_numpy(chalearn16_hand_crafted_spatial_path[i, :].ravel()).long().to(x.device)).permute((0, 3, 2, 1)).reshape((B * T, 3, C))
        #     spatial_path[B * T * i: B * T * (i + 1), :, :] = torch.index_select(x, dim=2, \
        #         index=torch.from_numpy(self.hand_crafted_path[i, :].ravel()).long().to(x.device)).permute((0, 3, 2, 1)).reshape((B * T, self.spatial_length, C))
        # spatial_path = spatial_path.contiguous()

        spatial_path = torch.gather(torch.unsqueeze(torch.Tensor.permute(x, (0, 1, 3, 2)), dim=3).repeat((1, 1, 1, J, 1)), dim=4, \
        index=torch.unsqueeze(self.hand_crafted_path, dim=0).unsqueeze(dim=2).repeat(B, 1, T, 1, 1).to(x.device)).contiguous()\
            .permute((0, 2, 3, 4, 1)).reshape((-1, self.spatial_length, C)) # Befpre permute: B * C * T * J * 3 
        # index = torch.unsqueeze(self.hand_crafted_path, dim=0).unsqueeze(dim=2).repeat(B, 1, T, 1, 1)
        # print(spatial_path.shape)
        # pdb.set_trace()
        

        spatial_sig = self.spatial_sig(spatial_path)
        spatial_sig = torch.Tensor.reshape(spatial_sig, (B, T, J, self.C_out)).permute((0, 3, 2, 1)) # B * C * J * T
        # pdb.set_trace()

        # temporal_path = torch.zeros((int(B * J * T), self.temporal_length, C), dtype=x.dtype, device=x.device)
        x_ps = torch.nn.functional.pad(x, (self.temporal_padding, self.temporal_padding, 0, 0), 'replicate')
        # for i in range(T):
        #     temporal_path[(B * J) * i:(B * J) * (i + 1), :, :] = x_ps[:, :, :, i:i + self.temporal_length].permute((0, 2, 3, 1)).reshape((B * J, self.temporal_length, C))
        # temporal_path = temporal_path.contiguous()
        temporal_path = self.unfold(x_ps).reshape(B, C, J, self.temporal_length, -1).contiguous()\
            .permute((0, 4, 2, 3, 1)).reshape((-1, self.temporal_length, C)) # Before permute: B * C * J * 9 * T
        # print(temporal_path.shape)
        # pdb.set_trace()
        
        temporal_sig = self.temporal_sig(temporal_path) 
        temporal_sig = torch.Tensor.reshape(temporal_sig, (B, T, J, self.C_out)).permute((0, 3, 2, 1)) # B * C * J * T
        
        
        sig = torch.cat((spatial_sig, temporal_sig), dim=1)
        sig = self.ps_fusion(sig)

        x_out = torch.cat((x_conv, sig), dim=1)
        x_out = self.fusion(x_out)
        # pdb.set_trace()

        return x_out


class STEM_v2_origin_ps(nn.Module):
    def __init__(self, C_in, C_out, temporal_length=7, spatial_length=5):
        super(STEM_v2_origin_ps, self).__init__()
        # self.spatial = spatial
        # self.temporal = temporal
        # self.conv = conv

        # self.with_bn = with_bn
        # self.bn_before_actfn = bn_before_actfn

        self.C_in = C_in
        self.C_out = C_out

        self.spatial_length = spatial_length
        self.spatial_padding = int((spatial_length - 1) // 2)
        self.temporal_length = temporal_length
        self.temporal_padding = int((temporal_length - 1) // 2)

        self.fusion = nn.Sequential()
        self.fusion.add_module("conv", nn.Conv2d(int(C_out * 2), C_out, kernel_size=(1, 1), padding=(0, 0)))
        self.fusion.add_module("bn", nn.BatchNorm2d(C_out))
        self.fusion.add_module("relu", nn.ReLU())

        self.ps_fusion = nn.Sequential()
        self.ps_fusion.add_module("conv", nn.Conv2d(int(84 * 2), C_out, kernel_size=(1, 1), padding=(0, 0)))
        self.ps_fusion.add_module("bn", nn.BatchNorm2d(C_out))
        self.ps_fusion.add_module("relu", nn.ReLU())


        self.spatial_sig = SigNet_origin_ps(C_in, C_out, sig_depth=3, include_time=True)
        self.temporal_sig = SigNet_origin_ps(C_in, C_out, sig_depth=3, include_time=True)
        
        self.raw = nn.Sequential()
        self.raw.add_module("conv", nn.Conv2d(C_in, C_out, kernel_size=(1, 3), padding=(0, 1)))
        self.raw.add_module("bn", nn.BatchNorm2d(C_out))
        self.raw.add_module("relu", nn.ReLU())
        
        # if self.spatial_length == 3:
        #     chalearn16_hand_crafted_spatial_path = chalearn16_hand_crafted_spatial_path_3_new
        # elif self.spatial_length == 5:
        #     chalearn16_hand_crafted_spatial_path = chalearn16_hand_crafted_spatial_path_5_new
        # elif self.spatial_length == 7:
        #     chalearn16_hand_crafted_spatial_path = chalearn16_hand_crafted_spatial_path_7_new
        # # elif self.spatial_length == 9:
        # #     chalearn16_hand_crafted_spatial_path = chalearn16_hand_crafted_spatial_path_9_new
        # else:
        #     raise RuntimeError
        chalearn16_hand_crafted_spatial_path = chalearn13_hand_crafted_spatial_path

        # self.hand_crafted_path = np.concatenate((chalearn16_hand_crafted_spatial_path, chalearn16_hand_crafted_spatial_path + 22), axis=0)
        self.hand_crafted_path = chalearn16_hand_crafted_spatial_path
        self.hand_crafted_path = torch.from_numpy(self.hand_crafted_path).long().unsqueeze(dim=0).repeat((C_in, 1, 1))

        self.unfold = torch.nn.Unfold(kernel_size=(1, self.temporal_length),
                                       padding=(0, 0),
                                       stride=(1, 1))
    
    def forward(self, x):
        B, C, J, T = x.shape
        x_conv = self.raw(x)

        # spatial_path = torch.zeros((int(B * J * T), self.spatial_length, C), dtype=x.dtype, device=x.device)
        # for i in range(J):
        #     # spatial_path[B * T * i: B * T * (i + 1), :, :] = torch.index_select(x, dim=2, \
        #     #     index=torch.from_numpy(chalearn16_hand_crafted_spatial_path[i, :].ravel()).long().to(x.device)).permute((0, 3, 2, 1)).reshape((B * T, 3, C))
        #     spatial_path[B * T * i: B * T * (i + 1), :, :] = torch.index_select(x, dim=2, \
        #         index=torch.from_numpy(self.hand_crafted_path[i, :].ravel()).long().to(x.device)).permute((0, 3, 2, 1)).reshape((B * T, self.spatial_length, C))
        # spatial_path = spatial_path.contiguous()

        spatial_path = torch.gather(torch.unsqueeze(torch.Tensor.permute(x, (0, 1, 3, 2)), dim=3).repeat((1, 1, 1, J, 1)), dim=4, \
        index=torch.unsqueeze(self.hand_crafted_path, dim=0).unsqueeze(dim=2).repeat(B, 1, T, 1, 1).to(x.device)).contiguous()\
            .permute((0, 2, 3, 4, 1)).reshape((-1, self.spatial_length, C)) # Befpre permute: B * C * T * J * 3 
        # index = torch.unsqueeze(self.hand_crafted_path, dim=0).unsqueeze(dim=2).repeat(B, 1, T, 1, 1)
        # print(spatial_path.shape)
        # pdb.set_trace()
        

        spatial_sig = self.spatial_sig(spatial_path)
        spatial_sig = torch.Tensor.reshape(spatial_sig, (B, T, J, 84)).permute((0, 3, 2, 1)) # B * C * J * T
        # pdb.set_trace()

        # temporal_path = torch.zeros((int(B * J * T), self.temporal_length, C), dtype=x.dtype, device=x.device)
        x_ps = torch.nn.functional.pad(x, (self.temporal_padding, self.temporal_padding, 0, 0), 'replicate')
        # for i in range(T):
        #     temporal_path[(B * J) * i:(B * J) * (i + 1), :, :] = x_ps[:, :, :, i:i + self.temporal_length].permute((0, 2, 3, 1)).reshape((B * J, self.temporal_length, C))
        # temporal_path = temporal_path.contiguous()
        temporal_path = self.unfold(x_ps).reshape(B, C, J, self.temporal_length, -1).contiguous()\
            .permute((0, 4, 2, 3, 1)).reshape((-1, self.temporal_length, C)) # Before permute: B * C * J * 9 * T
        # print(temporal_path.shape)
        # pdb.set_trace()
        
        temporal_sig = self.temporal_sig(temporal_path) 
        temporal_sig = torch.Tensor.reshape(temporal_sig, (B, T, J, 84)).permute((0, 3, 2, 1)) # B * C * J * T
        
        
        sig = torch.cat((spatial_sig, temporal_sig), dim=1)
        sig = self.ps_fusion(sig)

        x_out = torch.cat((x_conv, sig), dim=1)
        x_out = self.fusion(x_out)
        # pdb.set_trace()

        return x_out


class L_STEM(nn.Module):
    def __init__(self, C_in, C_out, with_bn=True, bn_before_actfn=False, temporal_length=7, spatial_length=5):
        super(L_STEM, self).__init__()

        self.with_bn = with_bn
        self.bn_before_actfn = bn_before_actfn

        self.C_in = C_in
        self.C_out = C_out

        self.spatial_length = spatial_length
        self.spatial_padding = int((spatial_length - 1) // 2)
        self.temporal_length = temporal_length
        self.temporal_padding = int((temporal_length - 1) // 2)

        self.fusion = nn.Sequential()
        self.fusion.add_module("conv", nn.Conv2d(int(C_out * 2), C_out, kernel_size=(1, 1), padding=(0, 0)))
        if self.with_bn and self.bn_before_actfn:
            self.fusion.add_module("bn", nn.BatchNorm2d(C_out))
        self.fusion.add_module("relu", nn.ReLU())
        if self.with_bn and not self.bn_before_actfn:
            self.fusion.add_module("bn", nn.BatchNorm2d(C_out))

        self.ps_fusion = nn.Sequential()
        self.ps_fusion.add_module("conv", nn.Conv2d(int(C_out * 2), C_out, kernel_size=(1, 1), padding=(0, 0)))
        if self.with_bn and self.bn_before_actfn:
            self.ps_fusion.add_module("bn", nn.BatchNorm2d(C_out))
        self.ps_fusion.add_module("relu", nn.ReLU())
        if self.with_bn and not self.bn_before_actfn:
            self.ps_fusion.add_module("bn", nn.BatchNorm2d(C_out))


        self.offset = ConvOffset2D_nonlocal3(C_in, spatial=True, temporal=False, concat=False, if_self=True, path_length=self.spatial_length)
        self.spatial_sig = SigNet(C_in, C_out, sig_depth=3, include_time=True)
        # self.spatial_sig = SigModuleParallel_cheng(32, 3, 32, 3, win_size=self.path_length, use_bottleneck=True, spatial_ps=False)

        self.temporal_sig = SigNet(C_in, C_out, sig_depth=3, include_time=True)
        
        self.raw = nn.Sequential()
        self.raw.add_module("conv", nn.Conv2d(C_in, C_out, kernel_size=(1, 3), padding=(0, 1)))
        if self.with_bn and self.bn_before_actfn:
            self.raw.add_module("bn", nn.BatchNorm2d(C_out))
        self.raw.add_module("relu", nn.ReLU())
        if self.with_bn and not self.bn_before_actfn:
            self.raw.add_module("bn", nn.BatchNorm2d(C_out))
        
        # if self.spatial_length == 3:
        #     chalearn16_hand_crafted_spatial_path = chalearn16_hand_crafted_spatial_path_3
        # elif self.spatial_length == 5:
        #     chalearn16_hand_crafted_spatial_path = chalearn16_hand_crafted_spatial_path_5
        # elif self.spatial_length == 7:
        #     chalearn16_hand_crafted_spatial_path = chalearn16_hand_crafted_spatial_path_7
        # elif self.spatial_length == 9:
        #     chalearn16_hand_crafted_spatial_path = chalearn16_hand_crafted_spatial_path_9
        # else:
        #     raise RuntimeError

        # self.hand_crafted_path = np.concatenate((chalearn16_hand_crafted_spatial_path, chalearn16_hand_crafted_spatial_path + 22), axis=0)

    
    def forward(self, x):
        B, C, J, T = x.shape

        x_nonlocal, x_in_off, offset = self.offset(x) # B C J T L 
        x_conv = self.raw(x_nonlocal)

        x_in_off = torch.Tensor.permute(x_in_off, (0, 2, 3, 4, 1)).reshape((int(B * J * T), self.spatial_length, self.C_in))
        spatial_sig = self.spatial_sig(x_in_off)
        spatial_sig = torch.Tensor.reshape(spatial_sig, (B, J, T, self.C_out)).permute((0, 3, 1, 2))

        temporal_path = torch.zeros((int(B * J * T), self.temporal_length, C), dtype=x.dtype, device=x.device)
        x_ps = torch.nn.functional.pad(x, (self.temporal_padding, self.temporal_padding, 0, 0), 'replicate')
        for i in range(T):
            temporal_path[(B * J) * i:(B * J) * (i + 1), :, :] = x_ps[:, :, :, i:i + self.temporal_length].permute((0, 2, 3, 1)).reshape((B * J, self.temporal_length, C))
        temporal_path = temporal_path.contiguous()
        
        temporal_sig = self.temporal_sig(temporal_path)
        temporal_sig = torch.Tensor.reshape(temporal_sig, (T, B, J, self.C_out)).permute((1, 3, 2, 0))
        
        sig = torch.cat((spatial_sig, temporal_sig), dim=1)
        sig = self.ps_fusion(sig)

        x_out = torch.cat((x_conv, sig), dim=1)
        x_out = self.fusion(x_out)

        return x_out



