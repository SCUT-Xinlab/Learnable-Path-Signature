'''
Author      : now more
Contact     : lin.honghui@qq.com
LastEditors: Please set LastEditors
Description : 
LastEditTime: 2020-11-29 16:30:50
'''
import torch
import torch.nn as nn
from copy import deepcopy
from src.model.utils import weight_init,input_prepare
from src.model.layers import *
from torch.utils.tensorboard import SummaryWriter
from src.model_MSG3D.msg3d import MSG3D_Model
from src.model_2sAGCN.agcn import AGCN_Model
# from src.model_ShiftGCN.shift_gcn import ShiftGCN_Model
from src.model_DecoupleGCN.decouple_gcn import DecoupleGCN_Model
from src.model_STGCN.st_gcn_aaai18 import ST_GCN_18


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

chalearn13_hand_crafted_decoupled_adjacent_matrix = np.array([
    [
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ],
    [
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    ],
    [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
    ]
])

chalearn13_handcrafted_bone_matrix = np.array([
    [1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, -1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, -1, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, -1, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, -1, 1, 0, 0, 0, 0],
    [0, -1, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, -1, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, -1, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 1]
])

class DEFORM_PSCNN_NONLOCAL_MULTI_4_2s_Toy(nn.Module):
    def __init__(self, with_bn=True, bn_before_actfn=True, joint_num=11, spatial_ps=False,dropout_rate=0.5):
        super(DEFORM_PSCNN_NONLOCAL_MULTI_4_2s_Toy, self).__init__()
        self.C_in = 3
        self.J = joint_num
        self.T = 39
        self.C_out = 48 * joint_num * 9
        self.cls_num = 20

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


        if self.increase_dimension_early:
            self.stem = STEM(spatial=self.spatial, temporal=self.temporal, conv=self.conv, C_in=self.C_in, C_out=32, with_bn=self.with_bn, \
                bn_before_actfn=self.bn_before_actfn)
            
        else:
            self.stem = nn.Sequential()
            self.stem.add_module("conv", nn.Conv2d(self.C_in, 32, kernel_size=(1, 1), padding=(0, 0)))
            if self.with_bn and self.bn_before_actfn:
                self.stem.add_module("bn", nn.BatchNorm2d(32))
            self.stem.add_module("relu", nn.ReLU())
            if self.with_bn and not self.bn_before_actfn:
                self.stem.add_module("bn", nn.BatchNorm2d(32))

        self.offset_1 = ConvOffset2D_nonlocal2(32, spatial=self.nonlocal_spatial, temporal=self.nonlocal_temporal, \
            concat=self.nonlocal_concat, if_self=self.nonlocal_self)
        self.offset_2 = ConvOffset2D_nonlocal2(32, spatial=self.nonlocal_spatial, temporal=self.nonlocal_temporal, \
            concat=self.nonlocal_concat, if_self=self.nonlocal_self)
        self.offset_3 = ConvOffset2D_nonlocal2(32, spatial=self.nonlocal_spatial, temporal=self.nonlocal_temporal, \
            concat=self.nonlocal_concat, if_self=self.nonlocal_self)
        self.offset_4 = ConvOffset2D_nonlocal2(32, spatial=self.nonlocal_spatial, temporal=self.nonlocal_temporal, \
            concat=self.nonlocal_concat, if_self=self.nonlocal_self)

        self.sig1 = SigModuleParallel_cheng(32, 3, 32, 3, win_size=3, use_bottleneck=True, spatial_ps=self.spatial_ps)


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
        self.conv_fused.add_module("conv", nn.Conv2d(32 + 32, 64, kernel_size=(1, 3), padding=(0, 1)))
        if self.with_bn and self.bn_before_actfn:
            self.conv_fused.add_module("bn", nn.BatchNorm2d(64))
        self.conv_fused.add_module("relu", nn.ReLU())
        if self.with_bn and not self.bn_before_actfn:
            self.conv_fused.add_module("bn", nn.BatchNorm2d(64))
            
        # self.pool_fused = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 2))
        self.pool_fused = nn.AdaptiveAvgPool2d(1)
        self.bnneck = nn.BatchNorm2d(64)
        self.bnneck.bias.requires_grad_(False)  # no shift

        # self.fc1 = nn.Linear(self.C_out, 256)
        self.dp1 = nn.Dropout(p=dropout_rate)
        self.fc2 = nn.Linear(64, self.cls_num)

        # self.apply(weight_init)

    def forward(self, x):
        x_in = input_prepare(x)
        # x_in.shape = (256, 3, 11, 39)

        x_in = self.stem(x_in)

        x_in = self.conv3(x_in)
        x_in = self.pool3(x_in)

        B, C, J, T = x_in.shape


        x_nonlocal_1, x_in_off_1, offset_1 = self.offset_1(x_in)
        x_nonlocal_2, x_in_off_2, offset_2 = self.offset_2(x_in)
        x_nonlocal_3, x_in_off_3, offset_3 = self.offset_3(x_in)
        x_nonlocal_4, x_in_off_4, offset_4 = self.offset_4(x_in)

        x_nonlocal = torch.cat([x_nonlocal_1, x_nonlocal_2, x_nonlocal_3, x_nonlocal_4], dim=0)
        x_in_off = torch.cat([x_in_off_1, x_in_off_2, x_in_off_3, x_in_off_4], dim=0)
        offset = torch.cat([offset_1, offset_2, offset_3, offset_4], dim=0)

        x_in_ps = self.sig1(x_in_off)

        x_in_ps = torch.reshape(x_in_ps, (4, B, 32, J, T)).permute((1, 0, 2, 3, 4)).reshape((B, -1, J, T))
        x_in_ps = self.sig_fusion(x_in_ps)

        x_nonlocal = torch.reshape(x_nonlocal, (4, B, 32, J, T)).permute((1, 0, 2, 3, 4)).reshape((B, -1, J, T))
        x_nonlocal = self.nonlocal_fusion(x_nonlocal)

        x_in = x_in + x_nonlocal
        x_in = torch.cat([x_in, x_in_ps], 1)

        x_4 = self.conv_fused(x_in)
        x_4 = self.pool_fused(x_4)
        x_4 = self.bnneck(x_4)
        x_4 = x_4.view(x_4.shape[0], -1)

        # x = self.fc1(x_4)
        x = self.dp1(x_4)
        x = self.fc2(x)
        

        return x

class BASELINE_2sMSG3D(nn.Module):
    def __init__(self, with_bn=True, bn_before_actfn=True, joint_num=11, dropout_rate=0.5):
        super(BASELINE_2sMSG3D, self).__init__()
        self.backbone_1 = MSG3D_Model(num_class=20, num_point=11, num_person=1, num_gcn_scales=3, \
            num_g3d_scales=5, graph=chalearn13_hand_crafted_adjacent_matrix)
        self.backbone_2 = MSG3D_Model(num_class=20, num_point=11, num_person=1, num_gcn_scales=3, \
            num_g3d_scales=5, graph=chalearn13_hand_crafted_adjacent_matrix)

    def forward(self, x):
        x_in = input_prepare(x)
        x_in = torch.Tensor.permute(x_in, (0, 1, 3, 2)).unsqueeze(dim=4)
        # N, C, T, J, 1

        output_1 = torch.softmax(self.backbone_1(x_in), dim=1)

        x_in = torch.Tensor.permute(x_in, (0, 1, 2, 4, 3)).matmul(torch.from_numpy(\
            chalearn13_handcrafted_bone_matrix.astype(np.float32)).to(device=x_in.device))
        x_in = torch.Tensor.permute(x_in, (0, 1, 2, 4, 3))

        output_2 = torch.softmax(self.backbone_2(x_in), dim=1)
        output = output_1 + output_2
        return output


class BASELINE_2sAGCN(nn.Module):
    def __init__(self, with_bn=True, bn_before_actfn=True, joint_num=11, dropout_rate=0.5):
        super(BASELINE_2sAGCN, self).__init__()
        self.backbone_1 = AGCN_Model(num_class=20, num_point=11, num_person=1, \
            graph=chalearn13_hand_crafted_adjacent_matrix, in_channels=3)
        self.backbone_2 = AGCN_Model(num_class=20, num_point=11, num_person=1, \
            graph=chalearn13_hand_crafted_adjacent_matrix, in_channels=3)

    def forward(self, x):
        x_in = input_prepare(x)
        x_in = torch.Tensor.permute(x_in, (0, 1, 3, 2)).unsqueeze(dim=4)
        # N, C, T, J, 1

        output_1 = torch.softmax(self.backbone_1(x_in), dim=1)

        x_in = torch.Tensor.permute(x_in, (0, 1, 2, 4, 3)).matmul(torch.from_numpy(\
            chalearn13_handcrafted_bone_matrix.astype(np.float32)).to(device=x_in.device))
        x_in = torch.Tensor.permute(x_in, (0, 1, 2, 4, 3))

        output_2 = torch.softmax(self.backbone_2(x_in), dim=1)
        output = output_1 + output_2
        return output


# class BASELINE_ShiftGCN(nn.Module):
#     def __init__(self, with_bn=True, bn_before_actfn=True, joint_num=11, dropout_rate=0.5):
#         super(BASELINE_ShiftGCN, self).__init__()
#         self.padding = nn.ReplicationPad2d((1, 0, 0, 0))
#         self.backbone_1 = ShiftGCN_Model(num_class=20, num_point=11, num_person=1, \
#             graph=chalearn13_hand_crafted_decoupled_adjacent_matrix, in_channels=3)


#     def forward(self, x):
#         x_in = input_prepare(x)
#         x_in = self.padding(x_in)
#         x_in = torch.Tensor.permute(x_in, (0, 1, 3, 2)).unsqueeze(dim=4)
#         # N, C, T, J, 1

#         output_1 = torch.softmax(self.backbone_1(x_in), dim=1)

#         return output_1


# class BASELINE_ShiftGCN_NL(nn.Module):
#     def __init__(self, with_bn=True, bn_before_actfn=True, joint_num=11, dropout_rate=0.5):
#         super(BASELINE_ShiftGCN_NL, self).__init__()
#         self.padding = nn.ReplicationPad2d((1, 0, 0, 0))
#         self.backbone_1 = ShiftGCN_Model(num_class=20, num_point=11, num_person=1, \
#             graph=chalearn13_hand_crafted_decoupled_adjacent_matrix, in_channels=64)

#         self.C_in = 3
#         self.J = joint_num
#         self.T = 39
#         self.C_out = 48 * joint_num * 9
#         self.cls_num = 20

#         self.with_bn = with_bn
#         self.bn_before_actfn = bn_before_actfn
#         self.spatial_ps = False

#         self.increase_dimension_early = True
#         self.spatial = True
#         self.temporal = True
#         self.conv = True

#         self.non_local = True
#         self.double = False

#         self.nonlocal_spatial = True
#         self.nonlocal_temporal = False
#         self.nonlocal_concat = False
#         self.nonlocal_self = True

#         if self.increase_dimension_early:
#             self.stem = STEM(spatial=self.spatial, temporal=self.temporal, conv=self.conv, C_in=3, C_out=32, with_bn=self.with_bn, \
#                 bn_before_actfn=self.bn_before_actfn)

#             self.offset_1 = ConvOffset2D_nonlocal2(32, spatial=self.nonlocal_spatial, temporal=self.nonlocal_temporal, \
#                 concat=self.nonlocal_concat, if_self=self.nonlocal_self)
#             self.offset_2 = ConvOffset2D_nonlocal2(32, spatial=self.nonlocal_spatial, temporal=self.nonlocal_temporal, \
#                 concat=self.nonlocal_concat, if_self=self.nonlocal_self)
#             self.offset_3 = ConvOffset2D_nonlocal2(32, spatial=self.nonlocal_spatial, temporal=self.nonlocal_temporal, \
#                 concat=self.nonlocal_concat, if_self=self.nonlocal_self)
#             self.offset_4 = ConvOffset2D_nonlocal2(32, spatial=self.nonlocal_spatial, temporal=self.nonlocal_temporal, \
#                 concat=self.nonlocal_concat, if_self=self.nonlocal_self)
#             self.offset_5 = ConvOffset2D_nonlocal2(32, spatial=self.nonlocal_spatial, temporal=self.nonlocal_temporal, \
#                 concat=self.nonlocal_concat, if_self=self.nonlocal_self)
#             self.offset_6 = ConvOffset2D_nonlocal2(32, spatial=self.nonlocal_spatial, temporal=self.nonlocal_temporal, \
#                 concat=self.nonlocal_concat, if_self=self.nonlocal_self)
#             self.offset_7 = ConvOffset2D_nonlocal2(32, spatial=self.nonlocal_spatial, temporal=self.nonlocal_temporal, \
#                 concat=self.nonlocal_concat, if_self=self.nonlocal_self)
#             self.offset_8 = ConvOffset2D_nonlocal2(32, spatial=self.nonlocal_spatial, temporal=self.nonlocal_temporal, \
#                 concat=self.nonlocal_concat, if_self=self.nonlocal_self)

            
#         else:
#             self.offset_1 = ConvOffset2D_nonlocal2(3, spatial=self.nonlocal_spatial, temporal=self.nonlocal_temporal, \
#                 concat=self.nonlocal_concat, if_self=self.nonlocal_self)
#             self.offset_2 = ConvOffset2D_nonlocal2(3, spatial=self.nonlocal_spatial, temporal=self.nonlocal_temporal, \
#                 concat=self.nonlocal_concat, if_self=self.nonlocal_self)
#             self.offset_3 = ConvOffset2D_nonlocal2(3, spatial=self.nonlocal_spatial, temporal=self.nonlocal_temporal, \
#                 concat=self.nonlocal_concat, if_self=self.nonlocal_self)
#             self.offset_4 = ConvOffset2D_nonlocal2(3, spatial=self.nonlocal_spatial, temporal=self.nonlocal_temporal, \
#                 concat=self.nonlocal_concat, if_self=self.nonlocal_self)
#             self.offset_5 = ConvOffset2D_nonlocal2(3, spatial=self.nonlocal_spatial, temporal=self.nonlocal_temporal, \
#                 concat=self.nonlocal_concat, if_self=self.nonlocal_self)
#             self.offset_6 = ConvOffset2D_nonlocal2(3, spatial=self.nonlocal_spatial, temporal=self.nonlocal_temporal, \
#                 concat=self.nonlocal_concat, if_self=self.nonlocal_self)
#             self.offset_7 = ConvOffset2D_nonlocal2(3, spatial=self.nonlocal_spatial, temporal=self.nonlocal_temporal, \
#                 concat=self.nonlocal_concat, if_self=self.nonlocal_self)
#             self.offset_8 = ConvOffset2D_nonlocal2(3, spatial=self.nonlocal_spatial, temporal=self.nonlocal_temporal, \
#                 concat=self.nonlocal_concat, if_self=self.nonlocal_self)


#         self.sig1 = SigModuleParallel_cheng(32, 3, 32, 3, win_size=3, use_bottleneck=True, spatial_ps=self.spatial_ps)


#         self.nonlocal_fusion = nn.Sequential()
#         self.nonlocal_fusion.add_module("conv", nn.Conv2d(32 * 8, 32, kernel_size=(1, 1), padding=(0, 0)))
#         if self.with_bn and self.bn_before_actfn:
#             self.nonlocal_fusion.add_module("bn", nn.BatchNorm2d(32))
#         self.nonlocal_fusion.add_module("relu", nn.ReLU())
#         if self.with_bn and not self.bn_before_actfn:
#             self.nonlocal_fusion.add_module("bn", nn.BatchNorm2d(32))

#         self.sig_fusion = nn.Sequential()
#         self.sig_fusion.add_module("conv", nn.Conv2d(32 * 8, 32, kernel_size=(1, 1), padding=(0, 0)))
#         if self.with_bn and self.bn_before_actfn:
#             self.sig_fusion.add_module("bn", nn.BatchNorm2d(32))
#         self.sig_fusion.add_module("relu", nn.ReLU())
#         if self.with_bn and not self.bn_before_actfn:
#             self.sig_fusion.add_module("bn", nn.BatchNorm2d(32))

#     def forward(self, x):
#         x_in = input_prepare(x)
#         x_in = self.padding(x_in)

#         B, C, J, T = x_in.shape

#         if self.increase_dimension_early:
#             x_in = self.stem(x_in)

#         x_nonlocal_1, x_in_off_1, offset_1 = self.offset_1(x_in)
#         x_nonlocal_2, x_in_off_2, offset_2 = self.offset_2(x_in)
#         x_nonlocal_3, x_in_off_3, offset_3 = self.offset_3(x_in)
#         x_nonlocal_4, x_in_off_4, offset_4 = self.offset_4(x_in)
#         x_nonlocal_5, x_in_off_5, offset_5 = self.offset_5(x_in)
#         x_nonlocal_6, x_in_off_6, offset_6 = self.offset_6(x_in)
#         x_nonlocal_7, x_in_off_7, offset_7 = self.offset_7(x_in)
#         x_nonlocal_8, x_in_off_8, offset_8 = self.offset_8(x_in)

#         x_nonlocal = torch.cat([x_nonlocal_1, x_nonlocal_2, x_nonlocal_3, x_nonlocal_4, x_nonlocal_5, x_nonlocal_6, x_nonlocal_7, x_nonlocal_8], dim=0)
#         x_in_off = torch.cat([x_in_off_1, x_in_off_2, x_in_off_3, x_in_off_4, x_in_off_5, x_in_off_6, x_in_off_7, x_in_off_8], dim=0)
#         offset = torch.cat([offset_1, offset_2, offset_3, offset_4, offset_5, offset_6, offset_7, offset_8], dim=0)

#         x_in_ps = self.sig1(x_in_off)

#         x_in_ps = torch.reshape(x_in_ps, (8, B, 32, J, T)).permute((1, 0, 2, 3, 4)).reshape((B, -1, J, T))
#         x_in_ps = self.sig_fusion(x_in_ps)

#         x_nonlocal = torch.reshape(x_nonlocal, (8, B, 32, J, T)).permute((1, 0, 2, 3, 4)).reshape((B, -1, J, T))
#         x_nonlocal = self.nonlocal_fusion(x_nonlocal)

#         x_in = x_in + x_nonlocal
#         x_in = torch.cat([x_in, x_in_ps], 1)
#         # print(x_in.shape)

#         x_in = torch.Tensor.permute(x_in, (0, 1, 3, 2)).unsqueeze(dim=4)
#         # N, C, T, J, 1

#         output_1 = torch.softmax(self.backbone_1(x_in), dim=1)

#         return output_1


# class BASELINE_ShiftGCN_WITHOUTNL(nn.Module):
#     def __init__(self, with_bn=True, bn_before_actfn=True, joint_num=11, dropout_rate=0.5):
#         super(BASELINE_ShiftGCN_WITHOUTNL, self).__init__()
#         self.padding = nn.ReplicationPad2d((1, 0, 0, 0))
#         self.backbone_1 = ShiftGCN_Model(num_class=20, num_point=11, num_person=1, \
#             graph=chalearn13_hand_crafted_decoupled_adjacent_matrix, in_channels=32)

#         self.C_in = 3
#         self.J = joint_num
#         self.T = 39
#         self.C_out = 48 * joint_num * 9
#         self.cls_num = 20

#         self.with_bn = with_bn
#         self.bn_before_actfn = bn_before_actfn
#         self.spatial_ps = False

#         self.increase_dimension_early = True
#         self.spatial = True
#         self.temporal = True
#         self.conv = True

#         self.non_local = True
#         self.double = False

#         self.nonlocal_spatial = True
#         self.nonlocal_temporal = False
#         self.nonlocal_concat = False
#         self.nonlocal_self = True

#         if self.increase_dimension_early:
#             self.stem = STEM(spatial=self.spatial, temporal=self.temporal, conv=self.conv, C_in=3, C_out=32, with_bn=self.with_bn, \
#                 bn_before_actfn=self.bn_before_actfn)


#     def forward(self, x):
#         x_in = input_prepare(x)
#         x_in = self.padding(x_in)

#         B, C, J, T = x_in.shape

#         if self.increase_dimension_early:
#             x_in = self.stem(x_in)


#         x_in = torch.Tensor.permute(x_in, (0, 1, 3, 2)).unsqueeze(dim=4)
#         # N, C, T, J, 1

#         output_1 = torch.softmax(self.backbone_1(x_in), dim=1)

#         return output_1


# class BASELINE_ShiftGCN_WITHOUTSTEM(nn.Module):
#     def __init__(self, with_bn=True, bn_before_actfn=True, joint_num=11, dropout_rate=0.5):
#         super(BASELINE_ShiftGCN_WITHOUTSTEM, self).__init__()
#         self.padding = nn.ReplicationPad2d((1, 0, 0, 0))
#         self.backbone_1 = ShiftGCN_Model(num_class=20, num_point=11, num_person=1, \
#             graph=chalearn13_hand_crafted_decoupled_adjacent_matrix, in_channels=64)

#         self.C_in = 3
#         self.J = joint_num
#         self.T = 39
#         self.C_out = 48 * joint_num * 9
#         self.cls_num = 20

#         self.with_bn = with_bn
#         self.bn_before_actfn = bn_before_actfn
#         self.spatial_ps = False

#         self.increase_dimension_early = False
#         self.spatial = True
#         self.temporal = True
#         self.conv = True

#         self.non_local = True
#         self.double = False

#         self.nonlocal_spatial = True
#         self.nonlocal_temporal = False
#         self.nonlocal_concat = False
#         self.nonlocal_self = True

#         if self.increase_dimension_early:
#             self.stem = STEM(spatial=self.spatial, temporal=self.temporal, conv=self.conv, C_in=3, C_out=32, with_bn=self.with_bn, \
#                 bn_before_actfn=self.bn_before_actfn)

#         else:
#             self.stem = nn.Sequential()
#             self.stem.add_module("conv", nn.Conv2d(3, 32, kernel_size=(1, 1), padding=(0, 0)))
#             if self.with_bn and self.bn_before_actfn:
#                 self.stem.add_module("bn", nn.BatchNorm2d(32))
#             self.stem.add_module("relu", nn.ReLU())
#             if self.with_bn and not self.bn_before_actfn:
#                 self.stem.add_module("bn", nn.BatchNorm2d(32))

#         self.offset_1 = ConvOffset2D_nonlocal2(32, spatial=self.nonlocal_spatial, temporal=self.nonlocal_temporal, \
#             concat=self.nonlocal_concat, if_self=self.nonlocal_self)
#         self.offset_2 = ConvOffset2D_nonlocal2(32, spatial=self.nonlocal_spatial, temporal=self.nonlocal_temporal, \
#             concat=self.nonlocal_concat, if_self=self.nonlocal_self)
#         self.offset_3 = ConvOffset2D_nonlocal2(32, spatial=self.nonlocal_spatial, temporal=self.nonlocal_temporal, \
#             concat=self.nonlocal_concat, if_self=self.nonlocal_self)
#         self.offset_4 = ConvOffset2D_nonlocal2(32, spatial=self.nonlocal_spatial, temporal=self.nonlocal_temporal, \
#             concat=self.nonlocal_concat, if_self=self.nonlocal_self)
#         self.offset_5 = ConvOffset2D_nonlocal2(32, spatial=self.nonlocal_spatial, temporal=self.nonlocal_temporal, \
#             concat=self.nonlocal_concat, if_self=self.nonlocal_self)
#         self.offset_6 = ConvOffset2D_nonlocal2(32, spatial=self.nonlocal_spatial, temporal=self.nonlocal_temporal, \
#             concat=self.nonlocal_concat, if_self=self.nonlocal_self)
#         self.offset_7 = ConvOffset2D_nonlocal2(32, spatial=self.nonlocal_spatial, temporal=self.nonlocal_temporal, \
#             concat=self.nonlocal_concat, if_self=self.nonlocal_self)
#         self.offset_8 = ConvOffset2D_nonlocal2(32, spatial=self.nonlocal_spatial, temporal=self.nonlocal_temporal, \
#             concat=self.nonlocal_concat, if_self=self.nonlocal_self)




#         self.sig1 = SigModuleParallel_cheng(32, 3, 32, 3, win_size=3, use_bottleneck=True, spatial_ps=self.spatial_ps)


#         self.nonlocal_fusion = nn.Sequential()
#         self.nonlocal_fusion.add_module("conv", nn.Conv2d(32 * 8, 32, kernel_size=(1, 1), padding=(0, 0)))
#         if self.with_bn and self.bn_before_actfn:
#             self.nonlocal_fusion.add_module("bn", nn.BatchNorm2d(32))
#         self.nonlocal_fusion.add_module("relu", nn.ReLU())
#         if self.with_bn and not self.bn_before_actfn:
#             self.nonlocal_fusion.add_module("bn", nn.BatchNorm2d(32))

#         self.sig_fusion = nn.Sequential()
#         self.sig_fusion.add_module("conv", nn.Conv2d(32 * 8, 32, kernel_size=(1, 1), padding=(0, 0)))
#         if self.with_bn and self.bn_before_actfn:
#             self.sig_fusion.add_module("bn", nn.BatchNorm2d(32))
#         self.sig_fusion.add_module("relu", nn.ReLU())
#         if self.with_bn and not self.bn_before_actfn:
#             self.sig_fusion.add_module("bn", nn.BatchNorm2d(32))

#     def forward(self, x):
#         x_in = input_prepare(x)
#         x_in = self.padding(x_in)

#         B, C, J, T = x_in.shape

#         x_in = self.stem(x_in)

#         x_nonlocal_1, x_in_off_1, offset_1 = self.offset_1(x_in)
#         x_nonlocal_2, x_in_off_2, offset_2 = self.offset_2(x_in)
#         x_nonlocal_3, x_in_off_3, offset_3 = self.offset_3(x_in)
#         x_nonlocal_4, x_in_off_4, offset_4 = self.offset_4(x_in)
#         x_nonlocal_5, x_in_off_5, offset_5 = self.offset_5(x_in)
#         x_nonlocal_6, x_in_off_6, offset_6 = self.offset_6(x_in)
#         x_nonlocal_7, x_in_off_7, offset_7 = self.offset_7(x_in)
#         x_nonlocal_8, x_in_off_8, offset_8 = self.offset_8(x_in)

#         x_nonlocal = torch.cat([x_nonlocal_1, x_nonlocal_2, x_nonlocal_3, x_nonlocal_4, x_nonlocal_5, x_nonlocal_6, x_nonlocal_7, x_nonlocal_8], dim=0)
#         x_in_off = torch.cat([x_in_off_1, x_in_off_2, x_in_off_3, x_in_off_4, x_in_off_5, x_in_off_6, x_in_off_7, x_in_off_8], dim=0)
#         offset = torch.cat([offset_1, offset_2, offset_3, offset_4, offset_5, offset_6, offset_7, offset_8], dim=0)

#         x_in_ps = self.sig1(x_in_off)

#         x_in_ps = torch.reshape(x_in_ps, (8, B, 32, J, T)).permute((1, 0, 2, 3, 4)).reshape((B, -1, J, T))
#         x_in_ps = self.sig_fusion(x_in_ps)

#         x_nonlocal = torch.reshape(x_nonlocal, (8, B, 32, J, T)).permute((1, 0, 2, 3, 4)).reshape((B, -1, J, T))
#         x_nonlocal = self.nonlocal_fusion(x_nonlocal)

#         x_in = x_in + x_nonlocal
#         x_in = torch.cat([x_in, x_in_ps], 1)

#         x_in = torch.Tensor.permute(x_in, (0, 1, 3, 2)).unsqueeze(dim=4)
#         # N, C, T, J, 1

#         output_1 = torch.softmax(self.backbone_1(x_in), dim=1)

#         return output_1


class BASELINE_DecoupleGCN(nn.Module):
    def __init__(self, with_bn=True, bn_before_actfn=True, joint_num=11, dropout_rate=0.5):
        super(BASELINE_DecoupleGCN, self).__init__()
        self.backbone_1 = DecoupleGCN_Model(num_class=20, num_point=11, num_person=1, \
            graph=chalearn13_hand_crafted_decoupled_adjacent_matrix, in_channels=3, groups=8)

    def forward(self, x):
        x_in = input_prepare(x)
        x_in = torch.Tensor.permute(x_in, (0, 1, 3, 2)).unsqueeze(dim=4)
        # N, C, T, J, 1

        output_1 = torch.softmax(self.backbone_1(x_in), dim=1)

        return output_1


class BASELINE_STGCN(nn.Module):
    def __init__(self, with_bn=True, bn_before_actfn=True, joint_num=11, dropout_rate=0.5):
        super(BASELINE_STGCN, self).__init__()
        A = torch.from_numpy(chalearn13_hand_crafted_decoupled_adjacent_matrix.astype(np.float32))
        self.backbone_1 = ST_GCN_18(num_class=20, in_channels=3, graph_cfg=A)

    def forward(self, x):
        x_in = input_prepare(x)
        x_in = torch.Tensor.permute(x_in, (0, 1, 3, 2)).unsqueeze(dim=4)
        # N, C, T, J, 1

        output_1 = torch.softmax(self.backbone_1(x_in), dim=1)

        return output_1


class BASELINE_STGCN(nn.Module):
    def __init__(self, with_bn=True, bn_before_actfn=True, joint_num=11, dropout_rate=0.5):
        super(BASELINE_STGCN, self).__init__()
        A = torch.from_numpy(chalearn13_hand_crafted_decoupled_adjacent_matrix.astype(np.float32))
        self.backbone_1 = ST_GCN_18(num_class=20, in_channels=3, graph_cfg=A)

    def forward(self, x):
        x_in = input_prepare(x)
        x_in = torch.Tensor.permute(x_in, (0, 1, 3, 2)).unsqueeze(dim=4)
        # N, C, T, J, 1

        output_1 = torch.softmax(self.backbone_1(x_in), dim=1)

        return output_1


class BASELINE_STGCN_NL(nn.Module):
    def __init__(self, with_bn=True, bn_before_actfn=True, joint_num=11, dropout_rate=0.5):
        super(BASELINE_STGCN_NL, self).__init__()
        A = torch.from_numpy(chalearn13_hand_crafted_decoupled_adjacent_matrix.astype(np.float32))
        self.backbone_1 = ST_GCN_18(num_class=20, in_channels=64, graph_cfg=A)

        self.C_in = 3
        self.J = joint_num
        self.T = 39
        self.C_out = 48 * joint_num * 9
        self.cls_num = 20

        self.with_bn = with_bn
        self.bn_before_actfn = bn_before_actfn
        self.spatial_ps = False

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

        if self.increase_dimension_early:
            self.stem = STEM(spatial=self.spatial, temporal=self.temporal, conv=self.conv, C_in=3, C_out=32, with_bn=self.with_bn, \
                bn_before_actfn=self.bn_before_actfn)

            self.offset_1 = ConvOffset2D_nonlocal2(32, spatial=self.nonlocal_spatial, temporal=self.nonlocal_temporal, \
                concat=self.nonlocal_concat, if_self=self.nonlocal_self)
            self.offset_2 = ConvOffset2D_nonlocal2(32, spatial=self.nonlocal_spatial, temporal=self.nonlocal_temporal, \
                concat=self.nonlocal_concat, if_self=self.nonlocal_self)
            self.offset_3 = ConvOffset2D_nonlocal2(32, spatial=self.nonlocal_spatial, temporal=self.nonlocal_temporal, \
                concat=self.nonlocal_concat, if_self=self.nonlocal_self)
            self.offset_4 = ConvOffset2D_nonlocal2(32, spatial=self.nonlocal_spatial, temporal=self.nonlocal_temporal, \
                concat=self.nonlocal_concat, if_self=self.nonlocal_self)
            self.offset_5 = ConvOffset2D_nonlocal2(32, spatial=self.nonlocal_spatial, temporal=self.nonlocal_temporal, \
                concat=self.nonlocal_concat, if_self=self.nonlocal_self)
            self.offset_6 = ConvOffset2D_nonlocal2(32, spatial=self.nonlocal_spatial, temporal=self.nonlocal_temporal, \
                concat=self.nonlocal_concat, if_self=self.nonlocal_self)
            self.offset_7 = ConvOffset2D_nonlocal2(32, spatial=self.nonlocal_spatial, temporal=self.nonlocal_temporal, \
                concat=self.nonlocal_concat, if_self=self.nonlocal_self)
            self.offset_8 = ConvOffset2D_nonlocal2(32, spatial=self.nonlocal_spatial, temporal=self.nonlocal_temporal, \
                concat=self.nonlocal_concat, if_self=self.nonlocal_self)

            
        else:
            self.offset_1 = ConvOffset2D_nonlocal2(3, spatial=self.nonlocal_spatial, temporal=self.nonlocal_temporal, \
                concat=self.nonlocal_concat, if_self=self.nonlocal_self)
            self.offset_2 = ConvOffset2D_nonlocal2(3, spatial=self.nonlocal_spatial, temporal=self.nonlocal_temporal, \
                concat=self.nonlocal_concat, if_self=self.nonlocal_self)
            self.offset_3 = ConvOffset2D_nonlocal2(3, spatial=self.nonlocal_spatial, temporal=self.nonlocal_temporal, \
                concat=self.nonlocal_concat, if_self=self.nonlocal_self)
            self.offset_4 = ConvOffset2D_nonlocal2(3, spatial=self.nonlocal_spatial, temporal=self.nonlocal_temporal, \
                concat=self.nonlocal_concat, if_self=self.nonlocal_self)
            self.offset_5 = ConvOffset2D_nonlocal2(3, spatial=self.nonlocal_spatial, temporal=self.nonlocal_temporal, \
                concat=self.nonlocal_concat, if_self=self.nonlocal_self)
            self.offset_6 = ConvOffset2D_nonlocal2(3, spatial=self.nonlocal_spatial, temporal=self.nonlocal_temporal, \
                concat=self.nonlocal_concat, if_self=self.nonlocal_self)
            self.offset_7 = ConvOffset2D_nonlocal2(3, spatial=self.nonlocal_spatial, temporal=self.nonlocal_temporal, \
                concat=self.nonlocal_concat, if_self=self.nonlocal_self)
            self.offset_8 = ConvOffset2D_nonlocal2(3, spatial=self.nonlocal_spatial, temporal=self.nonlocal_temporal, \
                concat=self.nonlocal_concat, if_self=self.nonlocal_self)


        self.sig1 = SigModuleParallel_cheng(32, 3, 32, 3, win_size=3, use_bottleneck=True, spatial_ps=self.spatial_ps)


        self.nonlocal_fusion = nn.Sequential()
        self.nonlocal_fusion.add_module("conv", nn.Conv2d(32 * 8, 32, kernel_size=(1, 1), padding=(0, 0)))
        if self.with_bn and self.bn_before_actfn:
            self.nonlocal_fusion.add_module("bn", nn.BatchNorm2d(32))
        self.nonlocal_fusion.add_module("relu", nn.ReLU())
        if self.with_bn and not self.bn_before_actfn:
            self.nonlocal_fusion.add_module("bn", nn.BatchNorm2d(32))

        self.sig_fusion = nn.Sequential()
        self.sig_fusion.add_module("conv", nn.Conv2d(32 * 8, 32, kernel_size=(1, 1), padding=(0, 0)))
        if self.with_bn and self.bn_before_actfn:
            self.sig_fusion.add_module("bn", nn.BatchNorm2d(32))
        self.sig_fusion.add_module("relu", nn.ReLU())
        if self.with_bn and not self.bn_before_actfn:
            self.sig_fusion.add_module("bn", nn.BatchNorm2d(32))

    def forward(self, x):
        x_in = input_prepare(x)


        B, C, J, T = x_in.shape

        if self.increase_dimension_early:
            x_in = self.stem(x_in)

        x_nonlocal_1, x_in_off_1, offset_1 = self.offset_1(x_in)
        x_nonlocal_2, x_in_off_2, offset_2 = self.offset_2(x_in)
        x_nonlocal_3, x_in_off_3, offset_3 = self.offset_3(x_in)
        x_nonlocal_4, x_in_off_4, offset_4 = self.offset_4(x_in)
        x_nonlocal_5, x_in_off_5, offset_5 = self.offset_5(x_in)
        x_nonlocal_6, x_in_off_6, offset_6 = self.offset_6(x_in)
        x_nonlocal_7, x_in_off_7, offset_7 = self.offset_7(x_in)
        x_nonlocal_8, x_in_off_8, offset_8 = self.offset_8(x_in)

        x_nonlocal = torch.cat([x_nonlocal_1, x_nonlocal_2, x_nonlocal_3, x_nonlocal_4, x_nonlocal_5, x_nonlocal_6, x_nonlocal_7, x_nonlocal_8], dim=0)
        x_in_off = torch.cat([x_in_off_1, x_in_off_2, x_in_off_3, x_in_off_4, x_in_off_5, x_in_off_6, x_in_off_7, x_in_off_8], dim=0)
        offset = torch.cat([offset_1, offset_2, offset_3, offset_4, offset_5, offset_6, offset_7, offset_8], dim=0)

        x_in_ps = self.sig1(x_in_off)

        x_in_ps = torch.reshape(x_in_ps, (8, B, 32, J, T)).permute((1, 0, 2, 3, 4)).reshape((B, -1, J, T))
        x_in_ps = self.sig_fusion(x_in_ps)

        x_nonlocal = torch.reshape(x_nonlocal, (8, B, 32, J, T)).permute((1, 0, 2, 3, 4)).reshape((B, -1, J, T))
        x_nonlocal = self.nonlocal_fusion(x_nonlocal)

        x_in = x_in + x_nonlocal
        x_in = torch.cat([x_in, x_in_ps], 1)


        x_in = torch.Tensor.permute(x_in, (0, 1, 3, 2)).unsqueeze(dim=4)
        # N, C, T, J, 1

        output_1 = torch.softmax(self.backbone_1(x_in), dim=1)

        return output_1


class DEFORM_PSCNN_NONLOCAL_MULTI(nn.Module):
    def __init__(self, with_bn=True, bn_before_actfn=True, joint_num=11, spatial_ps=False,dropout_rate=0.5):
        super(DEFORM_PSCNN_NONLOCAL_MULTI, self).__init__()

        self.C_in = 3
        self.J = joint_num
        self.T = 39
        self.C_out = 48 * joint_num * 9
        self.cls_num = 20

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


        if self.increase_dimension_early:
            self.stem = STEM(spatial=self.spatial, temporal=self.temporal, conv=self.conv, C_in=3, C_out=32, with_bn=self.with_bn, \
                bn_before_actfn=self.bn_before_actfn)

            self.offset_1 = ConvOffset2D_nonlocal2(32, spatial=self.nonlocal_spatial, temporal=self.nonlocal_temporal, \
                concat=self.nonlocal_concat, if_self=self.nonlocal_self)
            self.offset_2 = ConvOffset2D_nonlocal2(32, spatial=self.nonlocal_spatial, temporal=self.nonlocal_temporal, \
                concat=self.nonlocal_concat, if_self=self.nonlocal_self)
            self.offset_3 = ConvOffset2D_nonlocal2(32, spatial=self.nonlocal_spatial, temporal=self.nonlocal_temporal, \
                concat=self.nonlocal_concat, if_self=self.nonlocal_self)
            self.offset_4 = ConvOffset2D_nonlocal2(32, spatial=self.nonlocal_spatial, temporal=self.nonlocal_temporal, \
                concat=self.nonlocal_concat, if_self=self.nonlocal_self)

            
        else:
            self.offset_1 = ConvOffset2D_nonlocal2(3, spatial=self.nonlocal_spatial, temporal=self.nonlocal_temporal, \
                concat=self.nonlocal_concat, if_self=self.nonlocal_self)
            self.offset_2 = ConvOffset2D_nonlocal2(3, spatial=self.nonlocal_spatial, temporal=self.nonlocal_temporal, \
                concat=self.nonlocal_concat, if_self=self.nonlocal_self)
            self.offset_3 = ConvOffset2D_nonlocal2(3, spatial=self.nonlocal_spatial, temporal=self.nonlocal_temporal, \
                concat=self.nonlocal_concat, if_self=self.nonlocal_self)
            self.offset_4 = ConvOffset2D_nonlocal2(3, spatial=self.nonlocal_spatial, temporal=self.nonlocal_temporal, \
                concat=self.nonlocal_concat, if_self=self.nonlocal_self)


        self.sig1 = SigModuleParallel_cheng(32, 3, 32, 3, win_size=3, use_bottleneck=True, spatial_ps=self.spatial_ps)


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
        self.conv3.add_module("conv", nn.Conv2d(32 + 32, 32, kernel_size=(1, 3), padding=(0, 1)))
        if self.with_bn and self.bn_before_actfn:
            self.conv3.add_module("bn", nn.BatchNorm2d(32))
        self.conv3.add_module("relu", nn.ReLU())
        if self.with_bn and not self.bn_before_actfn:
            self.conv3.add_module("bn", nn.BatchNorm2d(32))

        self.pool3 = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 2))


        if self.double:
            self.offset_21 = ConvOffset2D_nonlocal2(32, spatial=self.nonlocal_spatial, temporal=self.nonlocal_temporal, \
                concat=self.nonlocal_concat, if_self=self.nonlocal_self)
            self.offset_22 = ConvOffset2D_nonlocal2(32, spatial=self.nonlocal_spatial, temporal=self.nonlocal_temporal, \
                concat=self.nonlocal_concat, if_self=self.nonlocal_self)
            self.offset_23 = ConvOffset2D_nonlocal2(32, spatial=self.nonlocal_spatial, temporal=self.nonlocal_temporal, \
                concat=self.nonlocal_concat, if_self=self.nonlocal_self)
            self.offset_24 = ConvOffset2D_nonlocal2(32, spatial=self.nonlocal_spatial, temporal=self.nonlocal_temporal, \
                concat=self.nonlocal_concat, if_self=self.nonlocal_self)

            self.sig12 = SigModuleParallel_cheng(32, 3, 32, 3, win_size=3, use_bottleneck=True, spatial_ps=self.spatial_ps)

            self.nonlocal_fusion2 = nn.Sequential()
            self.nonlocal_fusion2.add_module("conv", nn.Conv2d(32 * 4, 32, kernel_size=(1, 1), padding=(0, 0)))
            if self.with_bn and self.bn_before_actfn:
                self.nonlocal_fusion2.add_module("bn", nn.BatchNorm2d(32))
            self.nonlocal_fusion2.add_module("relu", nn.ReLU())
            if self.with_bn and not self.bn_before_actfn:
                self.nonlocal_fusion2.add_module("bn", nn.BatchNorm2d(32))

            self.sig_fusion2 = nn.Sequential()
            self.sig_fusion2.add_module("conv", nn.Conv2d(32 * 4, 32, kernel_size=(1, 1), padding=(0, 0)))
            if self.with_bn and self.bn_before_actfn:
                self.sig_fusion2.add_module("bn", nn.BatchNorm2d(32))
            self.sig_fusion2.add_module("relu", nn.ReLU())
            if self.with_bn and not self.bn_before_actfn:
                self.sig_fusion2.add_module("bn", nn.BatchNorm2d(32))

            self.conv_fused = nn.Sequential()
            self.conv_fused.add_module("conv", nn.Conv2d(32 + 32, 48, kernel_size=(1, 3), padding=(0, 1)))
            if self.with_bn and self.bn_before_actfn:
                self.conv_fused.add_module("bn", nn.BatchNorm2d(48))
            self.conv_fused.add_module("relu", nn.ReLU())
            if self.with_bn and not self.bn_before_actfn:
                self.conv_fused.add_module("bn", nn.BatchNorm2d(48))

            self.pool_fused = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 2))

        else:
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
        B, C, J, T = x_in.shape

        if self.increase_dimension_early:
            x_in = self.stem(x_in)

        x_nonlocal_1, x_in_off_1, offset_1 = self.offset_1(x_in)
        x_nonlocal_2, x_in_off_2, offset_2 = self.offset_2(x_in)
        x_nonlocal_3, x_in_off_3, offset_3 = self.offset_3(x_in)
        x_nonlocal_4, x_in_off_4, offset_4 = self.offset_4(x_in)

        x_nonlocal = torch.cat([x_nonlocal_1, x_nonlocal_2, x_nonlocal_3, x_nonlocal_4], dim=0)
        x_in_off = torch.cat([x_in_off_1, x_in_off_2, x_in_off_3, x_in_off_4], dim=0)
        offset = torch.cat([offset_1, offset_2, offset_3, offset_4], dim=0)

        x_in_ps = self.sig1(x_in_off)

        x_in_ps = torch.reshape(x_in_ps, (4, B, 32, J, T)).permute((1, 0, 2, 3, 4)).reshape((B, -1, J, T))
        x_in_ps = self.sig_fusion(x_in_ps)

        x_nonlocal = torch.reshape(x_nonlocal, (4, B, 32, J, T)).permute((1, 0, 2, 3, 4)).reshape((B, -1, J, T))
        x_nonlocal = self.nonlocal_fusion(x_nonlocal)

        x_in = x_in + x_nonlocal
        x_in = torch.cat([x_in, x_in_ps], 1)

        x_3 = self.conv3(x_in)
        x_3 = self.pool3(x_3)

        if self.double:
            B2, C2, J2, T2 = x_in.shape

            x_nonlocal_21, x_in_off_21, offset_21 = self.offset_21(x_3)
            x_nonlocal_22, x_in_off_22, offset_22 = self.offset_22(x_3)
            x_nonlocal_23, x_in_off_23, offset_23 = self.offset_23(x_3)
            x_nonlocal_24, x_in_off_24, offset_24 = self.offset_24(x_3)

            x_nonlocal2 = torch.cat([x_nonlocal_21, x_nonlocal_22, x_nonlocal_23, x_nonlocal_24], dim=0)
            x_in_off2 = torch.cat([x_in_off_21, x_in_off_22, x_in_off_23, x_in_off_24], dim=0)
            offset2 = torch.cat([offset_21, offset_22, offset_23, offset_24], dim=0)

            x_in_ps2 = self.sig12(x_in_off2)

            x_in_ps2 = torch.reshape(x_in_ps2, (4, B2, 32, J2, T2)).permute((1, 0, 2, 3, 4)).reshape((B2, -1, J2, T2))
            x_in_ps2 = self.sig_fusion2(x_in_ps2)

            x_nonlocal2 = torch.reshape(x_nonlocal2, (4, B2, 32, J2, T2)).permute((1, 0, 2, 3, 4)).reshape((B2, -1, J2, T2))
            x_nonlocal2 = self.nonlocal_fusion2(x_nonlocal2)

            x_3 = x_in2 + x_nonlocal2
            x_3 = torch.cat([x_3, x_in_ps], 1)

        x_4 = self.conv_fused(x_3)
        x_4 = self.pool_fused(x_4)
        x_4 = x_4.view(x_4.shape[0], -1)

        x = self.fc1(x_4)
        x = self.dp1(x)
        x = self.fc2(x)
        
        # offset = torch.Tensor.reshape(offset, (4, B, 32, 11, 39, 3)).float().mean(dim=0).squeeze()
        # return x, offset
        return x


class DEFORM_PSCNN_NONLOCAL(nn.Module):
    def __init__(self, with_bn=True, bn_before_actfn=True, joint_num=11, spatial_ps=False,dropout_rate=0.5):
        super(DEFORM_PSCNN_NONLOCAL, self).__init__()

        self.C_in = 3
        self.J = joint_num
        self.T = 39
        self.C_out = 48 * joint_num * 9
        self.cls_num = 20

        self.with_bn = with_bn
        self.bn_before_actfn = bn_before_actfn
        self.spatial_ps = spatial_ps

        self.increase_dimension_early = True
        self.spatial = True
        self.temporal = True
        self.conv = True

        self.non_local = True
        self.double = True

        self.nonlocal_spatial = True
        self.nonlocal_temporal = False
        self.nonlocal_concat = False
        self.nonlocal_self = True


        if self.increase_dimension_early:
            self.stem = STEM(spatial=self.spatial, temporal=self.temporal, conv=self.conv, C_in=3, C_out=32, with_bn=self.with_bn, \
                bn_before_actfn=self.bn_before_actfn)

            self.offset_1 = ConvOffset2D_nonlocal2(32, spatial=self.nonlocal_spatial, temporal=self.nonlocal_temporal, \
                concat=self.nonlocal_concat, if_self=self.nonlocal_self)
            
        else:
            self.offset_1 = ConvOffset2D_nonlocal2(3, spatial=self.nonlocal_spatial, temporal=self.nonlocal_temporal, \
                concat=self.nonlocal_concat, if_self=self.nonlocal_self)


        self.sig1 = SigModuleParallel_cheng(32, 3, 32, 3, win_size=3, use_bottleneck=True, spatial_ps=self.spatial_ps)

        self.conv3 = nn.Sequential()
        self.conv3.add_module("conv", nn.Conv2d(32 + 32, 32, kernel_size=(1, 3), padding=(0, 1)))
        if self.with_bn and self.bn_before_actfn:
            self.conv3.add_module("bn", nn.BatchNorm2d(32))
        self.conv3.add_module("relu", nn.ReLU())
        if self.with_bn and not self.bn_before_actfn:
            self.conv3.add_module("bn", nn.BatchNorm2d(32))

        self.pool3 = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 2))


        if self.double:
            self.offset_21 = ConvOffset2D_nonlocal2(32, spatial=self.nonlocal_spatial, temporal=self.nonlocal_temporal, \
                concat=self.nonlocal_concat, if_self=self.nonlocal_self)

            self.sig12 = SigModuleParallel_cheng(32, 3, 32, 3, win_size=3, use_bottleneck=True, spatial_ps=self.spatial_ps)

            self.conv_fused = nn.Sequential()
            self.conv_fused.add_module("conv", nn.Conv2d(32 + 32, 48, kernel_size=(1, 3), padding=(0, 1)))
            if self.with_bn and self.bn_before_actfn:
                self.conv_fused.add_module("bn", nn.BatchNorm2d(48))
            self.conv_fused.add_module("relu", nn.ReLU())
            if self.with_bn and not self.bn_before_actfn:
                self.conv_fused.add_module("bn", nn.BatchNorm2d(48))

            self.pool_fused = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 2))

        else:
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
        B, C, J, T = x_in.shape

        if self.increase_dimension_early:
            x_in = self.stem(x_in)

        x_nonlocal_1, x_in_off_1, offset_1 = self.offset_1(x_in)

        x_nonlocal = x_nonlocal_1
        x_in_off = x_in_off_1
        offset = offset_1

        x_in_ps = self.sig1(x_in_off)

        x_in = x_in + x_nonlocal
        x_in = torch.cat([x_in, x_in_ps], 1)

        x_3 = self.conv3(x_in)
        x_3 = self.pool3(x_3)

        if self.double:
            B2, C2, J2, T2 = x_in.shape

            x_nonlocal_21, x_in_off_21, offset_21 = self.offset_21(x_3)

            x_nonlocal2 = x_nonlocal_21
            x_in_off2 = x_in_off_21
            offset2 = offset_21

            x_in_ps2 = self.sig12(x_in_off2)

            x_3 = x_3 + x_nonlocal2
            x_3 = torch.cat([x_3, x_in_ps2], 1)

        x_4 = self.conv_fused(x_3)
        x_4 = self.pool_fused(x_4)
        x_4 = x_4.view(x_4.shape[0], -1)

        x = self.fc1(x_4)
        x = self.dp1(x)
        x = self.fc2(x)
        
        # offset = torch.Tensor.reshape(offset, (4, B, 32, 11, 39, 3)).float().mean(dim=0).squeeze()
        # return x, offset
        return x


class DEFORM_PSCNN_NONLOCAL_ST(nn.Module):

    def __init__(self, with_bn=True, bn_before_actfn=False, joint_num=11, cls_num=20,single=True):

        super(DEFORM_PSCNN_NONLOCAL_ST, self).__init__()

        self.C_in = 3
        self.J = joint_num
        self.T = 39
        self.C_out = 48 * joint_num * 9

        self.cls_num = cls_num

        self.with_bn = with_bn
        self.bn_before_actfn = bn_before_actfn
        self.spatial_ps = True

        self.increase_dimension_early = True
        self.spatial = True
        self.temporal = True
        self.conv = True

        self.non_local = True
        self.double = True
        self.single = single

        self.nonlocal_spatial = True
        self.nonlocal_temporal = False
        self.nonlocal_concat = False
        self.nonlocal_self = True

        self.stem = STEM(spatial=self.spatial, temporal=self.temporal, conv=self.conv, C_in=3, C_out=32, with_bn=self.with_bn, \
            bn_before_actfn=self.bn_before_actfn)

        if not self.non_local:
            self.offset = ConvOffset2D_nonlocal(32, spatial=self.nonlocal_spatial, temporal=self.nonlocal_temporal, \
                concat=self.nonlocal_concat, if_self=self.nonlocal_self)
        else:
            self.offset = ConvOffset2D_nonlocal2(32, spatial=self.nonlocal_spatial, temporal=self.nonlocal_temporal, \
                concat=self.nonlocal_concat, if_self=self.nonlocal_self)
        self.sig1 = SigModuleParallel_cheng(32, 3, 32, 3, win_size=3, use_bottleneck=True, spatial_ps=self.spatial_ps)

        if not self.non_local:
            self.nonlocal_conv = nn.Sequential()
            self.nonlocal_conv.add_module("conv", nn.Conv2d(32 * 3, 32, kernel_size=(1, 1), padding=(0, 0)))
            if self.with_bn and self.bn_before_actfn:
                self.nonlocal_conv.add_module("bn", nn.BatchNorm2d(32))
            self.nonlocal_conv.add_module("relu", nn.ReLU())
            if self.with_bn and not self.bn_before_actfn:
                self.nonlocal_conv.add_module("bn", nn.BatchNorm2d(32))
        else:
            self.non_local = nn.Identity()

        self.conv3 = nn.Sequential()
        self.conv3.add_module("conv", nn.Conv2d(32 + 32, 32, kernel_size=(1, 3), padding=(0, 1)))
        # self.conv3 = nn.Conv2d(32 + 32, 32, kernel_size=(3, 7), padding=(1, 3))
        if self.with_bn and self.bn_before_actfn:
            self.conv3.add_module("bn", nn.BatchNorm2d(32))
        self.conv3.add_module("relu", nn.ReLU())
        if self.with_bn and not self.bn_before_actfn:
            self.conv3.add_module("bn", nn.BatchNorm2d(32))

        self.sig_temporal = SigModuleParallel_cheng(32 + 32, 3, 32, 3, win_size=3, use_bottleneck=True, spatial_ps=self.spatial_ps)
        self.temporal_pad = nn.ReplicationPad2d((1, 1, 0, 0))
        self.temporal_fusion = nn.Sequential()
        self.temporal_fusion.add_module("conv", nn.Conv2d(32 + 32, 32, kernel_size=(1, 1), padding=(0, 0)))
        if self.with_bn and self.bn_before_actfn:
            self.temporal_fusion.add_module("bn", nn.BatchNorm2d(32))
        self.temporal_fusion.add_module("relu", nn.ReLU())
        if self.with_bn and not self.bn_before_actfn:
            self.temporal_fusion.add_module("bn", nn.BatchNorm2d(32))

        self.pool3 = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 2))


        if self.double:
            if not self.non_local:
                self.offset2 = ConvOffset2D_nonlocal(32, spatial=self.nonlocal_spatial, temporal=self.nonlocal_temporal, \
                    concat=self.nonlocal_concat, if_self=self.nonlocal_self)
            else:
                self.offset2 = ConvOffset2D_nonlocal2(32, spatial=self.nonlocal_spatial, temporal=self.nonlocal_temporal, \
                    concat=self.nonlocal_concat, if_self=self.nonlocal_self)
            self.sig12 = SigModuleParallel_cheng(32, 3, 32, 3, win_size=3, use_bottleneck=True, spatial_ps=self.spatial_ps)

            if not self.non_local:
                self.nonlocal_conv2 = nn.Sequential()
                self.nonlocal_conv2.add_module("conv", nn.Conv2d(32 * 3, 32, kernel_size=(1, 1), padding=(0, 0)))
                if self.with_bn and self.bn_before_actfn:
                    self.nonlocal_conv2.add_module("bn", nn.BatchNorm2d(32))
                self.nonlocal_conv2.add_module("relu", nn.ReLU())
                if self.with_bn and not self.bn_before_actfn:
                    self.nonlocal_conv2.add_module("bn", nn.BatchNorm2d(32))

        self.conv_fused = nn.Sequential()
        if self.double:
            self.conv_fused.add_module("conv", nn.Conv2d(32 + 32, 48, kernel_size=(1, 3), padding=(0, 1)))
            self.sig_temporal2 = SigModuleParallel_cheng(32 + 32, 3, 48, 3, win_size=3, use_bottleneck=True, spatial_ps=self.spatial_ps)

        else:
            self.conv_fused.add_module("conv", nn.Conv2d(32, 48, kernel_size=(1, 3), padding=(0, 1)))
            self.sig_temporal2 = SigModuleParallel_cheng(32, 3, 48, 3, win_size=3, use_bottleneck=True, spatial_ps=self.spatial_ps)

        if self.with_bn and self.bn_before_actfn:
            self.conv_fused.add_module("bn", nn.BatchNorm2d(48))
        self.conv_fused.add_module("relu", nn.ReLU())
        if self.with_bn and not self.bn_before_actfn:
            self.conv_fused.add_module("bn", nn.BatchNorm2d(48))

        self.temporal_pad2 = nn.ReplicationPad2d((1, 1, 0, 0))
        self.temporal_fusion2 = nn.Sequential()
        self.temporal_fusion2.add_module('conv', nn.Conv2d(48 + 48, 48, kernel_size=(1, 1), padding=(0, 0)))
        if self.with_bn and self.bn_before_actfn:
            self.temporal_fusion2.add_module('bn', nn.BatchNorm2d(48))
        self.temporal_fusion2.add_module('relu', nn.ReLU())
        if self.with_bn and not self.bn_before_actfn:
            self.temporal_fusion2.add_module('bn', nn.BatchNorm2d(48))
    
        self.pool_fused = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 2))

        self.fc1 = nn.Linear(self.C_out, 256)
        self.dp1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(256, self.cls_num)

        self.apply(weight_init)


    def forward(self, x):
        if self.single:
            x_in = self.stem(input_prepare(x))
        else:
            x_in = x

        B, C, J, T = x_in.shape

        x_nonlocal, x_in_off, offset = self.offset(x_in)

        x_in_ps = self.sig1(x_in_off)

        if not self.non_local:
            x_nonlocal = x_nonlocal.permute((0, 1, 4, 2, 3)).reshape((B, -1, J, T))
            x_nonlocal = self.nonlocal_conv(x_nonlocal)

        x_in = x_in + x_nonlocal
        x_in = torch.cat([x_in, x_in_ps], 1)

        x_3 = self.conv3(x_in)

        x_in = self.temporal_pad(x_in).unsqueeze(dim=4)
        x_in_tps = torch.cat([x_in[:, :, :, :-2, :], x_in[:, :, :, 1:-1, :], x_in[:, :, :, 2:, :]], dim=4)
        temporal_ps = self.sig_temporal(x_in_tps)

        x_3 = torch.cat([temporal_ps, x_3], dim=1)
        x_3 = self.temporal_fusion(x_3)
    
        x_3 = self.pool3(x_3)

        if self.double:
            B2, C2, J2, T2 = x_3.shape
            x_nonlocal2, x_in_off2, offset2 = self.offset2(x_3)
            x_in_ps2 = self.sig12(x_in_off2)
            if not self.non_local:
                x_nonlocal2 = x_nonlocal2.permute((0, 1, 4, 2, 3)).reshape((B2, -1, J2, T2))
                x_nonlocal2 = self.nonlocal_conv2(x_nonlocal2)
            x_3 = x_3 + x_nonlocal2
            x_3 = torch.cat([x_3, x_in_ps2], 1)

        x_4 = self.conv_fused(x_3)

        x_3 = self.temporal_pad(x_3).unsqueeze(dim=4)
        x_in_tps2 = torch.cat([x_3[:, :, :, :-2, :], x_3[:, :, :, 1:-1, :], x_3[:, :, :, 2:, :]], dim=4)

        temporal_ps2 = self.sig_temporal2(x_in_tps2)
        x_4 = torch.cat([x_4, temporal_ps2], dim=1)
        
        x_4 = self.temporal_fusion2(x_4)
        x_4 = self.pool_fused(x_4)

        x_4 = x_4.view(x_4.shape[0], -1)

        x = self.fc1(x_4)
        x = self.dp1(x)
        x = self.fc2(x)

        return x


class CNNG_MSG3D(nn.Module):
    def __init__(self, with_bn=True, bn_before_actfn=False, joint_num=11, cls_num=20,dropout_rate=0.5,single=True):
        super(CNNG_MSG3D, self).__init__()

        self.C_in = 3
        self.J = joint_num
        self.T = 39
        self.C_out = 48 * joint_num * 9

        self.cls_num = cls_num
        self.num_scales = 2
        self.win_size = 3

        self.with_bn = with_bn
        self.bn_before_actfn = bn_before_actfn
        
        self.increase_dimension_early = False
        self.spatial = True
        self.temporal = True
        self.conv = True
        self.single = single

        if self.increase_dimension_early:
            self.stem = STEM(spatial=self.spatial, temporal=self.temporal, conv=self.conv, C_in=3, C_out=32, with_bn=self.with_bn, \
                bn_before_actfn=self.bn_before_actfn)

        self.MS_G3D1 = MS_G3D(A_binary=chalearn13_hand_crafted_adjacent_matrix, in_channels=3, out_channels=32,\
         num_scales=self.num_scales, window_size=self.win_size, window_stride=1, window_dilation=1)

        self.pool1 = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 2))

        self.MS_G3D2 = MS_G3D(A_binary=chalearn13_hand_crafted_adjacent_matrix, in_channels=32, out_channels=48,\
         num_scales=self.num_scales, window_size=self.win_size, window_stride=1, window_dilation=1)        
        self.pool2 = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 2))

        self.fc1 = nn.Linear(self.C_out, 256)
        self.dp1 = nn.Dropout(p=dropout_rate)
        self.fc2 = nn.Linear(256, self.cls_num)

        self.apply(weight_init)


    def forward(self, x):
        if self.single:
            x = input_prepare(x)
            if self.increase_dimension_early:
                x_1 = self.stem(x)
            else:
                x_1 = x
        else:
            x_1 = x

        x_1 = torch.Tensor.permute(x_1, (0, 1, 3, 2))
        x_1 = self.MS_G3D1(x_1)
        x_1 = torch.Tensor.permute(x_1, (0, 1, 3, 2))
        x_1 = self.pool1(x_1)

        x_2 = torch.Tensor.permute(x_1, (0, 1, 3, 2))
        x_2 = self.MS_G3D2(x_2)
        x_2 = torch.Tensor.permute(x_2, (0, 1, 3, 2))
        x_2 = self.pool2(x_2)
        x_2 = x_2.view(x_2.shape[0], -1)


        x = self.fc1(x_2)
        x = self.dp1(x)
        x = self.fc2(x)
        
        return x


class CNNG_GCN(nn.Module):
    def __init__(self, with_bn=True, bn_before_actfn=False, joint_num=11, cls_num=20,dropout_rate=0.5,single=True):
        super(CNNG_GCN, self).__init__()

        self.C_in = 3
        self.J = joint_num
        self.T = 39
        self.C_out = 48 * joint_num * 9

        self.cls_num = cls_num
        self.num_scales = 2

        self.with_bn = with_bn
        self.bn_before_actfn = bn_before_actfn
        
        self.increase_dimension_early = False
        self.spatial = True
        self.temporal = True
        self.conv = True
        self.single = single

        if self.increase_dimension_early:
            self.stem = STEM(spatial=self.spatial, temporal=self.temporal, conv=self.conv, C_in=3, C_out=32, with_bn=self.with_bn, \
                bn_before_actfn=self.bn_before_actfn)

        self.A = torch.from_numpy(chalearn13_hand_crafted_adjacent_matrix)

        # self.MS_G3D1 = MS_G3D(A_binary=chalearn13_hand_crafted_adjacent_matrix, in_channels=32, out_channels=32,\
        #  num_scales=self.num_scales, window_size=3, window_stride=1, window_dilation=1)
        self.MS_G3D1 = GCN(A=self.A, dim_in=3, dim_out=32)

        self.pool1 = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 2))

        # self.MS_G3D2 = MS_G3D(A_binary=chalearn13_hand_crafted_adjacent_matrix, in_channels=32, out_channels=48,\
        #  num_scales=self.num_scales, window_size=3, window_stride=1, window_dilation=1)        
        self.MS_G3D2 = GCN(A=self.A, dim_in=32, dim_out=48)
        self.pool2 = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 2))

        self.fc1 = nn.Linear(self.C_out, 256)
        self.dp1 = nn.Dropout(p=dropout_rate)
        self.fc2 = nn.Linear(256, self.cls_num)

        self.apply(weight_init)


    def forward(self, x):
        if self.single:
            x = input_prepare(x)
            if self.increase_dimension_early:
                x_1 = self.stem(x)
            else:
                x_1 = x
        else:
            x_1 = x

        x_1 = torch.Tensor.permute(x_1, (0, 1, 3, 2))
        x_1 = self.MS_G3D1(x_1)
        x_1 = torch.Tensor.permute(x_1, (0, 1, 3, 2))
        x_1 = self.pool1(x_1)

        x_2 = torch.Tensor.permute(x_1, (0, 1, 3, 2))
        x_2 = self.MS_G3D2(x_2)
        x_2 = torch.Tensor.permute(x_2, (0, 1, 3, 2))
        x_2 = self.pool2(x_2)
        x_2 = x_2.view(x_2.shape[0], -1)


        x = self.fc1(x_2)
        x = self.dp1(x)
        x = self.fc2(x)
        
        return x


class CNNG_GCN_OUTPRODUCT(nn.Module):
    def __init__(self, with_bn=True, bn_before_actfn=False, joint_num=11, cls_num=20,dropout_rate=0.5,single=True):
        super(CNNG_GCN_OUTPRODUCT, self).__init__()

        self.C_in = 3
        self.J = joint_num
        self.T = 39
        self.C_out = 48 * joint_num * 9

        self.cls_num = cls_num
        self.num_scales = 2

        self.with_bn = with_bn
        self.bn_before_actfn = bn_before_actfn
        
        self.increase_dimension_early = True
        self.spatial = True
        self.temporal = True
        self.conv = True
        self.single = single

        if self.increase_dimension_early:
            self.stem = STEM(spatial=self.spatial, temporal=self.temporal, conv=self.conv, C_in=3, C_out=32, with_bn=self.with_bn, \
                bn_before_actfn=self.bn_before_actfn)

        self.A = torch.from_numpy(chalearn13_hand_crafted_adjacent_matrix)

        # self.MS_G3D1 = MS_G3D(A_binary=chalearn13_hand_crafted_adjacent_matrix, in_channels=32, out_channels=32,\
        #  num_scales=self.num_scales, window_size=3, window_stride=1, window_dilation=1)
        self.MS_G3D1 = GCN_OUTPRODUCT(A=self.A, dim_in=32, dim_out=32)

        self.pool1 = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 2))

        # self.MS_G3D2 = MS_G3D(A_binary=chalearn13_hand_crafted_adjacent_matrix, in_channels=32, out_channels=48,\
        #  num_scales=self.num_scales, window_size=3, window_stride=1, window_dilation=1)        
        self.MS_G3D2 = GCN_OUTPRODUCT(A=self.A, dim_in=32, dim_out=48)
        self.pool2 = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 2))

        self.fc1 = nn.Linear(self.C_out, 256)
        self.dp1 = nn.Dropout(p=dropout_rate)
        self.fc2 = nn.Linear(256, self.cls_num)

        self.apply(weight_init)


    def forward(self, x):
        if self.single:
            x = input_prepare(x)
            x_1 = self.stem(x)
        else:
            x_1 = x

        x_1 = torch.Tensor.permute(x_1, (0, 1, 3, 2))
        x_1 = self.MS_G3D1(x_1)
        x_1 = torch.Tensor.permute(x_1, (0, 1, 3, 2))
        x_1 = self.pool1(x_1)

        x_2 = torch.Tensor.permute(x_1, (0, 1, 3, 2))
        x_2 = self.MS_G3D2(x_2)
        x_2 = torch.Tensor.permute(x_2, (0, 1, 3, 2))
        x_2 = self.pool2(x_2)
        x_2 = x_2.view(x_2.shape[0], -1)


        x = self.fc1(x_2)
        x = self.dp1(x)
        x = self.fc2(x)
        
        return x


class CNNG(nn.Module):
    def __init__(self, with_bn=False, bn_before_actfn=True, joint_num=6,dropout_rate=0.5):
        super(CNNG, self).__init__()

        self.C_in = 3
        self.J = joint_num
        self.T = 39
        self.C_out = 48 * joint_num * 9
        self.cls_num = 20

        self.with_bn = with_bn
        self.bn_before_actfn = bn_before_actfn

        self.conv1 = nn.Conv2d(self.C_in, 32, kernel_size=(1, 11), padding=(0, 5))
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 2))

        self.conv2 = nn.Conv2d(32, 48, kernel_size=(1, 3), padding=(0, 1))
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 2))

        if self.with_bn:
            self.bn1 = nn.BatchNorm2d(32)
            self.bn2 = nn.BatchNorm2d(48)
            self.bn3 = nn.BatchNorm2d(48)
            self.bn4 = nn.BatchNorm2d(48)
            self.bn5 = nn.BatchNorm2d(48)
            self.bn6 = nn.BatchNorm2d(48)

        self.fc1 = nn.Linear(self.C_out, 256)
        self.dp1 = nn.Dropout(p=dropout_rate)
        self.fc2 = nn.Linear(256, self.cls_num)
        self.apply(weight_init)

    def forward(self, x):
        x_in = input_prepare(x)
        x_1 = self.conv1(x_in)
        if self.with_bn and self.bn_before_actfn:
            x_1 = self.bn1(x_1)
        x_1 = self.relu1(x_1)
        if self.with_bn and not self.bn_before_actfn:
            x_1 = self.bn1(x_1)
        x_1 = self.pool1(x_1)

        x_2 = self.conv2(x_1)
        if self.with_bn and self.bn_before_actfn:
            x_2 = self.bn2(x_2)
        x_2 = self.relu2(x_2)
        if self.with_bn and not self.bn_before_actfn:
            x_2 = self.bn2(x_2)
        x_2 = self.pool2(x_2)

        x_2 = x_2.view(x_2.shape[0], -1)
        x = self.fc1(x_2)
        x = self.dp1(x)
        x = self.fc2(x)
        return x


class PSCNN_TWOSTREAM(nn.Module):
    def __init__(self, with_bn=True, bn_before_actfn=False, joint_num=11):
        super(PSCNN_TWOSTREAM, self).__init__()

        self.sum = False
        self.increase_dimension_early = True

        self.T = 39
        self.J = 11
        self.C_in = 3

        self.spatial = True
        self.temporal = True
        self.conv = True

        self.with_bn = with_bn
        self.bn_before_actfn = bn_before_actfn
        self.joint_num = joint_num

        if self.sum:
            self.C_out = 20

        else:
            self.C_out = 32

        self.stem = STEM(spatial=self.spatial, temporal=self.temporal, conv=self.conv, C_in=3, C_out=32, with_bn=self.with_bn, \
                bn_before_actfn=self.bn_before_actfn)

        self.model_1 = CNNG_MSG3D(with_bn=self.with_bn, bn_before_actfn=self.bn_before_actfn, joint_num=self.joint_num, cls_num=self.C_out,single=False)
        self.model_2 = DEFORM_PSCNN_NONLOCAL_ST(with_bn=self.with_bn, bn_before_actfn=self.bn_before_actfn, joint_num=11, cls_num=self.C_out,single=False)

        if self.sum:
            self.output = nn.Linear(64, 20)

        model1_post = nn.Sequential()
        model2_post = nn.Sequential()
        if self.with_bn and self.bn_before_actfn:
            model1_post.add_module("bn", nn.BatchNorm1d(32))
            model2_post.add_module("bn", nn.BatchNorm1d(32))     
        model1_post.add_module("relu", nn.ReLU())
        model2_post.add_module("bn", nn.ReLU())
        if self.with_bn and not self.bn_before_actfn:
            model1_post.add_module("bn", nn.BatchNorm1d(32))
            model2_post.add_module("bn", nn.BatchNorm1d(32))  

        self.model_1_post = model1_post
        self.model_2_post = model2_post

    def forward(self, x):
        x_in = self._input_prepare(x)
        
        x_in_ps = self.stem(x_in)
        output_1 = self.model_1(x_in_ps)
        output_2 = self.model_2(x_in_ps)

        if self.sum:
            output = output_1 + output_2
            return output

        output_1 = self.model_1_post(output_1)
        output_2 = self.model_2_post(output_2)

        output = torch.cat([output_1, output_2], dim=1)

        return output

    def _input_prepare(self, x):
        """
            Transform from NC_in to NCJT
        """

        # pdb.set_trace()

        N = x.shape[0]

        x = x.view(N, self.J, self.C_in, self.T)   # NJCT

        x = x.permute(0, 2, 1, 3).contiguous() # NCJT

        # pdb.set_trace()

        return x


class DEFORM_PSCNN_NONLOCAL_MULTI_4_2s(nn.Module):
    def __init__(self, with_bn=True, bn_before_actfn=True, joint_num=44, spatial_ps=False,dropout_rate=0.5):
        super(DEFORM_PSCNN_NONLOCAL_MULTI_4_2s, self).__init__()

        self.C_in = 3
        self.J = joint_num
        self.T = 39
        self.C_out = 48 * joint_num * 9
        self.cls_num = 20

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


        if self.increase_dimension_early:
            self.stem = STEM(spatial=self.spatial, temporal=self.temporal, conv=self.conv, C_in=self.C_in, C_out=32, with_bn=self.with_bn, \
                bn_before_actfn=self.bn_before_actfn)

            
        else:
            self.stem = nn.Sequential()
            self.stem.add_module("conv", nn.Conv2d(self.C_in, 32, kernel_size=(1, 1), padding=(0, 0)))
            if self.with_bn and self.bn_before_actfn:
                self.stem.add_module("bn", nn.BatchNorm2d(32))
            self.stem.add_module("relu", nn.ReLU())
            if self.with_bn and not self.bn_before_actfn:
                self.stem.add_module("bn", nn.BatchNorm2d(32))

        self.offset_1 = ConvOffset2D_nonlocal2(32, spatial=self.nonlocal_spatial, temporal=self.nonlocal_temporal, \
            concat=self.nonlocal_concat, if_self=self.nonlocal_self)
        self.offset_2 = ConvOffset2D_nonlocal2(32, spatial=self.nonlocal_spatial, temporal=self.nonlocal_temporal, \
            concat=self.nonlocal_concat, if_self=self.nonlocal_self)
        self.offset_3 = ConvOffset2D_nonlocal2(32, spatial=self.nonlocal_spatial, temporal=self.nonlocal_temporal, \
            concat=self.nonlocal_concat, if_self=self.nonlocal_self)
        self.offset_4 = ConvOffset2D_nonlocal2(32, spatial=self.nonlocal_spatial, temporal=self.nonlocal_temporal, \
            concat=self.nonlocal_concat, if_self=self.nonlocal_self)

        self.sig1 = SigModuleParallel_cheng(32, 3, 32, 3, win_size=3, use_bottleneck=True, spatial_ps=self.spatial_ps)


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
        
        self.bnneck = nn.BatchNorm2d(48)
        self.bnneck.bias.requires_grad_(False)  # no shift

        self.fc1 = nn.Linear(self.C_out, 256)
        self.dp1 = nn.Dropout(p=dropout_rate)
        self.fc2 = nn.Linear(256, self.cls_num)

        # self.apply(weight_init)





    def forward(self, x):
        x_in = input_prepare(x)
        # x_in.shape = (256, 3, 11, 39)

        x_in = self.stem(x_in)

        x_in = self.conv3(x_in)
        x_in = self.pool3(x_in)

        B, C, J, T = x_in.shape


        x_nonlocal_1, x_in_off_1, offset_1 = self.offset_1(x_in)
        x_nonlocal_2, x_in_off_2, offset_2 = self.offset_2(x_in)
        x_nonlocal_3, x_in_off_3, offset_3 = self.offset_3(x_in)
        x_nonlocal_4, x_in_off_4, offset_4 = self.offset_4(x_in)

        x_nonlocal = torch.cat([x_nonlocal_1, x_nonlocal_2, x_nonlocal_3, x_nonlocal_4], dim=0)
        x_in_off = torch.cat([x_in_off_1, x_in_off_2, x_in_off_3, x_in_off_4], dim=0)
        offset = torch.cat([offset_1, offset_2, offset_3, offset_4], dim=0)

        x_in_ps = self.sig1(x_in_off)

        x_in_ps = torch.reshape(x_in_ps, (4, B, 32, J, T)).permute((1, 0, 2, 3, 4)).reshape((B, -1, J, T))
        x_in_ps = self.sig_fusion(x_in_ps)

        x_nonlocal = torch.reshape(x_nonlocal, (4, B, 32, J, T)).permute((1, 0, 2, 3, 4)).reshape((B, -1, J, T))
        x_nonlocal = self.nonlocal_fusion(x_nonlocal)

        x_in = x_in + x_nonlocal
        x_in = torch.cat([x_in, x_in_ps], 1)

        x_4 = self.conv_fused(x_in)
        x_4 = self.pool_fused(x_4)
        x_4 = self.bnneck(x_4)
        x_4 = x_4.view(x_4.shape[0], -1)

        x = self.fc1(x_4)
        x = self.dp1(x)
        x = self.fc2(x)
        
        # offset = torch.Tensor.reshape(offset, (4, B, 32, 11, 39, 3)).float().mean(dim=0).squeeze()
        # return x, offset
        return x


class DEFORM_PSCNN_NONLOCAL_MULTI_4_2s_origin_ps(nn.Module):
    def __init__(self, with_bn=True, bn_before_actfn=True, joint_num=44, spatial_ps=False,dropout_rate=0.5):
        super(DEFORM_PSCNN_NONLOCAL_MULTI_4_2s_origin_ps, self).__init__()

        self.C_in = 3
        self.J = joint_num
        self.T = 39
        self.C_out = 48 * joint_num * 9
        self.cls_num = 20

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


        if self.increase_dimension_early:
            self.stem = STEM_origin(spatial=self.spatial, temporal=self.temporal, conv=self.conv, C_in=self.C_in, C_out=32, with_bn=self.with_bn, \
                bn_before_actfn=self.bn_before_actfn)

            
        else:
            self.stem = nn.Sequential()
            self.stem.add_module("conv", nn.Conv2d(self.C_in, 32, kernel_size=(1, 1), padding=(0, 0)))
            if self.with_bn and self.bn_before_actfn:
                self.stem.add_module("bn", nn.BatchNorm2d(32))
            self.stem.add_module("relu", nn.ReLU())
            if self.with_bn and not self.bn_before_actfn:
                self.stem.add_module("bn", nn.BatchNorm2d(32))

        self.offset_1 = ConvOffset2D_nonlocal2(32, spatial=self.nonlocal_spatial, temporal=self.nonlocal_temporal, \
            concat=self.nonlocal_concat, if_self=self.nonlocal_self)
        self.offset_2 = ConvOffset2D_nonlocal2(32, spatial=self.nonlocal_spatial, temporal=self.nonlocal_temporal, \
            concat=self.nonlocal_concat, if_self=self.nonlocal_self)
        self.offset_3 = ConvOffset2D_nonlocal2(32, spatial=self.nonlocal_spatial, temporal=self.nonlocal_temporal, \
            concat=self.nonlocal_concat, if_self=self.nonlocal_self)
        self.offset_4 = ConvOffset2D_nonlocal2(32, spatial=self.nonlocal_spatial, temporal=self.nonlocal_temporal, \
            concat=self.nonlocal_concat, if_self=self.nonlocal_self)

        self.sig1 = SigModuleParallel_cheng_origin(32, 32, 32, 3, win_size=3, use_bottleneck=True, spatial_ps=self.spatial_ps)


        self.nonlocal_fusion = nn.Sequential()
        self.nonlocal_fusion.add_module("conv", nn.Conv2d(32 * 4, 32, kernel_size=(1, 1), padding=(0, 0)))
        if self.with_bn and self.bn_before_actfn:
            self.nonlocal_fusion.add_module("bn", nn.BatchNorm2d(32))
        self.nonlocal_fusion.add_module("relu", nn.ReLU())
        if self.with_bn and not self.bn_before_actfn:
            self.nonlocal_fusion.add_module("bn", nn.BatchNorm2d(32))

        self.sig_fusion = nn.Sequential()
        self.sig_fusion.add_module("conv", nn.Conv2d(37059 * 4, 32, kernel_size=(1, 1), padding=(0, 0)))
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
        
        self.bnneck = nn.BatchNorm2d(48)
        self.bnneck.bias.requires_grad_(False)  # no shift

        self.fc1 = nn.Linear(self.C_out, 256)
        self.dp1 = nn.Dropout(p=dropout_rate)
        self.fc2 = nn.Linear(256, self.cls_num)

        # self.apply(weight_init)





    def forward(self, x):
        x_in = input_prepare(x)
        # x_in.shape = (256, 3, 11, 39)

        x_in = self.stem(x_in)

        x_in = self.conv3(x_in)
        x_in = self.pool3(x_in)

        B, C, J, T = x_in.shape


        x_nonlocal_1, x_in_off_1, offset_1 = self.offset_1(x_in)
        x_nonlocal_2, x_in_off_2, offset_2 = self.offset_2(x_in)
        x_nonlocal_3, x_in_off_3, offset_3 = self.offset_3(x_in)
        x_nonlocal_4, x_in_off_4, offset_4 = self.offset_4(x_in)

        x_nonlocal = torch.cat([x_nonlocal_1, x_nonlocal_2, x_nonlocal_3, x_nonlocal_4], dim=0)
        x_in_off = torch.cat([x_in_off_1, x_in_off_2, x_in_off_3, x_in_off_4], dim=0)
        offset = torch.cat([offset_1, offset_2, offset_3, offset_4], dim=0)

        x_in_ps = self.sig1(x_in_off)

        x_in_ps = torch.reshape(x_in_ps, (4, B, 37059, J, T)).permute((1, 0, 2, 3, 4)).reshape((B, -1, J, T))
        x_in_ps = self.sig_fusion(x_in_ps)

        x_nonlocal = torch.reshape(x_nonlocal, (4, B, 32, J, T)).permute((1, 0, 2, 3, 4)).reshape((B, -1, J, T))
        x_nonlocal = self.nonlocal_fusion(x_nonlocal)

        x_in = x_in + x_nonlocal
        x_in = torch.cat([x_in, x_in_ps], 1)

        x_4 = self.conv_fused(x_in)
        x_4 = self.pool_fused(x_4)
        x_4 = self.bnneck(x_4)
        x_4 = x_4.view(x_4.shape[0], -1)

        x = self.fc1(x_4)
        x = self.dp1(x)
        x = self.fc2(x)
        
        # offset = torch.Tensor.reshape(offset, (4, B, 32, 11, 39, 3)).float().mean(dim=0).squeeze()
        # return x, offset
        return x


class DEFORM_PSCNN_NONLOCAL_MULTI_4_2s_path_length(nn.Module):
    def __init__(self, with_bn=True, bn_before_actfn=True, joint_num=44, spatial_ps=False,dropout_rate=0.5, path_length=3):
        super(DEFORM_PSCNN_NONLOCAL_MULTI_4_2s_path_length, self).__init__()

        self.C_in = 3
        self.J = joint_num
        self.T = 39
        self.C_out = 48 * joint_num * 9
        self.cls_num = 20

        self.path_length = path_length

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


        if self.increase_dimension_early:
            self.stem = STEM(spatial=self.spatial, temporal=self.temporal, conv=self.conv, C_in=self.C_in, C_out=32, with_bn=self.with_bn, \
                bn_before_actfn=self.bn_before_actfn)

            
        else:
            self.stem = nn.Sequential()
            self.stem.add_module("conv", nn.Conv2d(self.C_in, 32, kernel_size=(1, 1), padding=(0, 0)))
            if self.with_bn and self.bn_before_actfn:
                self.stem.add_module("bn", nn.BatchNorm2d(32))
            self.stem.add_module("relu", nn.ReLU())
            if self.with_bn and not self.bn_before_actfn:
                self.stem.add_module("bn", nn.BatchNorm2d(32))

        self.offset_1 = ConvOffset2D_nonlocal2_path_length(32, spatial=self.nonlocal_spatial, temporal=self.nonlocal_temporal, \
            concat=self.nonlocal_concat, if_self=self.nonlocal_self, path_length=self.path_length)
        self.offset_2 = ConvOffset2D_nonlocal2_path_length(32, spatial=self.nonlocal_spatial, temporal=self.nonlocal_temporal, \
            concat=self.nonlocal_concat, if_self=self.nonlocal_self, path_length=self.path_length)
        self.offset_3 = ConvOffset2D_nonlocal2_path_length(32, spatial=self.nonlocal_spatial, temporal=self.nonlocal_temporal, \
            concat=self.nonlocal_concat, if_self=self.nonlocal_self, path_length=self.path_length)
        self.offset_4 = ConvOffset2D_nonlocal2_path_length(32, spatial=self.nonlocal_spatial, temporal=self.nonlocal_temporal, \
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
        
        self.bnneck = nn.BatchNorm2d(48)
        self.bnneck.bias.requires_grad_(False)  # no shift

        self.fc1 = nn.Linear(self.C_out, 256)
        self.dp1 = nn.Dropout(p=dropout_rate)
        self.fc2 = nn.Linear(256, self.cls_num)

        # self.apply(weight_init)





    def forward(self, x):
        x_in = input_prepare(x)
        # x_in.shape = (256, 3, 11, 39)

        x_in = self.stem(x_in)

        x_in = self.conv3(x_in)
        x_in = self.pool3(x_in)

        B, C, J, T = x_in.shape


        x_nonlocal_1, x_in_off_1, offset_1 = self.offset_1(x_in)
        x_nonlocal_2, x_in_off_2, offset_2 = self.offset_2(x_in)
        x_nonlocal_3, x_in_off_3, offset_3 = self.offset_3(x_in)
        x_nonlocal_4, x_in_off_4, offset_4 = self.offset_4(x_in)

        x_nonlocal = torch.cat([x_nonlocal_1, x_nonlocal_2, x_nonlocal_3, x_nonlocal_4], dim=0)
        x_in_off = torch.cat([x_in_off_1, x_in_off_2, x_in_off_3, x_in_off_4], dim=0)
        offset = torch.cat([offset_1, offset_2, offset_3, offset_4], dim=0)

        x_in_ps = self.sig1(x_in_off)

        x_in_ps = torch.reshape(x_in_ps, (4, B, 32, J, T)).permute((1, 0, 2, 3, 4)).reshape((B, -1, J, T))
        x_in_ps = self.sig_fusion(x_in_ps)

        x_nonlocal = torch.reshape(x_nonlocal, (4, B, 32, J, T)).permute((1, 0, 2, 3, 4)).reshape((B, -1, J, T))
        x_nonlocal = self.nonlocal_fusion(x_nonlocal)

        x_in = x_in + x_nonlocal
        x_in = torch.cat([x_in, x_in_ps], 1)

        x_4 = self.conv_fused(x_in)
        x_4 = self.pool_fused(x_4)
        x_4 = self.bnneck(x_4)
        x_4 = x_4.view(x_4.shape[0], -1)

        x = self.fc1(x_4)
        x = self.dp1(x)
        x = self.fc2(x)
        
        # offset = torch.Tensor.reshape(offset, (4, B, 32, 11, 39, 3)).float().mean(dim=0).squeeze()
        # return x, offset
        return x


class DEFORM_PSCNN_WITHOUTSTEM_MULTI_4_2s(nn.Module):
    def __init__(self, with_bn=True, bn_before_actfn=True, joint_num=44, spatial_ps=False,dropout_rate=0.5):
        super(DEFORM_PSCNN_WITHOUTSTEM_MULTI_4_2s, self).__init__()

        self.C_in = 3
        self.J = joint_num
        self.T = 39
        self.C_out = 48 * joint_num * 9
        self.cls_num = 20

        self.with_bn = with_bn
        self.bn_before_actfn = bn_before_actfn
        self.spatial_ps = spatial_ps

        self.increase_dimension_early = False
        self.spatial = True
        self.temporal = True
        self.conv = True

        self.non_local = True
        self.double = False

        self.nonlocal_spatial = True
        self.nonlocal_temporal = False
        self.nonlocal_concat = False
        self.nonlocal_self = True


        if self.increase_dimension_early:
            self.stem = STEM(spatial=self.spatial, temporal=self.temporal, conv=self.conv, C_in=self.C_in, C_out=32, with_bn=self.with_bn, \
                bn_before_actfn=self.bn_before_actfn)

            
        else:
            self.stem = nn.Sequential()
            self.stem.add_module("conv", nn.Conv2d(self.C_in, 32, kernel_size=(1, 1), padding=(0, 0)))
            if self.with_bn and self.bn_before_actfn:
                self.stem.add_module("bn", nn.BatchNorm2d(32))
            self.stem.add_module("relu", nn.ReLU())
            if self.with_bn and not self.bn_before_actfn:
                self.stem.add_module("bn", nn.BatchNorm2d(32))

        self.offset_1 = ConvOffset2D_nonlocal2(32, spatial=self.nonlocal_spatial, temporal=self.nonlocal_temporal, \
            concat=self.nonlocal_concat, if_self=self.nonlocal_self)
        self.offset_2 = ConvOffset2D_nonlocal2(32, spatial=self.nonlocal_spatial, temporal=self.nonlocal_temporal, \
            concat=self.nonlocal_concat, if_self=self.nonlocal_self)
        self.offset_3 = ConvOffset2D_nonlocal2(32, spatial=self.nonlocal_spatial, temporal=self.nonlocal_temporal, \
            concat=self.nonlocal_concat, if_self=self.nonlocal_self)
        self.offset_4 = ConvOffset2D_nonlocal2(32, spatial=self.nonlocal_spatial, temporal=self.nonlocal_temporal, \
            concat=self.nonlocal_concat, if_self=self.nonlocal_self)

        self.sig1 = SigModuleParallel_cheng(32, 3, 32, 3, win_size=3, use_bottleneck=True, spatial_ps=self.spatial_ps)


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

        B, C, J, T = x_in.shape


        x_nonlocal_1, x_in_off_1, offset_1 = self.offset_1(x_in)
        x_nonlocal_2, x_in_off_2, offset_2 = self.offset_2(x_in)
        x_nonlocal_3, x_in_off_3, offset_3 = self.offset_3(x_in)
        x_nonlocal_4, x_in_off_4, offset_4 = self.offset_4(x_in)

        x_nonlocal = torch.cat([x_nonlocal_1, x_nonlocal_2, x_nonlocal_3, x_nonlocal_4], dim=0)
        x_in_off = torch.cat([x_in_off_1, x_in_off_2, x_in_off_3, x_in_off_4], dim=0)
        offset = torch.cat([offset_1, offset_2, offset_3, offset_4], dim=0)

        x_in_ps = self.sig1(x_in_off)

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


class DEFORM_PSCNN_NONLOCAL_MULTI_4_3s(nn.Module):
    def __init__(self, with_bn=True, bn_before_actfn=True, joint_num=44, spatial_ps=False,dropout_rate=0.5):
        super(DEFORM_PSCNN_NONLOCAL_MULTI_4_3s, self).__init__()

        self.C_in = 3
        self.J = joint_num
        self.T = 39
        self.C_out = 48 * joint_num * 9 * 2
        self.cls_num = 20

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


        if self.increase_dimension_early:
            self.stem = STEM(spatial=self.spatial, temporal=self.temporal, conv=self.conv, C_in=self.C_in, C_out=32, with_bn=self.with_bn, \
                bn_before_actfn=self.bn_before_actfn)

            
        else:
            self.stem = nn.Sequential()
            self.stem.add_module("conv", nn.Conv2d(self.C_in, 32, kernel_size=(1, 1), padding=(0, 0)))
            if self.with_bn and self.bn_before_actfn:
                self.stem.add_module("bn", nn.BatchNorm2d(32))
            self.stem.add_module("relu", nn.ReLU())
            if self.with_bn and not self.bn_before_actfn:
                self.stem.add_module("bn", nn.BatchNorm2d(32))

        self.offset_1 = ConvOffset2D_nonlocal2(48, spatial=self.nonlocal_spatial, temporal=self.nonlocal_temporal, \
            concat=self.nonlocal_concat, if_self=self.nonlocal_self)
        self.offset_2 = ConvOffset2D_nonlocal2(48, spatial=self.nonlocal_spatial, temporal=self.nonlocal_temporal, \
            concat=self.nonlocal_concat, if_self=self.nonlocal_self)
        self.offset_3 = ConvOffset2D_nonlocal2(48, spatial=self.nonlocal_spatial, temporal=self.nonlocal_temporal, \
            concat=self.nonlocal_concat, if_self=self.nonlocal_self)
        self.offset_4 = ConvOffset2D_nonlocal2(48, spatial=self.nonlocal_spatial, temporal=self.nonlocal_temporal, \
            concat=self.nonlocal_concat, if_self=self.nonlocal_self)

        self.sig1 = SigModuleParallel_cheng(48, 3, 48, 3, win_size=3, use_bottleneck=True, spatial_ps=self.spatial_ps)


        self.nonlocal_fusion = nn.Sequential()
        self.nonlocal_fusion.add_module("conv", nn.Conv2d(48 * 4, 48, kernel_size=(1, 1), padding=(0, 0)))
        if self.with_bn and self.bn_before_actfn:
            self.nonlocal_fusion.add_module("bn", nn.BatchNorm2d(48))
        self.nonlocal_fusion.add_module("relu", nn.ReLU())
        if self.with_bn and not self.bn_before_actfn:
            self.nonlocal_fusion.add_module("bn", nn.BatchNorm2d(48))

        self.sig_fusion = nn.Sequential()
        self.sig_fusion.add_module("conv", nn.Conv2d(48 * 4, 48, kernel_size=(1, 1), padding=(0, 0)))
        if self.with_bn and self.bn_before_actfn:
            self.sig_fusion.add_module("bn", nn.BatchNorm2d(48))
        self.sig_fusion.add_module("relu", nn.ReLU())
        if self.with_bn and not self.bn_before_actfn:
            self.sig_fusion.add_module("bn", nn.BatchNorm2d(48))

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
        x_in = self.conv_fused(x_in)
        x_in = self.pool_fused(x_in)

        B, C, J, T = x_in.shape


        x_nonlocal_1, x_in_off_1, offset_1 = self.offset_1(x_in)
        x_nonlocal_2, x_in_off_2, offset_2 = self.offset_2(x_in)
        x_nonlocal_3, x_in_off_3, offset_3 = self.offset_3(x_in)
        x_nonlocal_4, x_in_off_4, offset_4 = self.offset_4(x_in)

        x_nonlocal = torch.cat([x_nonlocal_1, x_nonlocal_2, x_nonlocal_3, x_nonlocal_4], dim=0)
        x_in_off = torch.cat([x_in_off_1, x_in_off_2, x_in_off_3, x_in_off_4], dim=0)
        offset = torch.cat([offset_1, offset_2, offset_3, offset_4], dim=0)

        x_in_ps = self.sig1(x_in_off)

        x_in_ps = torch.reshape(x_in_ps, (4, B, C, J, T)).permute((1, 0, 2, 3, 4)).reshape((B, -1, J, T))
        x_in_ps = self.sig_fusion(x_in_ps)

        x_nonlocal = torch.reshape(x_nonlocal, (4, B, C, J, T)).permute((1, 0, 2, 3, 4)).reshape((B, -1, J, T))
        x_nonlocal = self.nonlocal_fusion(x_nonlocal)

        x_in = x_in + x_nonlocal
        x_in = torch.cat([x_in, x_in_ps], 1)

        x_4 = x_in.view(x_in.shape[0], -1)

        x = self.fc1(x_4)
        x = self.dp1(x)
        x = self.fc2(x)
        
        # offset = torch.Tensor.reshape(offset, (4, B, 32, 11, 39, 3)).float().mean(dim=0).squeeze()
        # return x, offset
        return x


class DEFORM_PSCNN_WITHOUTSTEM_MULTI_4_3s(nn.Module):
    def __init__(self, with_bn=True, bn_before_actfn=True, joint_num=44, spatial_ps=False,dropout_rate=0.5):
        super(DEFORM_PSCNN_WITHOUTSTEM_MULTI_4_3s, self).__init__()

        self.C_in = 3
        self.J = joint_num
        self.T = 39
        self.C_out = 48 * joint_num * 9 * 2
        self.cls_num = 20

        self.with_bn = with_bn
        self.bn_before_actfn = bn_before_actfn
        self.spatial_ps = spatial_ps

        self.increase_dimension_early = False
        self.spatial = True
        self.temporal = True
        self.conv = True

        self.non_local = True
        self.double = False

        self.nonlocal_spatial = True
        self.nonlocal_temporal = False
        self.nonlocal_concat = False
        self.nonlocal_self = True


        if self.increase_dimension_early:
            self.stem = STEM(spatial=self.spatial, temporal=self.temporal, conv=self.conv, C_in=self.C_in, C_out=32, with_bn=self.with_bn, \
                bn_before_actfn=self.bn_before_actfn)

            
        else:
            self.stem = nn.Sequential()
            self.stem.add_module("conv", nn.Conv2d(self.C_in, 32, kernel_size=(1, 1), padding=(0, 0)))
            if self.with_bn and self.bn_before_actfn:
                self.stem.add_module("bn", nn.BatchNorm2d(32))
            self.stem.add_module("relu", nn.ReLU())
            if self.with_bn and not self.bn_before_actfn:
                self.stem.add_module("bn", nn.BatchNorm2d(32))

        self.offset_1 = ConvOffset2D_nonlocal2(48, spatial=self.nonlocal_spatial, temporal=self.nonlocal_temporal, \
            concat=self.nonlocal_concat, if_self=self.nonlocal_self)
        self.offset_2 = ConvOffset2D_nonlocal2(48, spatial=self.nonlocal_spatial, temporal=self.nonlocal_temporal, \
            concat=self.nonlocal_concat, if_self=self.nonlocal_self)
        self.offset_3 = ConvOffset2D_nonlocal2(48, spatial=self.nonlocal_spatial, temporal=self.nonlocal_temporal, \
            concat=self.nonlocal_concat, if_self=self.nonlocal_self)
        self.offset_4 = ConvOffset2D_nonlocal2(48, spatial=self.nonlocal_spatial, temporal=self.nonlocal_temporal, \
            concat=self.nonlocal_concat, if_self=self.nonlocal_self)

        self.sig1 = SigModuleParallel_cheng(48, 3, 48, 3, win_size=3, use_bottleneck=True, spatial_ps=self.spatial_ps)


        self.nonlocal_fusion = nn.Sequential()
        self.nonlocal_fusion.add_module("conv", nn.Conv2d(48 * 4, 48, kernel_size=(1, 1), padding=(0, 0)))
        if self.with_bn and self.bn_before_actfn:
            self.nonlocal_fusion.add_module("bn", nn.BatchNorm2d(48))
        self.nonlocal_fusion.add_module("relu", nn.ReLU())
        if self.with_bn and not self.bn_before_actfn:
            self.nonlocal_fusion.add_module("bn", nn.BatchNorm2d(48))

        self.sig_fusion = nn.Sequential()
        self.sig_fusion.add_module("conv", nn.Conv2d(48 * 4, 48, kernel_size=(1, 1), padding=(0, 0)))
        if self.with_bn and self.bn_before_actfn:
            self.sig_fusion.add_module("bn", nn.BatchNorm2d(48))
        self.sig_fusion.add_module("relu", nn.ReLU())
        if self.with_bn and not self.bn_before_actfn:
            self.sig_fusion.add_module("bn", nn.BatchNorm2d(48))

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
        x_in = self.conv_fused(x_in)
        x_in = self.pool_fused(x_in)

        B, C, J, T = x_in.shape


        x_nonlocal_1, x_in_off_1, offset_1 = self.offset_1(x_in)
        x_nonlocal_2, x_in_off_2, offset_2 = self.offset_2(x_in)
        x_nonlocal_3, x_in_off_3, offset_3 = self.offset_3(x_in)
        x_nonlocal_4, x_in_off_4, offset_4 = self.offset_4(x_in)

        x_nonlocal = torch.cat([x_nonlocal_1, x_nonlocal_2, x_nonlocal_3, x_nonlocal_4], dim=0)
        x_in_off = torch.cat([x_in_off_1, x_in_off_2, x_in_off_3, x_in_off_4], dim=0)
        offset = torch.cat([offset_1, offset_2, offset_3, offset_4], dim=0)

        x_in_ps = self.sig1(x_in_off)

        x_in_ps = torch.reshape(x_in_ps, (4, B, C, J, T)).permute((1, 0, 2, 3, 4)).reshape((B, -1, J, T))
        x_in_ps = self.sig_fusion(x_in_ps)

        x_nonlocal = torch.reshape(x_nonlocal, (4, B, C, J, T)).permute((1, 0, 2, 3, 4)).reshape((B, -1, J, T))
        x_nonlocal = self.nonlocal_fusion(x_nonlocal)

        x_in = x_in + x_nonlocal
        x_in = torch.cat([x_in, x_in_ps], 1)

        x_4 = x_in.view(x_in.shape[0], -1)

        x = self.fc1(x_4)
        x = self.dp1(x)
        x = self.fc2(x)
        
        # offset = torch.Tensor.reshape(offset, (4, B, 32, 11, 39, 3)).float().mean(dim=0).squeeze()
        # return x, offset
        return x


class DEFORM_PSCNN_NONLOCAL_MULTI_4_1s(nn.Module):
    def __init__(self, with_bn=True, bn_before_actfn=True, joint_num=44, spatial_ps=False,dropout_rate=0.5):
        super(DEFORM_PSCNN_NONLOCAL_MULTI_4_1s, self).__init__()

        self.C_in = 3
        self.J = joint_num
        self.T = 39
        self.C_out = 48 * joint_num * 9
        self.cls_num = 20

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


        if self.increase_dimension_early:
            self.stem = STEM(spatial=self.spatial, temporal=self.temporal, conv=self.conv, C_in=self.C_in, C_out=32, with_bn=self.with_bn, \
                bn_before_actfn=self.bn_before_actfn)

            
        else:
            self.stem = nn.Sequential()
            self.stem.add_module("conv", nn.Conv2d(self.C_in, 32, kernel_size=(1, 1), padding=(0, 0)))
            if self.with_bn and self.bn_before_actfn:
                self.stem.add_module("bn", nn.BatchNorm2d(32))
            self.stem.add_module("relu", nn.ReLU())
            if self.with_bn and not self.bn_before_actfn:
                self.stem.add_module("bn", nn.BatchNorm2d(32))

        self.offset_1 = ConvOffset2D_nonlocal2(32, spatial=self.nonlocal_spatial, temporal=self.nonlocal_temporal, \
            concat=self.nonlocal_concat, if_self=self.nonlocal_self)
        self.offset_2 = ConvOffset2D_nonlocal2(32, spatial=self.nonlocal_spatial, temporal=self.nonlocal_temporal, \
            concat=self.nonlocal_concat, if_self=self.nonlocal_self)
        self.offset_3 = ConvOffset2D_nonlocal2(32, spatial=self.nonlocal_spatial, temporal=self.nonlocal_temporal, \
            concat=self.nonlocal_concat, if_self=self.nonlocal_self)
        self.offset_4 = ConvOffset2D_nonlocal2(32, spatial=self.nonlocal_spatial, temporal=self.nonlocal_temporal, \
            concat=self.nonlocal_concat, if_self=self.nonlocal_self)

        self.sig1 = SigModuleParallel_cheng(32, 3, 32, 3, win_size=3, use_bottleneck=True, spatial_ps=self.spatial_ps)


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
        self.conv3.add_module("conv", nn.Conv2d(32 + 32, 32, kernel_size=(1, 3), padding=(0, 1)))
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


        B, C, J, T = x_in.shape


        x_nonlocal_1, x_in_off_1, offset_1 = self.offset_1(x_in)
        x_nonlocal_2, x_in_off_2, offset_2 = self.offset_2(x_in)
        x_nonlocal_3, x_in_off_3, offset_3 = self.offset_3(x_in)
        x_nonlocal_4, x_in_off_4, offset_4 = self.offset_4(x_in)

        x_nonlocal = torch.cat([x_nonlocal_1, x_nonlocal_2, x_nonlocal_3, x_nonlocal_4], dim=0)
        x_in_off = torch.cat([x_in_off_1, x_in_off_2, x_in_off_3, x_in_off_4], dim=0)
        offset = torch.cat([offset_1, offset_2, offset_3, offset_4], dim=0)

        x_in_ps = self.sig1(x_in_off)

        x_in_ps = torch.reshape(x_in_ps, (4, B, C, J, T)).permute((1, 0, 2, 3, 4)).reshape((B, -1, J, T))
        x_in_ps = self.sig_fusion(x_in_ps)

        x_nonlocal = torch.reshape(x_nonlocal, (4, B, C, J, T)).permute((1, 0, 2, 3, 4)).reshape((B, -1, J, T))
        x_nonlocal = self.nonlocal_fusion(x_nonlocal)

        x_in = x_in + x_nonlocal
        x_in = torch.cat([x_in, x_in_ps], 1)


        x_in = self.conv3(x_in)
        x_in = self.pool3(x_in)
        x_in = self.conv_fused(x_in)
        x_in = self.pool_fused(x_in)

        x_4 = x_in.view(x_in.shape[0], -1)

        x = self.fc1(x_4)
        x = self.dp1(x)
        x = self.fc2(x)
        
        # offset = torch.Tensor.reshape(offset, (4, B, 32, 11, 39, 3)).float().mean(dim=0).squeeze()
        # return x, offset
        return x


class DEFORM_PSCNN_WITHOUTSTEM_MULTI_4_1s(nn.Module):
    def __init__(self, with_bn=True, bn_before_actfn=True, joint_num=44, spatial_ps=False,dropout_rate=0.5):
        super(DEFORM_PSCNN_WITHOUTSTEM_MULTI_4_1s, self).__init__()

        self.C_in = 3
        self.J = joint_num
        self.T = 39
        self.C_out = 48 * joint_num * 9
        self.cls_num = 20

        self.with_bn = with_bn
        self.bn_before_actfn = bn_before_actfn
        self.spatial_ps = spatial_ps

        self.increase_dimension_early = False
        self.spatial = True
        self.temporal = True
        self.conv = True

        self.non_local = True
        self.double = False

        self.nonlocal_spatial = True
        self.nonlocal_temporal = False
        self.nonlocal_concat = False
        self.nonlocal_self = True


        if self.increase_dimension_early:
            self.stem = STEM(spatial=self.spatial, temporal=self.temporal, conv=self.conv, C_in=self.C_in, C_out=32, with_bn=self.with_bn, \
                bn_before_actfn=self.bn_before_actfn)

            
        else:
            self.stem = nn.Sequential()
            self.stem.add_module("conv", nn.Conv2d(self.C_in, 32, kernel_size=(1, 1), padding=(0, 0)))
            if self.with_bn and self.bn_before_actfn:
                self.stem.add_module("bn", nn.BatchNorm2d(32))
            self.stem.add_module("relu", nn.ReLU())
            if self.with_bn and not self.bn_before_actfn:
                self.stem.add_module("bn", nn.BatchNorm2d(32))

        self.offset_1 = ConvOffset2D_nonlocal2(32, spatial=self.nonlocal_spatial, temporal=self.nonlocal_temporal, \
            concat=self.nonlocal_concat, if_self=self.nonlocal_self)
        self.offset_2 = ConvOffset2D_nonlocal2(32, spatial=self.nonlocal_spatial, temporal=self.nonlocal_temporal, \
            concat=self.nonlocal_concat, if_self=self.nonlocal_self)
        self.offset_3 = ConvOffset2D_nonlocal2(32, spatial=self.nonlocal_spatial, temporal=self.nonlocal_temporal, \
            concat=self.nonlocal_concat, if_self=self.nonlocal_self)
        self.offset_4 = ConvOffset2D_nonlocal2(32, spatial=self.nonlocal_spatial, temporal=self.nonlocal_temporal, \
            concat=self.nonlocal_concat, if_self=self.nonlocal_self)

        self.sig1 = SigModuleParallel_cheng(32, 3, 32, 3, win_size=3, use_bottleneck=True, spatial_ps=self.spatial_ps)


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
        self.conv3.add_module("conv", nn.Conv2d(32 + 32, 32, kernel_size=(1, 3), padding=(0, 1)))
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


        B, C, J, T = x_in.shape


        x_nonlocal_1, x_in_off_1, offset_1 = self.offset_1(x_in)
        x_nonlocal_2, x_in_off_2, offset_2 = self.offset_2(x_in)
        x_nonlocal_3, x_in_off_3, offset_3 = self.offset_3(x_in)
        x_nonlocal_4, x_in_off_4, offset_4 = self.offset_4(x_in)

        x_nonlocal = torch.cat([x_nonlocal_1, x_nonlocal_2, x_nonlocal_3, x_nonlocal_4], dim=0)
        x_in_off = torch.cat([x_in_off_1, x_in_off_2, x_in_off_3, x_in_off_4], dim=0)
        offset = torch.cat([offset_1, offset_2, offset_3, offset_4], dim=0)

        x_in_ps = self.sig1(x_in_off)

        x_in_ps = torch.reshape(x_in_ps, (4, B, C, J, T)).permute((1, 0, 2, 3, 4)).reshape((B, -1, J, T))
        x_in_ps = self.sig_fusion(x_in_ps)

        x_nonlocal = torch.reshape(x_nonlocal, (4, B, C, J, T)).permute((1, 0, 2, 3, 4)).reshape((B, -1, J, T))
        x_nonlocal = self.nonlocal_fusion(x_nonlocal)

        x_in = x_in + x_nonlocal
        x_in = torch.cat([x_in, x_in_ps], 1)


        x_in = self.conv3(x_in)
        x_in = self.pool3(x_in)
        x_in = self.conv_fused(x_in)
        x_in = self.pool_fused(x_in)


        x_4 = x_in.view(x_in.shape[0], -1)

        x = self.fc1(x_4)
        x = self.dp1(x)
        x = self.fc2(x)
        
        # offset = torch.Tensor.reshape(offset, (4, B, 32, 11, 39, 3)).float().mean(dim=0).squeeze()
        # return x, offset
        return x


class DEFORM_PSCNN_NONLOCAL_MULTI_4_12s(nn.Module):
    def __init__(self, with_bn=True, bn_before_actfn=True, joint_num=44, spatial_ps=False,dropout_rate=0.5):
        super(DEFORM_PSCNN_NONLOCAL_MULTI_4_12s, self).__init__()

        self.C_in = 3
        self.J = joint_num
        self.T = 39
        self.C_out = 48 * joint_num * 9
        self.cls_num = 20

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


        if self.increase_dimension_early:
            self.stem = STEM(spatial=self.spatial, temporal=self.temporal, conv=self.conv, C_in=self.C_in, C_out=32, with_bn=self.with_bn, \
                bn_before_actfn=self.bn_before_actfn)

            
        else:
            self.stem = nn.Sequential()
            self.stem.add_module("conv", nn.Conv2d(self.C_in, 32, kernel_size=(1, 1), padding=(0, 0)))
            if self.with_bn and self.bn_before_actfn:
                self.stem.add_module("bn", nn.BatchNorm2d(32))
            self.stem.add_module("relu", nn.ReLU())
            if self.with_bn and not self.bn_before_actfn:
                self.stem.add_module("bn", nn.BatchNorm2d(32))

        self.block_1 = NON_LOCAL_BLOCK(with_bn=self.with_bn, bn_before_actfn=self.bn_before_actfn, spatial_ps=self.spatial_ps, in_channels=32)

        self.conv3 = nn.Sequential()
        self.conv3.add_module("conv", nn.Conv2d(32 + 32, 32, kernel_size=(1, 3), padding=(0, 1)))
        if self.with_bn and self.bn_before_actfn:
            self.conv3.add_module("bn", nn.BatchNorm2d(32))
        self.conv3.add_module("relu", nn.ReLU())
        if self.with_bn and not self.bn_before_actfn:
            self.conv3.add_module("bn", nn.BatchNorm2d(32))

        self.pool3 = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 2))

        self.block_2 = NON_LOCAL_BLOCK(with_bn=self.with_bn, bn_before_actfn=self.bn_before_actfn, spatial_ps=self.spatial_ps, in_channels=32)

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


        x_in = self.block_1(x_in)


        x_in = self.conv3(x_in)
        x_in = self.pool3(x_in)


        x_in = self.block_2(x_in)


        x_in = self.conv_fused(x_in)
        x_in = self.pool_fused(x_in)

        x_4 = x_in.view(x_in.shape[0], -1)

        x = self.fc1(x_4)
        x = self.dp1(x)
        x = self.fc2(x)
        
        # offset = torch.Tensor.reshape(offset, (4, B, 32, 11, 39, 3)).float().mean(dim=0).squeeze()
        # return x, offset
        return x


class DEFORM_PSCNN_WITHOUTSTEM_MULTI_4_12s(nn.Module):
    def __init__(self, with_bn=True, bn_before_actfn=True, joint_num=44, spatial_ps=False,dropout_rate=0.5):
        super(DEFORM_PSCNN_WITHOUTSTEM_MULTI_4_12s, self).__init__()

        self.C_in = 3
        self.J = joint_num
        self.T = 39
        self.C_out = 48 * joint_num * 9
        self.cls_num = 20

        self.with_bn = with_bn
        self.bn_before_actfn = bn_before_actfn
        self.spatial_ps = spatial_ps

        self.increase_dimension_early = False
        self.spatial = True
        self.temporal = True
        self.conv = True

        self.non_local = True
        self.double = False

        self.nonlocal_spatial = True
        self.nonlocal_temporal = False
        self.nonlocal_concat = False
        self.nonlocal_self = True


        if self.increase_dimension_early:
            self.stem = STEM(spatial=self.spatial, temporal=self.temporal, conv=self.conv, C_in=self.C_in, C_out=32, with_bn=self.with_bn, \
                bn_before_actfn=self.bn_before_actfn)

            
        else:
            self.stem = nn.Sequential()
            self.stem.add_module("conv", nn.Conv2d(self.C_in, 32, kernel_size=(1, 1), padding=(0, 0)))
            if self.with_bn and self.bn_before_actfn:
                self.stem.add_module("bn", nn.BatchNorm2d(32))
            self.stem.add_module("relu", nn.ReLU())
            if self.with_bn and not self.bn_before_actfn:
                self.stem.add_module("bn", nn.BatchNorm2d(32))

        self.block_1 = NON_LOCAL_BLOCK(with_bn=self.with_bn, bn_before_actfn=self.bn_before_actfn, spatial_ps=self.spatial_ps, in_channels=32)

        self.conv3 = nn.Sequential()
        self.conv3.add_module("conv", nn.Conv2d(32 + 32, 32, kernel_size=(1, 3), padding=(0, 1)))
        if self.with_bn and self.bn_before_actfn:
            self.conv3.add_module("bn", nn.BatchNorm2d(32))
        self.conv3.add_module("relu", nn.ReLU())
        if self.with_bn and not self.bn_before_actfn:
            self.conv3.add_module("bn", nn.BatchNorm2d(32))

        self.pool3 = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 2))

        self.block_2 = NON_LOCAL_BLOCK(with_bn=self.with_bn, bn_before_actfn=self.bn_before_actfn, spatial_ps=self.spatial_ps, in_channels=32)

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


        x_in = self.block_1(x_in)


        x_in = self.conv3(x_in)
        x_in = self.pool3(x_in)


        x_in = self.block_2(x_in)


        x_in = self.conv_fused(x_in)
        x_in = self.pool_fused(x_in)

        x_4 = x_in.view(x_in.shape[0], -1)

        x = self.fc1(x_4)
        x = self.dp1(x)
        x = self.fc2(x)
        
        # offset = torch.Tensor.reshape(offset, (4, B, 32, 11, 39, 3)).float().mean(dim=0).squeeze()
        # return x, offset
        return x


class DEFORM_PSCNN_NONLOCAL_MULTI_4_123s(nn.Module):
    def __init__(self, with_bn=True, bn_before_actfn=True, joint_num=44, spatial_ps=False,dropout_rate=0.5):
        super(DEFORM_PSCNN_NONLOCAL_MULTI_4_123s, self).__init__()

        self.C_in = 3
        self.J = joint_num
        self.T = 39
        self.C_out = 48 * joint_num * 9 * 2
        self.cls_num = 20

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


        if self.increase_dimension_early:
            self.stem = STEM(spatial=self.spatial, temporal=self.temporal, conv=self.conv, C_in=self.C_in, C_out=32, with_bn=self.with_bn, \
                bn_before_actfn=self.bn_before_actfn)

            
        else:
            self.stem = nn.Sequential()
            self.stem.add_module("conv", nn.Conv2d(self.C_in, 32, kernel_size=(1, 1), padding=(0, 0)))
            if self.with_bn and self.bn_before_actfn:
                self.stem.add_module("bn", nn.BatchNorm2d(32))
            self.stem.add_module("relu", nn.ReLU())
            if self.with_bn and not self.bn_before_actfn:
                self.stem.add_module("bn", nn.BatchNorm2d(32))

        self.block_1 = NON_LOCAL_BLOCK(with_bn=self.with_bn, bn_before_actfn=self.bn_before_actfn, spatial_ps=self.spatial_ps, in_channels=32)

        self.conv3 = nn.Sequential()
        self.conv3.add_module("conv", nn.Conv2d(32 + 32, 32, kernel_size=(1, 3), padding=(0, 1)))
        if self.with_bn and self.bn_before_actfn:
            self.conv3.add_module("bn", nn.BatchNorm2d(32))
        self.conv3.add_module("relu", nn.ReLU())
        if self.with_bn and not self.bn_before_actfn:
            self.conv3.add_module("bn", nn.BatchNorm2d(32))

        self.pool3 = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 2))

        self.block_2 = NON_LOCAL_BLOCK(with_bn=self.with_bn, bn_before_actfn=self.bn_before_actfn, spatial_ps=self.spatial_ps, in_channels=32)

        self.conv_fused = nn.Sequential()
        self.conv_fused.add_module("conv", nn.Conv2d(32 + 32, 48, kernel_size=(1, 3), padding=(0, 1)))
        if self.with_bn and self.bn_before_actfn:
            self.conv_fused.add_module("bn", nn.BatchNorm2d(48))
        self.conv_fused.add_module("relu", nn.ReLU())
        if self.with_bn and not self.bn_before_actfn:
            self.conv_fused.add_module("bn", nn.BatchNorm2d(48))
            
        self.pool_fused = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 2))

        self.block_3 = NON_LOCAL_BLOCK(with_bn=self.with_bn, bn_before_actfn=self.bn_before_actfn, spatial_ps=self.spatial_ps, in_channels=48)

        self.fc1 = nn.Linear(self.C_out, 256)
        self.dp1 = nn.Dropout(p=dropout_rate)
        self.fc2 = nn.Linear(256, self.cls_num)

        self.apply(weight_init)





    def forward(self, x):
        x_in = input_prepare(x)
        # x_in.shape = (256, 3, 11, 39)

        x_in = self.stem(x_in)


        x_in = self.block_1(x_in)


        x_in = self.conv3(x_in)
        x_in = self.pool3(x_in)


        x_in = self.block_2(x_in)


        x_in = self.conv_fused(x_in)
        x_in = self.pool_fused(x_in)

        x_in = self.block_3(x_in)

        x_4 = x_in.view(x_in.shape[0], -1)

        x = self.fc1(x_4)
        x = self.dp1(x)
        x = self.fc2(x)
        
        # offset = torch.Tensor.reshape(offset, (4, B, 32, 11, 39, 3)).float().mean(dim=0).squeeze()
        # return x, offset
        return x


class DEFORM_PSCNN_WITHOUTSTEM_MULTI_4_123s(nn.Module):
    def __init__(self, with_bn=True, bn_before_actfn=True, joint_num=44, spatial_ps=False,dropout_rate=0.5):
        super(DEFORM_PSCNN_WITHOUTSTEM_MULTI_4_123s, self).__init__()

        self.C_in = 3
        self.J = joint_num
        self.T = 39
        self.C_out = 48 * joint_num * 9 * 2
        self.cls_num = 20

        self.with_bn = with_bn
        self.bn_before_actfn = bn_before_actfn
        self.spatial_ps = spatial_ps

        self.increase_dimension_early = False
        self.spatial = True
        self.temporal = True
        self.conv = True

        self.non_local = True
        self.double = False

        self.nonlocal_spatial = True
        self.nonlocal_temporal = False
        self.nonlocal_concat = False
        self.nonlocal_self = True


        if self.increase_dimension_early:
            self.stem = STEM(spatial=self.spatial, temporal=self.temporal, conv=self.conv, C_in=self.C_in, C_out=32, with_bn=self.with_bn, \
                bn_before_actfn=self.bn_before_actfn)

            
        else:
            self.stem = nn.Sequential()
            self.stem.add_module("conv", nn.Conv2d(self.C_in, 32, kernel_size=(1, 1), padding=(0, 0)))
            if self.with_bn and self.bn_before_actfn:
                self.stem.add_module("bn", nn.BatchNorm2d(32))
            self.stem.add_module("relu", nn.ReLU())
            if self.with_bn and not self.bn_before_actfn:
                self.stem.add_module("bn", nn.BatchNorm2d(32))

        self.block_1 = NON_LOCAL_BLOCK(with_bn=self.with_bn, bn_before_actfn=self.bn_before_actfn, spatial_ps=self.spatial_ps, in_channels=32)

        self.conv3 = nn.Sequential()
        self.conv3.add_module("conv", nn.Conv2d(32 + 32, 32, kernel_size=(1, 3), padding=(0, 1)))
        if self.with_bn and self.bn_before_actfn:
            self.conv3.add_module("bn", nn.BatchNorm2d(32))
        self.conv3.add_module("relu", nn.ReLU())
        if self.with_bn and not self.bn_before_actfn:
            self.conv3.add_module("bn", nn.BatchNorm2d(32))

        self.pool3 = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 2))

        self.block_2 = NON_LOCAL_BLOCK(with_bn=self.with_bn, bn_before_actfn=self.bn_before_actfn, spatial_ps=self.spatial_ps, in_channels=32)

        self.conv_fused = nn.Sequential()
        self.conv_fused.add_module("conv", nn.Conv2d(32 + 32, 48, kernel_size=(1, 3), padding=(0, 1)))
        if self.with_bn and self.bn_before_actfn:
            self.conv_fused.add_module("bn", nn.BatchNorm2d(48))
        self.conv_fused.add_module("relu", nn.ReLU())
        if self.with_bn and not self.bn_before_actfn:
            self.conv_fused.add_module("bn", nn.BatchNorm2d(48))
            
        self.pool_fused = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 2))

        self.block_3 = NON_LOCAL_BLOCK(with_bn=self.with_bn, bn_before_actfn=self.bn_before_actfn, spatial_ps=self.spatial_ps, in_channels=48)

        self.fc1 = nn.Linear(self.C_out, 256)
        self.dp1 = nn.Dropout(p=dropout_rate)
        self.fc2 = nn.Linear(256, self.cls_num)

        self.apply(weight_init)





    def forward(self, x):
        x_in = input_prepare(x)
        # x_in.shape = (256, 3, 11, 39)

        x_in = self.stem(x_in)


        x_in = self.block_1(x_in)


        x_in = self.conv3(x_in)
        x_in = self.pool3(x_in)


        x_in = self.block_2(x_in)


        x_in = self.conv_fused(x_in)
        x_in = self.pool_fused(x_in)


        x_in = self.block_3(x_in)


        x_4 = x_in.view(x_in.shape[0], -1)

        x = self.fc1(x_4)
        x = self.dp1(x)
        x = self.fc2(x)
        
        # offset = torch.Tensor.reshape(offset, (4, B, 32, 11, 39, 3)).float().mean(dim=0).squeeze()
        # return x, offset
        return x


class DEFORM_PSCNN_WITHOUTSTEM_MULTI_4_1s(nn.Module):
    def __init__(self, with_bn=True, bn_before_actfn=True, joint_num=44, spatial_ps=False,dropout_rate=0.5):
        super(DEFORM_PSCNN_WITHOUTSTEM_MULTI_4_1s, self).__init__()

        self.C_in = 3
        self.J = joint_num
        self.T = 39
        self.C_out = 48 * joint_num * 9
        self.cls_num = 20

        self.with_bn = with_bn
        self.bn_before_actfn = bn_before_actfn
        self.spatial_ps = spatial_ps

        self.increase_dimension_early = False
        self.spatial = True
        self.temporal = True
        self.conv = True

        self.non_local = True
        self.double = False

        self.nonlocal_spatial = True
        self.nonlocal_temporal = False
        self.nonlocal_concat = False
        self.nonlocal_self = True


        if self.increase_dimension_early:
            self.stem = STEM(spatial=self.spatial, temporal=self.temporal, conv=self.conv, C_in=self.C_in, C_out=32, with_bn=self.with_bn, \
                bn_before_actfn=self.bn_before_actfn)

            
        else:
            self.stem = nn.Sequential()
            self.stem.add_module("conv", nn.Conv2d(self.C_in, 32, kernel_size=(1, 1), padding=(0, 0)))
            if self.with_bn and self.bn_before_actfn:
                self.stem.add_module("bn", nn.BatchNorm2d(32))
            self.stem.add_module("relu", nn.ReLU())
            if self.with_bn and not self.bn_before_actfn:
                self.stem.add_module("bn", nn.BatchNorm2d(32))

        self.offset_1 = ConvOffset2D_nonlocal2(32, spatial=self.nonlocal_spatial, temporal=self.nonlocal_temporal, \
            concat=self.nonlocal_concat, if_self=self.nonlocal_self)
        self.offset_2 = ConvOffset2D_nonlocal2(32, spatial=self.nonlocal_spatial, temporal=self.nonlocal_temporal, \
            concat=self.nonlocal_concat, if_self=self.nonlocal_self)
        self.offset_3 = ConvOffset2D_nonlocal2(32, spatial=self.nonlocal_spatial, temporal=self.nonlocal_temporal, \
            concat=self.nonlocal_concat, if_self=self.nonlocal_self)
        self.offset_4 = ConvOffset2D_nonlocal2(32, spatial=self.nonlocal_spatial, temporal=self.nonlocal_temporal, \
            concat=self.nonlocal_concat, if_self=self.nonlocal_self)

        self.sig1 = SigModuleParallel_cheng(32, 3, 32, 3, win_size=3, use_bottleneck=True, spatial_ps=self.spatial_ps)


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
        self.conv3.add_module("conv", nn.Conv2d(32 + 32, 32, kernel_size=(1, 3), padding=(0, 1)))
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


        B, C, J, T = x_in.shape


        x_nonlocal_1, x_in_off_1, offset_1 = self.offset_1(x_in)
        x_nonlocal_2, x_in_off_2, offset_2 = self.offset_2(x_in)
        x_nonlocal_3, x_in_off_3, offset_3 = self.offset_3(x_in)
        x_nonlocal_4, x_in_off_4, offset_4 = self.offset_4(x_in)

        x_nonlocal = torch.cat([x_nonlocal_1, x_nonlocal_2, x_nonlocal_3, x_nonlocal_4], dim=0)
        x_in_off = torch.cat([x_in_off_1, x_in_off_2, x_in_off_3, x_in_off_4], dim=0)
        offset = torch.cat([offset_1, offset_2, offset_3, offset_4], dim=0)

        x_in_ps = self.sig1(x_in_off)

        x_in_ps = torch.reshape(x_in_ps, (4, B, C, J, T)).permute((1, 0, 2, 3, 4)).reshape((B, -1, J, T))
        x_in_ps = self.sig_fusion(x_in_ps)

        x_nonlocal = torch.reshape(x_nonlocal, (4, B, C, J, T)).permute((1, 0, 2, 3, 4)).reshape((B, -1, J, T))
        x_nonlocal = self.nonlocal_fusion(x_nonlocal)

        x_in = x_in + x_nonlocal
        x_in = torch.cat([x_in, x_in_ps], 1)


        x_in = self.conv3(x_in)
        x_in = self.pool3(x_in)
        x_in = self.conv_fused(x_in)
        x_in = self.pool_fused(x_in)


        x_4 = x_in.view(x_in.shape[0], -1)

        x = self.fc1(x_4)
        x = self.dp1(x)
        x = self.fc2(x)
        
        # offset = torch.Tensor.reshape(offset, (4, B, 32, 11, 39, 3)).float().mean(dim=0).squeeze()
        # return x, offset
        return x


class NON_LOCAL_BLOCK(nn.Module):
    def __init__(self, with_bn, bn_before_actfn, spatial_ps, in_channels):
        super(NON_LOCAL_BLOCK, self).__init__()

        self.with_bn = with_bn
        self.bn_before_actfn = bn_before_actfn
        self.spatial_ps = spatial_ps

        self.non_local = True
        self.double = False
        self.in_channels = in_channels

        self.nonlocal_spatial = True
        self.nonlocal_temporal = False
        self.nonlocal_concat = False
        self.nonlocal_self = True


        self.offset_1 = ConvOffset2D_nonlocal2(self.in_channels, spatial=self.nonlocal_spatial, temporal=self.nonlocal_temporal, \
            concat=self.nonlocal_concat, if_self=self.nonlocal_self)
        self.offset_2 = ConvOffset2D_nonlocal2(self.in_channels, spatial=self.nonlocal_spatial, temporal=self.nonlocal_temporal, \
            concat=self.nonlocal_concat, if_self=self.nonlocal_self)
        self.offset_3 = ConvOffset2D_nonlocal2(self.in_channels, spatial=self.nonlocal_spatial, temporal=self.nonlocal_temporal, \
            concat=self.nonlocal_concat, if_self=self.nonlocal_self)
        self.offset_4 = ConvOffset2D_nonlocal2(self.in_channels, spatial=self.nonlocal_spatial, temporal=self.nonlocal_temporal, \
            concat=self.nonlocal_concat, if_self=self.nonlocal_self)

        self.sig1 = SigModuleParallel_cheng(self.in_channels, 3, self.in_channels, 3, win_size=3, use_bottleneck=True, spatial_ps=self.spatial_ps)


        self.nonlocal_fusion = nn.Sequential()
        self.nonlocal_fusion.add_module("conv", nn.Conv2d(self.in_channels * 4, self.in_channels, kernel_size=(1, 1), padding=(0, 0)))
        if self.with_bn and self.bn_before_actfn:
            self.nonlocal_fusion.add_module("bn", nn.BatchNorm2d(self.in_channels))
        self.nonlocal_fusion.add_module("relu", nn.ReLU())
        if self.with_bn and not self.bn_before_actfn:
            self.nonlocal_fusion.add_module("bn", nn.BatchNorm2d(self.in_channels))

        self.sig_fusion = nn.Sequential()
        self.sig_fusion.add_module("conv", nn.Conv2d(self.in_channels * 4, self.in_channels, kernel_size=(1, 1), padding=(0, 0)))
        if self.with_bn and self.bn_before_actfn:
            self.sig_fusion.add_module("bn", nn.BatchNorm2d(self.in_channels))
        self.sig_fusion.add_module("relu", nn.ReLU())
        if self.with_bn and not self.bn_before_actfn:
            self.sig_fusion.add_module("bn", nn.BatchNorm2d(self.in_channels))


    def forward(self, x_in):
        B, C, J, T = x_in.shape


        x_nonlocal_1, x_in_off_1, offset_1 = self.offset_1(x_in)
        x_nonlocal_2, x_in_off_2, offset_2 = self.offset_2(x_in)
        x_nonlocal_3, x_in_off_3, offset_3 = self.offset_3(x_in)
        x_nonlocal_4, x_in_off_4, offset_4 = self.offset_4(x_in)

        x_nonlocal = torch.cat([x_nonlocal_1, x_nonlocal_2, x_nonlocal_3, x_nonlocal_4], dim=0)
        x_in_off = torch.cat([x_in_off_1, x_in_off_2, x_in_off_3, x_in_off_4], dim=0)
        offset = torch.cat([offset_1, offset_2, offset_3, offset_4], dim=0)

        x_in_ps = self.sig1(x_in_off)

        x_in_ps = torch.reshape(x_in_ps, (4, B, C, J, T)).permute((1, 0, 2, 3, 4)).reshape((B, -1, J, T))
        x_in_ps = self.sig_fusion(x_in_ps)

        x_nonlocal = torch.reshape(x_nonlocal, (4, B, C, J, T)).permute((1, 0, 2, 3, 4)).reshape((B, -1, J, T))
        x_nonlocal = self.nonlocal_fusion(x_nonlocal)

        x_in = x_in + x_nonlocal
        x_in = torch.cat([x_in, x_in_ps], 1)

        return x_in


class BASELINE_DecoupleGCN_NL_1(nn.Module):
    def __init__(self, with_bn=True, bn_before_actfn=True, joint_num=11, dropout_rate=0.5):
        super(BASELINE_DecoupleGCN_NL_1, self).__init__()
        self.backbone_1 = DecoupleGCN_Model(num_class=20, num_point=11, num_person=1, \
            graph=chalearn13_hand_crafted_decoupled_adjacent_matrix, in_channels=64, groups=8)

        self.C_in = 3
        self.J = joint_num
        self.T = 39
        self.C_out = 48 * joint_num * 9
        self.cls_num = 20

        self.with_bn = with_bn
        self.bn_before_actfn = bn_before_actfn
        self.spatial_ps = False

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

        if self.increase_dimension_early:
            self.stem = STEM(spatial=self.spatial, temporal=self.temporal, conv=self.conv, C_in=3, C_out=32, with_bn=self.with_bn, \
                bn_before_actfn=self.bn_before_actfn)

            self.offset_1 = ConvOffset2D_nonlocal2(32, spatial=self.nonlocal_spatial, temporal=self.nonlocal_temporal, \
                concat=self.nonlocal_concat, if_self=self.nonlocal_self)

            
        else:
            self.offset_1 = ConvOffset2D_nonlocal2(3, spatial=self.nonlocal_spatial, temporal=self.nonlocal_temporal, \
                concat=self.nonlocal_concat, if_self=self.nonlocal_self)


        self.sig1 = SigModuleParallel_cheng(32, 3, 32, 3, win_size=3, use_bottleneck=True, spatial_ps=self.spatial_ps)


        self.nonlocal_fusion = nn.Sequential()
        self.nonlocal_fusion.add_module("conv", nn.Conv2d(32 * 1, 32, kernel_size=(1, 1), padding=(0, 0)))
        if self.with_bn and self.bn_before_actfn:
            self.nonlocal_fusion.add_module("bn", nn.BatchNorm2d(32))
        self.nonlocal_fusion.add_module("relu", nn.ReLU())
        if self.with_bn and not self.bn_before_actfn:
            self.nonlocal_fusion.add_module("bn", nn.BatchNorm2d(32))

        self.sig_fusion = nn.Sequential()
        self.sig_fusion.add_module("conv", nn.Conv2d(32 * 1, 32, kernel_size=(1, 1), padding=(0, 0)))
        if self.with_bn and self.bn_before_actfn:
            self.sig_fusion.add_module("bn", nn.BatchNorm2d(32))
        self.sig_fusion.add_module("relu", nn.ReLU())
        if self.with_bn and not self.bn_before_actfn:
            self.sig_fusion.add_module("bn", nn.BatchNorm2d(32))

    def forward(self, x):
        x_in = input_prepare(x)


        B, C, J, T = x_in.shape

        if self.increase_dimension_early:
            x_in = self.stem(x_in)

        x_nonlocal_1, x_in_off_1, offset_1 = self.offset_1(x_in)

        x_nonlocal = torch.cat([x_nonlocal_1], dim=0)
        x_in_off = torch.cat([x_in_off_1], dim=0)
        offset = torch.cat([offset_1], dim=0)

        x_in_ps = self.sig1(x_in_off)

        x_in_ps = torch.reshape(x_in_ps, (1, B, 32, J, T)).permute((1, 0, 2, 3, 4)).reshape((B, -1, J, T))
        x_in_ps = self.sig_fusion(x_in_ps)

        x_nonlocal = torch.reshape(x_nonlocal, (1, B, 32, J, T)).permute((1, 0, 2, 3, 4)).reshape((B, -1, J, T))
        x_nonlocal = self.nonlocal_fusion(x_nonlocal)

        x_in = x_in + x_nonlocal
        x_in = torch.cat([x_in, x_in_ps], 1)


        x_in = torch.Tensor.permute(x_in, (0, 1, 3, 2)).unsqueeze(dim=4)
        # N, C, T, J, 1

        output_1 = torch.softmax(self.backbone_1(x_in), dim=1)

        return output_1


class BASELINE_DecoupleGCN_WITHOUTNL(nn.Module):
    def __init__(self, with_bn=True, bn_before_actfn=True, joint_num=11, dropout_rate=0.5):
        super(BASELINE_DecoupleGCN_WITHOUTNL, self).__init__()
        self.backbone_1 = DecoupleGCN_Model(num_class=20, num_point=11, num_person=1, \
            graph=chalearn13_hand_crafted_decoupled_adjacent_matrix, in_channels=32, groups=8)

        self.C_in = 3
        self.J = joint_num
        self.T = 39
        self.C_out = 48 * joint_num * 9
        self.cls_num = 20

        self.with_bn = with_bn
        self.bn_before_actfn = bn_before_actfn
        self.spatial_ps = False

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

        if self.increase_dimension_early:
            self.stem = STEM(spatial=self.spatial, temporal=self.temporal, conv=self.conv, C_in=3, C_out=32, with_bn=self.with_bn, \
                bn_before_actfn=self.bn_before_actfn)


    def forward(self, x):
        x_in = input_prepare(x)


        B, C, J, T = x_in.shape

        if self.increase_dimension_early:
            x_in = self.stem(x_in)

        x_in = torch.Tensor.permute(x_in, (0, 1, 3, 2)).unsqueeze(dim=4)
        # N, C, T, J, 1

        output_1 = torch.softmax(self.backbone_1(x_in), dim=1)

        return output_1


class DEFORM_PSCNN_NONLOCAL_MULTI_4_2s_withall(nn.Module):
    def __init__(self, with_bn=True, bn_before_actfn=True, joint_num=44, spatial_ps=False,dropout_rate=0.5):
        super(DEFORM_PSCNN_NONLOCAL_MULTI_4_2s_withall, self).__init__()

        self.C_in = 3
        self.J = joint_num
        self.T = 39
        self.C_out = 48 * joint_num * 9
        self.cls_num = 20

        self.with_bn = with_bn
        self.bn_before_actfn = bn_before_actfn
        self.spatial_ps = spatial_ps

        self.increase_dimension_early = True
        self.spatial = True
        self.temporal = True
        self.conv = True

        self.non_local = True
        self.double = False

        self.use_bottleneck = True
        self.use_linear = True

        self.nonlocal_spatial = True
        self.nonlocal_temporal = False
        self.nonlocal_concat = False
        self.nonlocal_self = True


        if self.increase_dimension_early:
            self.stem = STEM2(spatial=self.spatial, temporal=self.temporal, conv=self.conv, C_in=self.C_in, C_out=32, with_bn=self.with_bn, \
                bn_before_actfn=self.bn_before_actfn, use_bottleneck=self.use_bottleneck, use_linear=self.use_linear)

            
        else:
            self.stem = nn.Sequential()
            self.stem.add_module("conv", nn.Conv2d(self.C_in, 32, kernel_size=(1, 1), padding=(0, 0)))
            if self.with_bn and self.bn_before_actfn:
                self.stem.add_module("bn", nn.BatchNorm2d(32))
            self.stem.add_module("relu", nn.ReLU())
            if self.with_bn and not self.bn_before_actfn:
                self.stem.add_module("bn", nn.BatchNorm2d(32))

        self.offset_1 = ConvOffset2D_nonlocal2(32, spatial=self.nonlocal_spatial, temporal=self.nonlocal_temporal, \
            concat=self.nonlocal_concat, if_self=self.nonlocal_self)
        self.offset_2 = ConvOffset2D_nonlocal2(32, spatial=self.nonlocal_spatial, temporal=self.nonlocal_temporal, \
            concat=self.nonlocal_concat, if_self=self.nonlocal_self)
        self.offset_3 = ConvOffset2D_nonlocal2(32, spatial=self.nonlocal_spatial, temporal=self.nonlocal_temporal, \
            concat=self.nonlocal_concat, if_self=self.nonlocal_self)
        self.offset_4 = ConvOffset2D_nonlocal2(32, spatial=self.nonlocal_spatial, temporal=self.nonlocal_temporal, \
            concat=self.nonlocal_concat, if_self=self.nonlocal_self)

        if not self.use_bottleneck:
            if self.use_linear:
                self.sig_channels = 32
            else:
                self.sig_channels = signatory.signature_channels(channels=32+1, depth=3)
        else:
            if self.use_linear:
                self.sig_channels = 32
            else:
                self.sig_channels = signatory.signature_channels(channels=3+1, depth=3)


        self.sig1 = SigModuleParallel_cheng(32, 3, self.sig_channels, 3, win_size=3, use_bottleneck=self.use_bottleneck, \
        spatial_ps=self.spatial_ps, use_linear=self.use_linear)

        self.nonlocal_fusion = nn.Sequential()
        self.nonlocal_fusion.add_module("conv", nn.Conv2d(32 * 4, 32, kernel_size=(1, 1), padding=(0, 0)))
        if self.with_bn and self.bn_before_actfn:
            self.nonlocal_fusion.add_module("bn", nn.BatchNorm2d(32))
        self.nonlocal_fusion.add_module("relu", nn.ReLU())
        if self.with_bn and not self.bn_before_actfn:
            self.nonlocal_fusion.add_module("bn", nn.BatchNorm2d(32))

        self.sig_fusion = nn.Sequential()
        self.sig_fusion.add_module("conv", nn.Conv2d(self.sig_channels * 4, 32, kernel_size=(1, 1), padding=(0, 0)))
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

        B, C, J, T = x_in.shape


        x_nonlocal_1, x_in_off_1, offset_1 = self.offset_1(x_in)
        x_nonlocal_2, x_in_off_2, offset_2 = self.offset_2(x_in)
        x_nonlocal_3, x_in_off_3, offset_3 = self.offset_3(x_in)
        x_nonlocal_4, x_in_off_4, offset_4 = self.offset_4(x_in)

        x_nonlocal = torch.cat([x_nonlocal_1, x_nonlocal_2, x_nonlocal_3, x_nonlocal_4], dim=0)
        x_in_off = torch.cat([x_in_off_1, x_in_off_2, x_in_off_3, x_in_off_4], dim=0)
        offset = torch.cat([offset_1, offset_2, offset_3, offset_4], dim=0)

        x_in_ps = self.sig1(x_in_off)

        if self.use_linear:
            x_in_ps = torch.reshape(x_in_ps, (4, B, 32, J, T)).permute((1, 0, 2, 3, 4)).reshape((B, -1, J, T))
        else:
             x_in_ps = torch.reshape(x_in_ps, (4, B, self.sig_channels, J, T)).permute((1, 0, 2, 3, 4)).reshape((B, -1, J, T))
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


class DEFORM_PSCNN_NONLOCAL_MULTI_4_2s_STEM_baseline(nn.Module):
    def __init__(self, with_bn=True, bn_before_actfn=True, joint_num=44, spatial_ps=False,dropout_rate=0.5):
        super(DEFORM_PSCNN_NONLOCAL_MULTI_4_2s_STEM_baseline, self).__init__()

        self.C_in = 3
        self.J = joint_num
        self.T = 39
        self.C_out = 48 * joint_num * 9
        self.cls_num = 20

        self.with_bn = with_bn
        self.bn_before_actfn = bn_before_actfn
        self.spatial_ps = spatial_ps

        self.increase_dimension_early = True
        self.spatial = True
        self.temporal = True
        self.conv = True

        self.non_local = False
        self.double = False

        self.use_bottleneck = False
        self.use_linear = False

        self.nonlocal_spatial = True
        self.nonlocal_temporal = False
        self.nonlocal_concat = False
        self.nonlocal_self = True


        if self.increase_dimension_early:
            self.stem = STEM2(spatial=self.spatial, temporal=self.temporal, conv=self.conv, C_in=self.C_in, C_out=32, with_bn=self.with_bn, \
                bn_before_actfn=self.bn_before_actfn, use_bottleneck=self.use_bottleneck, use_linear=self.use_linear)

        else:
            self.stem = nn.Sequential()
            self.stem.add_module("conv", nn.Conv2d(self.C_in, 32, kernel_size=(1, 1), padding=(0, 0)))
            if self.with_bn and self.bn_before_actfn:
                self.stem.add_module("bn", nn.BatchNorm2d(32))
            self.stem.add_module("relu", nn.ReLU())
            if self.with_bn and not self.bn_before_actfn:
                self.stem.add_module("bn", nn.BatchNorm2d(32))

        if not self.use_bottleneck:
            if self.use_linear:
                self.sig_channels = 32
            else:
                self.sig_channels = signatory.signature_channels(channels=3+1, depth=3)
        else:
            if self.use_linear:
                self.sig_channels = 32
            else:
                self.sig_channels = signatory.signature_channels(channels=3+1, depth=3)


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


class DEFORM_PSCNN_NONLOCAL_MULTI_4_2s_STEM_with1(nn.Module):
    def __init__(self, with_bn=True, bn_before_actfn=True, joint_num=44, spatial_ps=False,dropout_rate=0.5):
        super(DEFORM_PSCNN_NONLOCAL_MULTI_4_2s_STEM_with1, self).__init__()

        self.C_in = 3
        self.J = joint_num
        self.T = 39
        self.C_out = 48 * joint_num * 9
        self.cls_num = 20

        self.with_bn = with_bn
        self.bn_before_actfn = bn_before_actfn
        self.spatial_ps = spatial_ps

        self.increase_dimension_early = True
        self.spatial = True
        self.temporal = True
        self.conv = True

        self.non_local = False
        self.double = False

        self.use_bottleneck = True
        self.use_linear = False

        self.nonlocal_spatial = True
        self.nonlocal_temporal = False
        self.nonlocal_concat = False
        self.nonlocal_self = True


        if self.increase_dimension_early:
            self.stem = STEM2(spatial=self.spatial, temporal=self.temporal, conv=self.conv, C_in=self.C_in, C_out=32, with_bn=self.with_bn, \
                bn_before_actfn=self.bn_before_actfn, use_bottleneck=self.use_bottleneck, use_linear=self.use_linear)

        else:
            self.stem = nn.Sequential()
            self.stem.add_module("conv", nn.Conv2d(self.C_in, 32, kernel_size=(1, 1), padding=(0, 0)))
            if self.with_bn and self.bn_before_actfn:
                self.stem.add_module("bn", nn.BatchNorm2d(32))
            self.stem.add_module("relu", nn.ReLU())
            if self.with_bn and not self.bn_before_actfn:
                self.stem.add_module("bn", nn.BatchNorm2d(32))

        if not self.use_bottleneck:
            if self.use_linear:
                self.sig_channels = 32
            else:
                self.sig_channels = signatory.signature_channels(channels=3+1, depth=3)
        else:
            if self.use_linear:
                self.sig_channels = 32
            else:
                self.sig_channels = signatory.signature_channels(channels=3+1, depth=3)


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


class DEFORM_PSCNN_NONLOCAL_MULTI_4_2s_STEM_with2(nn.Module):
    def __init__(self, with_bn=True, bn_before_actfn=True, joint_num=44, spatial_ps=False,dropout_rate=0.5):
        super(DEFORM_PSCNN_NONLOCAL_MULTI_4_2s_STEM_with2, self).__init__()

        self.C_in = 3
        self.J = joint_num
        self.T = 39
        self.C_out = 48 * joint_num * 9
        self.cls_num = 20

        self.with_bn = with_bn
        self.bn_before_actfn = bn_before_actfn
        self.spatial_ps = spatial_ps

        self.increase_dimension_early = True
        self.spatial = True
        self.temporal = True
        self.conv = True

        self.non_local = False
        self.double = False

        self.use_bottleneck = False
        self.use_linear = True

        self.nonlocal_spatial = True
        self.nonlocal_temporal = False
        self.nonlocal_concat = False
        self.nonlocal_self = True


        if self.increase_dimension_early:
            self.stem = STEM2(spatial=self.spatial, temporal=self.temporal, conv=self.conv, C_in=self.C_in, C_out=32, with_bn=self.with_bn, \
                bn_before_actfn=self.bn_before_actfn, use_bottleneck=self.use_bottleneck, use_linear=self.use_linear)

        else:
            self.stem = nn.Sequential()
            self.stem.add_module("conv", nn.Conv2d(self.C_in, 32, kernel_size=(1, 1), padding=(0, 0)))
            if self.with_bn and self.bn_before_actfn:
                self.stem.add_module("bn", nn.BatchNorm2d(32))
            self.stem.add_module("relu", nn.ReLU())
            if self.with_bn and not self.bn_before_actfn:
                self.stem.add_module("bn", nn.BatchNorm2d(32))

        if not self.use_bottleneck:
            if self.use_linear:
                self.sig_channels = 32
            else:
                self.sig_channels = signatory.signature_channels(channels=3+1, depth=3)
        else:
            if self.use_linear:
                self.sig_channels = 32
            else:
                self.sig_channels = signatory.signature_channels(channels=3+1, depth=3)


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


class DEFORM_PSCNN_NONLOCAL_MULTI_4_2s_STEM_withall(nn.Module):
    def __init__(self, with_bn=True, bn_before_actfn=True, joint_num=44, spatial_ps=False,dropout_rate=0.5):
        super(DEFORM_PSCNN_NONLOCAL_MULTI_4_2s_STEM_withall, self).__init__()

        self.C_in = 3
        self.J = joint_num
        self.T = 39
        self.C_out = 48 * joint_num * 9
        self.cls_num = 20

        self.with_bn = with_bn
        self.bn_before_actfn = bn_before_actfn
        self.spatial_ps = spatial_ps

        self.increase_dimension_early = True
        self.spatial = True
        self.temporal = True
        self.conv = True

        self.non_local = False
        self.double = False

        self.use_bottleneck = True
        self.use_linear = True

        self.nonlocal_spatial = True
        self.nonlocal_temporal = False
        self.nonlocal_concat = False
        self.nonlocal_self = True


        if self.increase_dimension_early:
            self.stem = STEM2(spatial=self.spatial, temporal=self.temporal, conv=self.conv, C_in=self.C_in, C_out=32, with_bn=self.with_bn, \
                bn_before_actfn=self.bn_before_actfn, use_bottleneck=self.use_bottleneck, use_linear=self.use_linear)

        else:
            self.stem = nn.Sequential()
            self.stem.add_module("conv", nn.Conv2d(self.C_in, 32, kernel_size=(1, 1), padding=(0, 0)))
            if self.with_bn and self.bn_before_actfn:
                self.stem.add_module("bn", nn.BatchNorm2d(32))
            self.stem.add_module("relu", nn.ReLU())
            if self.with_bn and not self.bn_before_actfn:
                self.stem.add_module("bn", nn.BatchNorm2d(32))

        if not self.use_bottleneck:
            if self.use_linear:
                self.sig_channels = 32
            else:
                self.sig_channels = signatory.signature_channels(channels=3+1, depth=3)
        else:
            if self.use_linear:
                self.sig_channels = 32
            else:
                self.sig_channels = signatory.signature_channels(channels=3+1, depth=3)


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


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # x = torch.randn(1,12)
    # model = BASELINE_MSG3D()

    # N, C, T, V, M = 6, 3, 39, 11, 1
    # x = torch.randn(N,C,T,V,M)
    # x = torch.Tensor.reshape(x, (x.shape[0], -1))
    # model.forward(x)

    import sys
    sys.path.append('../..')
    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')

    model = BASELINE_2sAGCN.to(device)

    N, C, T, V, M = 6, 3, 39, 11, 1
    x = torch.randn(N,C,T,V,M).to(device)
    x = torch.Tensor.reshape(x, (x.shape[0], -1))
    model.forward(x)

    print('Model total # params:', count_params(model))
    pass
