import sys
sys.path.insert(0, '')

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, in_channels, out_channels, activation='relu', dropout=0):
        super().__init__()
        channels = [in_channels] + out_channels
        self.layers = nn.ModuleList()
        for i in range(1, len(channels)):
            if dropout > 0.001:
                self.layers.append(nn.Dropout(p=dropout))
            self.layers.append(nn.Conv2d(channels[i-1], channels[i], kernel_size=1))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.BatchNorm2d(channels[i]))

    def forward(self, x):
        # Input shape: (N,C,T,V)
        for layer in self.layers:
            x = layer(x)
        return x


def normalize_adjacency_matrix(A):
    node_degrees = A.sum(-1)
    degs_inv_sqrt = np.power(node_degrees, -0.5)
    norm_degs_matrix = np.eye(len(node_degrees)) * degs_inv_sqrt
    return (norm_degs_matrix @ A @ norm_degs_matrix).astype(np.float32)


class MS_G3D(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 A_binary,
                 num_scales,
                 window_size,
                 window_stride,
                 window_dilation,
                 embed_factor=1,
                 activation='relu'):
        super().__init__()
        self.window_size = window_size
        self.out_channels = out_channels
        self.embed_channels_in = self.embed_channels_out = out_channels // embed_factor
        if embed_factor == 1:
            self.in1x1 = nn.Identity()
            self.embed_channels_in = self.embed_channels_out = in_channels
            # The first STGC block changes channels right away; others change at collapse
            if in_channels == 3:
                self.embed_channels_out = out_channels
        else:
            self.in1x1 = MLP(in_channels, [self.embed_channels_in])

        self.gcn3d = nn.Sequential(
            UnfoldTemporalWindows(window_size, window_stride, window_dilation),
            # SpatialTemporal_MS_GCN(
            SpatialTemporal_MS_GCN_2s(
                in_channels=self.embed_channels_in,
                out_channels=self.embed_channels_out,
                A_binary=A_binary,
                num_scales=num_scales,
                window_size=window_size,
                use_Ares=True
            )
        )

        self.out_conv = nn.Conv3d(self.embed_channels_out, out_channels, kernel_size=(1, self.window_size, 1))
        self.out_bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        N, _, T, V = x.shape
        x = self.in1x1(x)
        # Construct temporal windows and apply MS-GCN
        x = self.gcn3d(x)

        # Collapse the window dimension
        x = x.view(N, self.embed_channels_out, -1, self.window_size, V)
        x = self.out_conv(x).squeeze(dim=3)
        x = self.out_bn(x)

        # no activation
        return x


class UnfoldTemporalWindows(nn.Module):
    def __init__(self, window_size, window_stride, window_dilation=1):
        super().__init__()
        self.window_size = window_size
        self.window_stride = window_stride
        self.window_dilation = window_dilation

        self.padding = (window_size + (window_size-1) * (window_dilation-1) - 1) // 2
        self.unfold = nn.Unfold(kernel_size=(self.window_size, 1),
                                dilation=(self.window_dilation, 1),
                                stride=(self.window_stride, 1),
                                padding=(self.padding, 0))

    def forward(self, x):
        # Input shape: (N,C,T,V), out: (N,C,T,V*window_size)
        N, C, T, V = x.shape
        x = self.unfold(x)
        # Permute extra channels from window size to the graph dimension; -1 for number of windows
        x = x.view(N, C, self.window_size, -1, V).permute(0,1,3,2,4).contiguous()
        x = x.view(N, C, -1, self.window_size * V)
        return x


def k_adjacency(A, k, with_self=False, self_factor=1):
    assert isinstance(A, np.ndarray)
    I = np.eye(len(A), dtype=A.dtype)
    if k == 0:
        return I
    Ak = np.minimum(np.linalg.matrix_power(A + I, k), 1) \
       - np.minimum(np.linalg.matrix_power(A + I, k - 1), 1)
    if with_self:
        Ak += (self_factor * I)
    return Ak


def k_adjacency_laplacian(A, k, with_self=False, self_factor=1, normalized=True):
    assert isinstance(A, np.ndarray)
    I = np.eye(len(A), dtype=A.dtype)
    if k == 0:
        return I
    Ak = np.minimum(np.linalg.matrix_power(A + I, k), 1) \
       - np.minimum(np.linalg.matrix_power(A + I, k - 1), 1)
    # if with_self:
    #     Ak += (self_factor * I)
    D = Ak.sum(-1)

    degs_inv_sqrt = np.power(D, -0.5)
    norm_degs_matrix = np.eye(len(D)) * degs_inv_sqrt

    Ak = np.eye(len(D)) * (D + 2) - Ak
    if normalized:
        return (norm_degs_matrix @ Ak @ norm_degs_matrix).astype(np.float32)
    return Ak


class SpatialTemporal_MS_GCN(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 A_binary,
                 num_scales,
                 window_size,
                 disentangled_agg=True,
                 use_Ares=True,
                 residual=False,
                 dropout=0,
                 activation='relu'):

        super().__init__()
        self.num_scales = num_scales
        self.window_size = window_size
        self.use_Ares = use_Ares
        A = self.build_spatial_temporal_graph(A_binary, window_size)

        if disentangled_agg:
            # A_scales = [k_adjacency(A, k, with_self=True) for k in range(num_scales)]
            # A_scales = np.concatenate([normalize_adjacency_matrix(g) for g in A_scales])
            A_scales = np.concatenate([k_adjacency_laplacian(A, k) for k in range(num_scales)])

        else:
            # Self-loops have already been included in A
            A_scales = [normalize_adjacency_matrix(A) for k in range(num_scales)]
            A_scales = [np.linalg.matrix_power(g, k) for k, g in enumerate(A_scales)]
            A_scales = np.concatenate(A_scales)

        self.A_scales = torch.Tensor(A_scales)
        self.V = len(A_binary)

        if use_Ares:
            self.A_res = nn.init.uniform_(nn.Parameter(torch.randn(self.A_scales.shape)), -1e-6, 1e-6)
        else:
            self.A_res = torch.tensor(0)

        self.mlp = MLP(in_channels * num_scales, [out_channels], dropout=dropout, activation='linear')

        # Residual connection
        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels):
            self.residual = lambda x: x
        else:
            self.residual = MLP(in_channels, [out_channels], activation='linear')

        # self.act = activation_factory(activation)
        self.act = nn.ReLU()

    def build_spatial_temporal_graph(self, A_binary, window_size):
        assert isinstance(A_binary, np.ndarray), 'A_binary should be of type `np.ndarray`'
        V = len(A_binary)
        V_large = V * window_size
        A_binary_with_I = A_binary + np.eye(len(A_binary), dtype=A_binary.dtype)
        # Build spatial-temporal graph
        A_large = np.tile(A_binary_with_I, (window_size, window_size)).copy()
        return A_large

    def forward(self, x):
        N, C, T, V = x.shape    # T = number of windows

        # Build graphs
        A = self.A_scales.to(x.dtype).to(x.device) + self.A_res.to(x.dtype).to(x.device)

        # Perform Graph Convolution
        res = self.residual(x)
        agg = torch.einsum('vu,nctu->nctv', A, x)
        agg = agg.view(N, C, T, self.num_scales, V)
        agg = agg.permute(0,3,1,2,4).contiguous().view(N, self.num_scales*C, T, V)
        out = self.mlp(agg)
        out += res
        return self.act(out)


class SpatialTemporal_MS_GCN_2s(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 A_binary,
                 num_scales,
                 window_size,
                 disentangled_agg=True,
                 use_Ares=True,
                 residual=False,
                 dropout=0,
                 activation='relu'):

        super().__init__()
        self.num_scales = num_scales
        self.window_size = window_size
        self.use_Ares = use_Ares
        A = self.build_spatial_temporal_graph(A_binary, window_size)

        if disentangled_agg:
            A_scales = [k_adjacency(A, k, with_self=True) for k in range(num_scales)]
            A_scales = np.concatenate([normalize_adjacency_matrix(g) for g in A_scales])

            B_scales = np.concatenate([k_adjacency_laplacian(A, k) for k in range(num_scales)])
        else:
            # Self-loops have already been included in A
            A_scales = [normalize_adjacency_matrix(A) for k in range(num_scales)]
            A_scales = [np.linalg.matrix_power(g, k) for k, g in enumerate(A_scales)]
            A_scales = np.concatenate(A_scales)

        self.A_scales = torch.Tensor(A_scales)
        self.B_scales = torch.Tensor(B_scales)
        self.V = len(A_binary)

        if use_Ares:
            self.A_res = nn.init.uniform_(nn.Parameter(torch.randn(self.A_scales.shape)), -1e-6, 1e-6)
        else:
            self.A_res = torch.tensor(0)

        self.mlp = MLP(in_channels * num_scales * 2, [out_channels], dropout=dropout, activation='linear')

        # Residual connection
        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels):
            self.residual = lambda x: x
        else:
            self.residual = MLP(in_channels, [out_channels], activation='linear')

        # self.act = activation_factory(activation)
        self.act = nn.ReLU()

    def build_spatial_temporal_graph(self, A_binary, window_size):
        assert isinstance(A_binary, np.ndarray), 'A_binary should be of type `np.ndarray`'
        V = len(A_binary)
        V_large = V * window_size
        A_binary_with_I = A_binary + np.eye(len(A_binary), dtype=A_binary.dtype)
        # Build spatial-temporal graph
        A_large = np.tile(A_binary_with_I, (window_size, window_size)).copy()
        return A_large

    def forward(self, x):
        N, C, T, V = x.shape    # T = number of windows

        # Build graphs
        A = self.A_scales.to(x.dtype).to(x.device) + self.A_res.to(x.dtype).to(x.device)
        B = self.B_scales.to(x.dtype).to(x.device) + self.A_res.to(x.dtype).to(x.device)

        # Perform Graph Convolution
        res = self.residual(x)
        agg = torch.einsum('vu,nctu->nctv', A, x)
        agg = agg.view(N, C, T, self.num_scales, V)
        agg = agg.permute(0,3,1,2,4).contiguous().view(N, self.num_scales*C, T, V)

        agg2 = torch.einsum('vu,nctu->nctv', B, x)
        agg2 = agg2.view(N, C, T, self.num_scales, V)
        agg2 = agg2.permute(0,3,1,2,4).contiguous().view(N, self.num_scales*C, T, V)

        agg = torch.cat([agg, agg2], dim=1)

        out = self.mlp(agg)
        out += res
        return self.act(out)


def normalize(A , symmetric=True):
	# A = A+I
	# A = A + torch.eye(A.size(0))
	# 所有节点的度
	d = A.sum(1)
	if symmetric:
		#D = D^-1/2
		D = torch.diag(torch.pow(d , -0.5))
		return D.mm(A).mm(D)
	else :
		# D=D^-1
		D =torch.diag(torch.pow(d,-1))
		return D.mm(A)


class GCN(nn.Module):
    def __init__(self, A, dim_in, dim_out):
        super(GCN, self).__init__()
        self.A = normalize(A.float())
        self.fc1 = nn.Conv2d(dim_in, dim_in, kernel_size=(1, 1), padding=(0, 0))
        self.relu = nn.ReLU()

        self.temporal_conv = nn.Conv2d(dim_in, dim_out, kernel_size=(3, 1), padding=(1, 0))
        self.temporal_relu = nn.ReLU()
        self.temporal_bn = nn.BatchNorm2d(dim_out)

    def forward(self, input):
        B, C, T, J = input.shape
        input = torch.Tensor.permute(input, (0, 2, 1, 3))

        self.A = self.A.to(device=input.device)
        x = input.matmul(self.A).permute((0, 2, 1, 3))
        x = self.fc1(x)
        x = self.relu(x)

        x = self.temporal_conv(x)
        x = self.temporal_relu(x)
        x = self.temporal_bn(x)
        return x

def normalize_outproduct(A , symmetric=True):
    # A = A+I
    # A = A + torch.eye(A.size(0))
    A_withoutI = A - torch.eye(A.size(0))
    d_withoutI = A_withoutI.sum(1)
    A_2 = torch.eye(A.size(0)) * d_withoutI - A_withoutI

    d = A.sum(1)
    if symmetric:
        #D = D^-1/2
        D = torch.diag(torch.pow(d , -0.5))
        return D.mm(A).mm(D), A_2
    else :
        # D=D^-1
        D =torch.diag(torch.pow(d,-1))
        return D.mm(A), A_2


class GCN_OUTPRODUCT(nn.Module):
    def __init__(self, A, dim_in, dim_out, out_level=1):
        super(GCN_OUTPRODUCT, self).__init__()
        self.out_level = out_level
        A, A_2 = normalize_outproduct(A.float())
        self.A = A
        self.A_2 = A_2

        self.fc1 = nn.Conv2d(dim_in, dim_in, kernel_size=(1, 1), padding=(0, 0))
        self.relu1 = nn.ReLU()

        self.fc2 = nn.Conv2d(dim_in, dim_in, kernel_size=(1, 1), padding=(0, 0))
        self.relu2 = nn.ReLU()

        if self.out_level > 0:
            in_channel = 0
            for i in range(out_level):
                in_channel += dim_in ** (i + 2)
            self.fc3 = nn.Conv2d(in_channel, dim_in, kernel_size=(1, 1), padding=(0, 0))
            self.relu3 = nn.ReLU()

            self.temporal_conv = nn.Conv2d(dim_in * 3, dim_out, kernel_size=(3, 1), padding=(1, 0))
        else:
            self.temporal_conv = nn.Conv2d(dim_in * 2, dim_out, kernel_size=(3, 1), padding=(1, 0))

        self.temporal_relu = nn.ReLU()
        self.temporal_bn = nn.BatchNorm2d(dim_out)

    def forward(self, input):
        B, C, T, J = input.shape
        input = torch.Tensor.permute(input, (0, 2, 1, 3))

        self.A = self.A.to(device=input.device)
        x1 = input.matmul(self.A).permute((0, 2, 1, 3))
        x1 = self.fc1(x1)
        x1 = self.relu1(x1)

        self.A_2 = self.A_2.to(device=input.device)
        x2 = input.matmul(self.A_2).permute((0, 2, 1, 3))
        x3 = x2.permute((0, 2, 3, 1))
        x2 = self.fc2(x2)
        x2 = self.relu2(x2)

        if self.out_level > 0:
            tmp = x3
            for i in range(self.out_level):
                left = x3.unsqueeze(dim=4)
                right = tmp.unsqueeze(dim=3)
                x3 = torch.Tensor.reshape(right * left, (x3.shape[0], x3.shape[1], x3.shape[2], -1))

                if i == 0:
                    result = x3
                else:
                    result = torch.cat([result, x3], dim=3)

            x3 = result.permute((0, 3, 1, 2))
            x3 = self.fc3(x3)
            x3 = self.relu3(x3)
            x = torch.cat([x1, x2, x3], dim=1)

        else:
            x = torch.cat([x1, x2], dim=1)

        x = self.temporal_conv(x)
        x = self.temporal_relu(x)
        x = self.temporal_bn(x)
        return x
