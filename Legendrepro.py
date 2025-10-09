from typing import Optional
from torch_geometric.typing import OptTensor
import math
import torch
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops,to_dense_adj, degree
from torch_geometric.utils import get_laplacian
from scipy.special import comb
import torch.nn.functional as F
from torch_geometric.nn.conv.gcn_conv import gcn_norm
import numpy as np


from manifolds.base import Manifold
from utils.math_utils import artanh, tanh


class Bern_prop(MessagePassing):
    def __init__(self, K, c, manifold, bias=True, adaptive=True, **kwargs):
        super(Bern_prop, self).__init__(aggr='add', **kwargs)

        self.K = K
        #self.adaptive = adaptive
        #if adaptive:
            # 初始化自适应阶数参数
            #self.alpha = Parameter(torch.Tensor(1))
            #self.beta = Parameter(torch.Tensor(1))
            #self.reset_adaptive_params()
        self.temp = Parameter(torch.Tensor(self.K + 1))
        self.reset_parameters()
        self.c = c
        self.manifold = manifold

    #def reset_adaptive_params(self):
        # 初始化自适应参数
        #self.alpha.data.fill_(1.0)
        #self.beta.data.fill_(0.5)

    def reset_parameters(self):
        self.temp.data.fill_(1)

    #def get_adaptive_K(self, x):
        #if not self.adaptive:
            #return self.K

        # 计算自适应阶数
        # 使用sigmoid确保K在合理范围内
        #adaptive_K = torch.sigmoid(self.alpha) * self.K + torch.sigmoid(self.beta) * 2
        #return int(adaptive_K.item())

    def construct_laplacian(self, edge_indexi, normi, edge_indexii, normii, num_nodes):
        L = torch.sparse_coo_tensor(edge_indexi, normi, (num_nodes, num_nodes))
        L_mod = torch.sparse_coo_tensor(edge_indexii, normii, (num_nodes, num_nodes))
        return L, L_mod

    def forward(self, x, edge_index, edge_weight=None):
        TEMP = F.relu(self.temp)
        num_nodes = x.size(self.node_dim)

        # 获取自适应阶数
        #current_K = self.get_adaptive_K(x)

        # 构造拉普拉斯矩阵
        edge_index1, norm1 = get_laplacian(edge_index, edge_weight, normalization='sym',
                                           dtype=x.dtype, num_nodes=x.size(self.node_dim))
        edge_index2, norm2 = add_self_loops(edge_index1, -norm1, fill_value=2.,
                                            num_nodes=x.size(self.node_dim))
        L, L_mod = self.construct_laplacian(edge_index1, norm1, edge_index2, norm2,
                                            num_nodes=num_nodes)

        # 勒让德多项式展开
        tmp = []
        #x = x.transpose(-1, -2)
        tmp.append(x)

        # 使用当前阶数进行传播
        for i in range(self.K):
            x = self.propagate(edge_index2, x=x, norm=norm2, size=None)
            #x = self.manifold.mobius_matvec(L_mod, x, self.c)
            tmp.append(x)

        # 勒让德多项式求和
        out = (1 / math.factorial(0)) * (1 / (2 ** 0)) * TEMP[0] * tmp[0]

        for i in range(self.K):
            x = tmp[i + 1]
            x = self.propagate(edge_index1, x=x, norm=norm1, size=None)
            #x = self.manifold.mobius_matvec(L, x, self.c)
            for j in range(i):
                #x = self.manifold.mobius_matvec(L, x, self.c)
                x = self.propagate(edge_index1, x=x, norm=norm1, size=None)
            out = out + (-1) ** (i + 1) * (1 / math.factorial(i + 1) *comb(2 * (i + 1), i + 1)) * TEMP[i + 1] * x

        #out = out.transpose(-1, -2)
        return out

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def __repr__(self):
        return '{}(K={}, temp={})'.format(
            self.__class__.__name__, self.K,  self.temp)


