import os
import sys
from typing import Union, Tuple, Optional
from torch_geometric.typing import (OptPairTensor, Adj, Size, OptTensor)

import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn import Parameter
from torch_sparse import SparseTensor, set_diag
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax

sys.path.append(os.path.join(os.path.dirname("__file__"), '..', '..'))
import src.utils as utils
import src.mapping_layer as mp


class M_GATLayer(MessagePassing):
    _alpha: OptTensor

    def __init__(self, in_channels: Union[int, Tuple[int, int]], out_channels: int,
                 mapping_function: str = 'clean', base_number: int = 2, heads: int = 1, concat: bool = True,
                 negative_slope: float = 0.2, dropout: float = 0.0, add_self_loops: bool = True, bias: bool = True,
                 normalize: bool = True, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(M_GATLayer, self).__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = add_self_loops
        self.mapping_function = mapping_function
        self.normalize = normalize

        if self.mapping_function != 'sin' and self.mapping_function != 'clean':
            self.mapping_layer = mp.MappingLayer(mapping_func_type=mapping_function, basis_number=base_number,
                                                 heads=heads)

        if isinstance(in_channels, int):
            self.lin_src = Linear(in_channels, heads * out_channels, bias=False, weight_initializer='glorot')
            self.lin_dst = self.lin_src
        else:
            self.lin_src = Linear(in_channels[0], heads * out_channels, False, weight_initializer='glorot')
            self.lin_dst = Linear(in_channels[1], heads * out_channels, False, weight_initializer='glorot')

        self.att_src = Parameter(torch.Tensor(1, heads, out_channels))
        self.att_dst = Parameter(torch.Tensor(1, heads, out_channels))

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self._alpha = None

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_src.reset_parameters()
        self.lin_dst.reset_parameters()
        utils.glorot(self.att_src)
        utils.glorot(self.att_dst)
        utils.zeros(self.bias)
        if self.mapping_function != 'sin' and self.mapping_function != 'clean':
            self.mapping_layer.reset_parameters()

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj, size: Size = None,
                return_attention_weights=None):
        H, C = self.heads, self.out_channels

        if isinstance(x, Tensor):
            assert x.dim() == 2, "Static graphs not supported in 'GATConv'"
            x_src = x_dst = self.lin_src(x).view(-1, H, C)
        else:  # Tuple of source and target node features:
            x_src, x_dst = x
            assert x_src.dim() == 2, "Static graphs not supported in 'GATConv'"
            x_src = self.lin_src(x_src).view(-1, H, C)
            if x_dst is not None:
                x_dst = self.lin_dst(x_dst).view(-1, H, C)

        x = (x_src, x_dst)

        alpha_src = (x_src * self.att_src).sum(dim=-1)
        alpha_dst = None if x_dst is None else (x_dst * self.att_dst).sum(-1)
        alpha = (alpha_src, alpha_dst)

        if self.add_self_loops:
            if isinstance(edge_index, Tensor):
                num_nodes = x_src.size(0)
                if x_dst is not None:
                    num_nodes = min(num_nodes, x_dst.size(0))
                num_nodes = min(size) if size is not None else num_nodes
                edge_index, _ = remove_self_loops(edge_index)
                edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
            elif isinstance(edge_index, SparseTensor):
                edge_index = set_diag(edge_index)

        out = self.propagate(edge_index, x=x, alpha=alpha, size=size)

        alpha = self._alpha
        assert alpha is not None
        self._alpha = None

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out += self.bias

        if isinstance(return_attention_weights, bool):
            if isinstance(edge_index, Tensor):
                return out, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout='coo')
        else:
            return out

    def message(self, x_j: Tensor, alpha_j: Tensor, alpha_i: OptTensor,
                index: Tensor, ptr: OptTensor, size_i: Optional[int]) -> Tensor:
        alpha = alpha_j if alpha_i is None else alpha_j + alpha_i

        alpha = F.leaky_relu(alpha, self.negative_slope)

        if self.normalize:
            alpha = utils.max_normalize(alpha, index, size_i)

        if self.mapping_function == 'sin':
            alpha = torch.sin(alpha)
        elif self.mapping_function != 'clean':
            alpha = self.mapping_layer(alpha)

        alpha = softmax(alpha, index, ptr, size_i)
        self._alpha = alpha
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return x_j * alpha.unsqueeze(-1)

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)
