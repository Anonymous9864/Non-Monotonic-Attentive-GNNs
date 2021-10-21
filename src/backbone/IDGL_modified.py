import os
import sys
import torch
import torch_geometric
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter
from torch_sparse import SparseTensor, set_diag
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import softmax
from torch_geometric.typing import Adj, OptTensor
from typing import Optional

sys.path.append(os.path.join(os.path.dirname("__file__"), '..', '..'))
import src.utils as utils
import src.mapping_layer as mp


class M_IDGLLayer(MessagePassing):

    def __init__(self, in_channels: int, out_channels: int, mapping_function: str = 'clean', base_number: int = 2,
                 heads: int = 1, add_self_loops: bool = True, bias: bool = True, normalize: bool = True, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(M_IDGLLayer, self).__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.mapping_function = mapping_function
        self.normalize = normalize
        self.add_self_loops = add_self_loops

        self.weight = Parameter(Tensor(in_channels, out_channels))
        self.att = Parameter(Tensor(1, heads, out_channels))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.alpha = None

        if self.mapping_function != 'sin' and self.mapping_function != 'clean':
            self.mapping_layer = mp.MappingLayer(mapping_func_type=mapping_function, basis_number=base_number)

        self.reset_parameters()

    def reset_parameters(self):
        utils.glorot(self.weight)
        utils.glorot(self.att)
        utils.zeros(self.bias)
        if self.mapping_function != 'sin' and self.mapping_function != 'clean':
            self.mapping_layer.reset_parameters()

    def forward(self, x: Tensor, edge_index: Adj, return_attention_weights=None):
        if self.add_self_loops:
            if isinstance(edge_index, Tensor):
                num_nodes = x.size(0)
                edge_index, _ = torch_geometric.utils.remove_self_loops(edge_index)
                edge_index, _ = torch_geometric.utils.add_self_loops(edge_index, num_nodes=num_nodes)
            elif isinstance(edge_index, SparseTensor):
                edge_index = set_diag(edge_index)

        x = x.matmul(self.weight)

        out = self.propagate(edge_index, x=x)

        if self.bias is not None:
            out += self.bias

        if isinstance(return_attention_weights, bool):
            if isinstance(edge_index, Tensor):
                return out, (edge_index, self.alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(self.alpha, layout='coo')
        else:
            return out

    def message(self, x_i: Tensor, x_j: Tensor, index: Tensor, size_i: Optional[int], ptr: OptTensor) -> Tensor:
        x_i_copy = torch.unsqueeze(x_i, 1)
        x_j_copy = torch.unsqueeze(x_j, 1)
        x_i_copy = x_i_copy * self.att
        x_j_copy = x_j_copy * self.att
        alpha = F.cosine_similarity(x_i_copy, x_j_copy, dim=-1)
        alpha = alpha.mean(-1).view(-1, 1)

        if self.normalize:
            alpha = utils.max_normalize(alpha, index, size_i)

        alpha = softmax(alpha, index, ptr, size_i)
        self.alpha = alpha
        result = x_j * alpha
        return result
