import math
import torch
from torch_scatter import scatter
from torch import Tensor
from typing import Optional


def glorot(tensor):
    if tensor is not None:
        stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
        tensor.data.uniform_(-stdv, stdv)


def zeros(tensor):
    if tensor is not None:
        tensor.data.fill_(0)


def ones(tensor):
    if tensor is not None:
        tensor.data.fill_(1)


def rand_zero_to_ones(tensor):
    if tensor is not None:
        tensor.data.uniform_(0, 1)


def diag_ones(tensor):
    if tensor is not None:
        tensor.data.fill_(0)
        for i in range(tensor.size(1)):
            tensor.data[i + 1][i] = 1


def simple_normalize(src: Tensor, index: Tensor, num_nodes: Optional[int] = None) -> Tensor:
    N = maybe_num_nodes(index, num_nodes)
    out = scatter(src, index, dim=0, dim_size=N, reduce='min')[index]
    out[out >= 0] = 0
    out = src + torch.abs(out)
    out_sum = scatter(out, index, dim=0, dim_size=N, reduce='sum')[index]
    return out / (out_sum + 1e-16)


def softmax_normalize(src: Tensor, index: Tensor, num_nodes: Optional[int] = None) -> Tensor:
    N = maybe_num_nodes(index, num_nodes)
    out = src - scatter(src, index, dim=0, dim_size=N, reduce='max')[index]
    out = out.exp()
    out_sum = scatter(out, index, dim=0, dim_size=N, reduce='sum')[index]
    return out / (out_sum + 1e-16)


def max_normalize(src: Tensor, index: Tensor, num_nodes: Optional[int] = None) -> Tensor:
    N = maybe_num_nodes(index, num_nodes)
    out = scatter(src, index, dim=0, dim_size=N, reduce='min')[index]
    out[out >= 0] = 0
    out = src + torch.abs(out)
    out_max = scatter(out, index, dim=0, dim_size=N, reduce='max')[index]
    return out / (out_max + 1e-16)


def maybe_num_nodes(edge_index, num_nodes=None):
    if num_nodes is not None:
        return num_nodes
    elif isinstance(edge_index, Tensor):
        return int(edge_index.max()) + 1 if edge_index.numel() > 0 else 0
    else:
        return max(edge_index.size(0), edge_index.size(1))
