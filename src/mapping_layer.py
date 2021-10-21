import os
import sys
import math
import torch
from torch import nn
from torch import Tensor
from torch.nn import Parameter, ParameterList

sys.path.append(os.path.join(os.path.dirname("__file__"), '..'))
import src.utils as utils


class MappingLayer(nn.Module):
    def __init__(self, mapping_func_type: str = 'constant', basis_number: int = 1, heads: int = 1):
        super(MappingLayer, self).__init__()
        self.mapping_func_type = mapping_func_type

        if mapping_func_type == 'polynomial':
            self.mapping_func = PolynomialMapping(basis_number, heads=heads)
        elif mapping_func_type == 'fourier':
            self.mapping_func = FourierMapping(basis_number, heads=heads)
        elif mapping_func_type == 'radial':
            self.mapping_func = GaussianMapping(basis_number, heads=heads)
        else:
            raise Exception('The mapping function has not been realized.')

        self.mapping_func.reset_parameters()

    def forward(self, x: Tensor) -> Tensor:
        return self.mapping_func(x)

    def reset_parameters(self):
        self.mapping_func.reset_parameters()


class MappingFunction(nn.Module):
    def __init__(self, **kwargs):
        super(MappingFunction, self).__init__()

    def params(self) -> ParameterList:
        raise NotImplementedError

    def forward(self, input_tensor: Tensor) -> Tensor:
        """
        Args:
            input_tensor: tensor of size [E, 1]
        """
        raise NotImplementedError

    def reset_parameters(self):
        pass

    def params(self) -> ParameterList:
        pass


class PolynomialMapping(MappingFunction):
    def __init__(self, basis_number: int, heads: int = 1):
        super(PolynomialMapping, self).__init__()
        self.basis_number = basis_number
        self.coefficient = Parameter(torch.Tensor(heads, basis_number))
        self.heads = heads

    def params(self) -> ParameterList:
        return ParameterList([self.coefficient])

    def forward(self, input_tensor: Tensor) -> Tensor:
        input_tensor = input_tensor.reshape(-1, self.heads, 1)
        x = torch.cat([input_tensor ** i for i in range(self.basis_number)], -1)
        result = x.mul(self.coefficient)
        result = result.sum(dim=-1)
        return result

    def reset_parameters(self):
        utils.glorot(self.coefficient)


class FourierMapping(MappingFunction):
    def __init__(self, basis_number: int, heads: int = 1):
        super(FourierMapping, self).__init__()
        self.basis_number = basis_number
        self.cos_coefficient = Parameter(torch.Tensor(heads, basis_number))
        self.sin_coefficient = Parameter(torch.Tensor(heads, basis_number))
        self.heads = heads

    def params(self) -> ParameterList:
        return ParameterList([self.cos_coefficient, self.sin_coefficient])

    def forward(self, input_tensor: Tensor) -> Tensor:
        input_tensor = input_tensor.reshape(-1, self.heads, 1)
        x = torch.cat([input_tensor * 2 * i * math.pi for i in range(self.basis_number)], -1)
        result = torch.sin(x).mul(self.sin_coefficient) + torch.cos(x).mul(self.cos_coefficient)
        result = result.sum(dim=-1)
        return result

    def reset_parameters(self):
        utils.rand_zero_to_ones(self.cos_coefficient)
        utils.rand_zero_to_ones(self.sin_coefficient)


class GaussianMapping(MappingFunction):
    def __init__(self, basis_number: int, heads: int = 1):
        super(GaussianMapping, self).__init__()
        self.basis_number = basis_number
        self.alpha = Parameter(torch.Tensor(basis_number, heads))
        self.beta = Parameter(torch.Tensor(basis_number, heads))
        self.center = Parameter(torch.Tensor(basis_number, heads))
        self.heads = heads

    def params(self) -> ParameterList:
        return ParameterList([self.alpha, self.beta, self.center])

    def forward(self, input_tensor: Tensor) -> Tensor:
        input_tensor = input_tensor.reshape(-1, 1, self.heads)
        x = torch.cat([input_tensor - self.center[i] for i in range(self.basis_number)], -2)
        x = x ** 2
        x = -0.5 * x / (self.beta ** 2)
        x = x.exp()
        result = x.mul(self.alpha).sum(-2)
        return result

    def reset_parameters(self):
        utils.glorot(self.alpha)
        utils.ones(self.beta)
        utils.rand_zero_to_ones(self.center)
