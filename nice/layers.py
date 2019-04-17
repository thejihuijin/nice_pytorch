"""
Implementation of NICE bijective triangular-jacobian layers.
"""
import torch
import torch.nn as nn
import numpy as np


# ===== ===== Coupling Layer Implementations ===== =====

_get_even = lambda xs: xs[:,0::2]
_get_odd = lambda xs: xs[:,1::2]

def _interleave(first, second, order):
    """
    Given 2 rank-2 tensors with same batch dimension, interleave their columns.
    
    The tensors "first" and "second" are assumed to be of shape (B,M) and (B,N)
    where M = N or N+1, repsectively.
    """
    cols = []
    if order == 'even':
        for k in range(second.shape[1]):
            cols.append(first[:,k])
            cols.append(second[:,k])
        if first.shape[1] > second.shape[1]:
            cols.append(first[:,-1])
    else:
        for k in range(first.shape[1]):
            cols.append(second[:,k])
            cols.append(first[:,k])
        if second.shape[1] > first.shape[1]:
            cols.append(second[:,-1])
    return torch.stack(cols, dim=1)

def _build_relu_network(latent_dim, hidden_dim, num_layers, norm):
    """Helper function to construct a ReLU network of varying number of layers."""
    _modules = [ nn.Linear(latent_dim, hidden_dim) ]
    for _ in range(num_layers):
        _modules.append( nn.Linear(hidden_dim, hidden_dim) )
        _modules.append( nn.ReLU() )
        if norm:
            _modules.append( nn.BatchNorm1d(hidden_dim) )
    _modules.append( nn.Linear(hidden_dim, latent_dim) )
    return nn.Sequential( *_modules )

class DoubleAffineLayer(nn.Module):
    def __init__(self, dim, partition, hidden_dim, num_layers, norm):
        """
        Base coupling layer that handles the permutation of the inputs and wraps
        an instance of torch.nn.Module.

        Usage:
        >> layer = AdditiveCouplingLayer(1000, 'even', nn.Sequential(...))
        
        Args:
        * dim: dimension of the inputs.
        * partition: str, 'even' or 'odd'. If 'even', the even-valued columns are sent to
        pass through the activation module.
        * nonlinearity: an instance of torch.nn.Module.
        """
        super(DoubleAffineLayer, self).__init__()
        # store input dimension of incoming values:
        self.dim = dim
        # store partition choice and make shorthands for 1st and second partitions:
        assert (partition in ['even', 'odd']), "[_BaseCouplingLayer] Partition type must be `even` or `odd`!"
        self.partition = partition
        if (partition == 'even'):
            self._first = _get_even
            self._second = _get_odd
        else:
            self._first = _get_odd
            self._second = _get_even
        # store nonlinear function module:
        # (n.b. this can be a complex instance of torch.nn.Module, for ex. a deep ReLU network)
        #self.add_module('nonlinearity', nonlinearity)
        # Define Networks
        latent_dim = int(dim/2)
        self.s1 = _build_relu_network(latent_dim, hidden_dim, num_layers, norm)
        self.t1 = _build_relu_network(latent_dim, hidden_dim, num_layers, norm)
        self.s2 = _build_relu_network(latent_dim, hidden_dim, num_layers, norm)
        self.t2 = _build_relu_network(latent_dim, hidden_dim, num_layers, norm)
        
    def forward(self, x):
        """Map an input through the partition and nonlinearity."""
        u1 = self._first(x)
        u2 = self._second(x)
        
        v1 = u1*self.s2(u2).exp()+self.t2(u2)
        v2 = u2*self.s1(v1).exp()+self.t1(v1)

        return _interleave(
            v1,
            v2,
            self.partition
        )

    def inverse(self, y):
        """Inverse mapping through the layer. Gradients should be turned off for this pass."""
        v1 = self._first(y)
        v2 = self._second(y)
        u2 = (v2-self.t1(v1))*self.s1(v1).mul(-1).exp()
        u1 = (v1-self.t2(u2))*self.s2(u2).mul(-1).exp()
        
        return _interleave(
            u1,
            u2,
            self.partition
        )


class _BaseCouplingLayer(nn.Module):
    def __init__(self, dim, partition, nonlinearity):
        """
        Base coupling layer that handles the permutation of the inputs and wraps
        an instance of torch.nn.Module.

        Usage:
        >> layer = AdditiveCouplingLayer(1000, 'even', nn.Sequential(...))
        
        Args:
        * dim: dimension of the inputs.
        * partition: str, 'even' or 'odd'. If 'even', the even-valued columns are sent to
        pass through the activation module.
        * nonlinearity: an instance of torch.nn.Module.
        """
        super(_BaseCouplingLayer, self).__init__()
        # store input dimension of incoming values:
        self.dim = dim
        # store partition choice and make shorthands for 1st and second partitions:
        assert (partition in ['even', 'odd']), "[_BaseCouplingLayer] Partition type must be `even` or `odd`!"
        self.partition = partition
        if (partition == 'even'):
            self._first = _get_even
            self._second = _get_odd
        else:
            self._first = _get_odd
            self._second = _get_even
        # store nonlinear function module:
        # (n.b. this can be a complex instance of torch.nn.Module, for ex. a deep ReLU network)
        self.add_module('nonlinearity', nonlinearity)

    def forward(self, x):
        """Map an input through the partition and nonlinearity."""
        return _interleave(
            self._first(x),
            self.coupling_law(self._second(x), self.nonlinearity(self._first(x))),
            self.partition
        )

    def inverse(self, y):
        """Inverse mapping through the layer. Gradients should be turned off for this pass."""
        return _interleave(
            self._first(y),
            self.anticoupling_law(self._second(y), self.nonlinearity(self._first(y))),
            self.partition
        )

    def coupling_law(self, a, b):
        # (a,b) --> g(a,b)
        raise NotImplementedError("[_BaseCouplingLayer] Don't call abstract base layer!")

    def anticoupling_law(self, a, b):
        # (a,b) --> g^{-1}(a,b)
        raise NotImplementedError("[_BaseCouplingLayer] Don't call abstract base layer!")


class AdditiveCouplingLayer(_BaseCouplingLayer):
    """Layer with coupling law g(a;b) := a + b."""
    def coupling_law(self, a, b):
        return (a + b)
    def anticoupling_law(self, a, b):
        return (a - b)


class MultiplicativeCouplingLayer(_BaseCouplingLayer):
    """Layer with coupling law g(a;b) := a .* b."""
    def coupling_law(self, a, b):
        return torch.mul(a,b)
    def anticoupling_law(self, a, b):
        return torch.mul(a, torch.reciprocal(b))


class AffineCouplingLayer(_BaseCouplingLayer):
    """Layer with coupling law g(a;b) := a .* b1 + b2, where (b1,b2) is a partition of b."""
    def coupling_law(self, a, b):
        return torch.mul(a, self._first(b)) + self._second(b)
    def anticoupling_law(self, a, b):
        # TODO
        raise NotImplementedError("TODO: AffineCouplingLayer (sorry!)")
        
