import torch
from torch import nn

def weight(dim_in, dim_out, factorize_k = None):
    if factorize_k is None:
        return nn.Linear(dim_in, dim_out, bias = False)

    assert factorize_k < dim_in and factorize_k < dim_out, 'k must be of relative lower rank'

    return nn.Sequential(
        nn.Linear(dim_in, factorize_k, bias = False),
        nn.Linear(factorize_k, dim_out, bias = False)
    )

class Mogrifier(nn.Module):
    def __init__(self, dim, iters = 5, factorize_k = None):
        super().__init__()
        self.dim = dim
        self.weights = nn.ModuleList([weight(dim, dim, factorize_k) for _ in range(iters)])

    def forward(self, x, h):
        shape = x.shape
        *_, dim = shape
        assert dim == self.dim, f'mogrifier accepts a dimension of {self.dim}'

        x, h = map(lambda t: t.reshape(-1, dim), (x, h))

        for ind, W in enumerate(self.weights):
            if (ind % 2) == 0:
                x = 2 * W(h).sigmoid() * x
            else:
                h = 2 * W(x).sigmoid() * h

        x, h = map(lambda t: t.reshape(*shape), (x, h))
        return x, h
