import torch

from samesh.data.common import TorchTensor


def discretize(t: TorchTensor, lb=-1, ub=1, resolution=128) -> TorchTensor:
    """
    Given tensor of continuous values, return corresponding discrete logits.
    """
    return torch.round((t - lb) / (ub - lb) * resolution).long()


def undiscretize(t: TorchTensor, lb=-1, ub=1, resolution=128) -> TorchTensor:
    """
    Given tensor of discrete logits, return corresponding continuous values.
    """
    return t.float() / resolution * (ub - lb) + lb


def range_norm(t: TorchTensor, lb=None, ub=None, offset=None, eps=1e-8) -> TorchTensor:
    """
    Given tensor of continuous values, return corresponding range normalized values.
    """
    if lb is None: lb = t.min() - offset if offset else t.min()
    if ub is None: ub = t.max()
    return (t - lb) / (ub - lb + eps)