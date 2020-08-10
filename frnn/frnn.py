# create an interface similar to pytorch3d's knn
from collections import namedtuple
from typing import Union

import torch

from frnn import _C
import torch
from torch.autograd import Function
from torch.autograd.function import once_differentiable

# TODO: implement backward pass for frnn and implement this operation as an autograd.Function

def frnn_grid_points(
  points1: torch.Tensor,
  points2: torch.Tensor,
  lengths1: Union[torch.Tensor, None] = None,
  lengths2: Union[torch.Tensor, None] = None,
  sorted_point_idx: Union[torch.Tensor, None] = None,
  grid_off: Union[torch.Tensor, None] = None,
  K: int = 1,
  r: float = -1,
  return_nn: bool = False,
  return_sorted: bool = True,     # for now we always sort the neighbors by dist
  return_grid: bool = False,      # for reusing grid structure
):
  """
  TODO: add docs here
  """

  if points1.shape[0] != points2.shape[0]:
    raise ValueError("points1 and points2 must have the same batch  dimension")
  if points1.shape[2] != 3 or points2.shape[2] != 3:
    raise ValueError("for now only grid in 3D is supported")

  points1 = points1.contiguous()
  points2 = points2.contiguous()

  P1 = points1.shape[1]
  P2 = points1.shape[1]

  if lengths1 is None:
    lengths1 = torch.full((points1.shape[0],), P1, dtype=torch.long, device=points1.device)
  if lengths2 is None:
    lengths2 = torch.full((points2.shape[0],), P2, dtype=torch.long, device=points2.device)
  
  return None