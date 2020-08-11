# create an interface similar to pytorch3d's knn
from collections import namedtuple
from typing import Union

import torch

from frnn import _C
import torch
from torch.autograd import Function
from torch.autograd.function import once_differentiable

# TODO: implement backward pass for frnn and implement this operation as an autograd.Function

GRID_PARAMS_SIZE = 8
MAX_RES = 100

def frnn_grid_points(
  points1: torch.Tensor,
  points2: torch.Tensor,
  lengths1: Union[torch.Tensor, None] = None,
  lengths2: Union[torch.Tensor, None] = None,
  sorted_points2: Union[torch.Tensor, None] = None,
  sorted_point_idx: Union[torch.Tensor, None] = None,
  grid_off: Union[torch.Tensor, None] = None,
  K: int = -1,
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
  if not points1.is_cuda or not points2.is_cuda:
    raise TypeError("for now only cuda version is supported")

  points1 = points1.contiguous()
  points2 = points2.contiguous()

  P1 = points1.shape[1]
  P2 = points1.shape[1]

  if lengths1 is None:
    lengths1 = torch.full((points1.shape[0],), P1, dtype=torch.long, device=points1.device)
  if lengths2 is None:
    lengths2 = torch.full((points2.shape[0],), P2, dtype=torch.long, device=points2.device)
  
  # setup grid params
  N = points1.shape[0]
  grid_params_cuda = torch.zeros((N, GRID_PARAMS_SIZE), dtype=torch.float, device=points1.device)

  for i in range(N):
    # 0-2 grid_min; 3 grid_delta; 4-6 grid_res; 7 grid_total
    grid_min = points2[i, :lengths2[i]].min(dim=0)[0]
    grid_max = points2[i, :lengths2[i]].max(dim=0)[0]
    grid_params_cuda[i, :3] = grid_min
    grid_size = grid_max - grid_min
    cell_size = r
    if cell_size < grid_size.min()/MAX_RES:
      cell_size = grid_size.min() / MAX_RES
    grid_params_cuda[i, 3] = 1 / cell_size
    grid_params_cuda[i, 4:7] = torch.floor(grid_size / cell_size) + 1
    grid_params_cuda[i, 7] = grid_params_cuda[i, 4] * grid_params_cuda[i, 5] * grid_params_cuda[i, 6] 
  
  print(grid_params_cuda)

  return


