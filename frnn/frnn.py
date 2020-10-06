# create an interface similar to pytorch3d's knn
from collections import namedtuple
from typing import Union

import torch

from frnn import _C
from torch.autograd import Function
from torch.autograd.function import once_differentiable

_GRID = namedtuple("GRID", "sorted_points2 pc2_grid_off sorted_points2_idxs grid_params")

GRID_PARAMS_SIZE = 8
MAX_RES = 100

class _frnn_grid_points(Function):
  """
  Torch autograd Function wrapper for FRNN CUDA implementation
  """

  @staticmethod
  def forward(
      ctx, 
      points1, 
      points2, 
      lengths1, 
      lengths2, 
      K: int,
      r: float,
      # grid: Union[_GRID, None] = None, # autograd.function only supports tensor as input/output
      sorted_points2 = None,
      pc2_grid_off = None,
      sorted_points2_idxs = None,
      grid_params_cuda = None,
      return_sorted: bool = True,
      radius_cell_ratio: float = 2.0,
      # filename: str = None
  ):
    """
    TODO: add docs
    """
    use_cached_grid = sorted_points2 is not None and pc2_grid_off is not None and sorted_points2_idxs is not None and grid_params_cuda is not None
    N = points1.shape[0]
    if not use_cached_grid:
      # create grid from scratch
      # setup grid params
      grid_params_cuda = torch.zeros((N, GRID_PARAMS_SIZE), dtype=torch.float, device=points1.device)
      G = -1
      for i in range(N):
        # 0-2 grid_min; 3 grid_delta; 4-6 grid_res; 7 grid_total
        grid_min = points2[i, :lengths2[i]].min(dim=0)[0]
        grid_max = points2[i, :lengths2[i]].max(dim=0)[0]
        grid_params_cuda[i, :3] = grid_min
        grid_size = grid_max - grid_min
        cell_size = r / radius_cell_ratio
        if cell_size < grid_size.min()/MAX_RES:
          cell_size = grid_size.min() / MAX_RES
        grid_params_cuda[i, 3] = 1 / cell_size
        grid_params_cuda[i, 4:7] = torch.floor(grid_size / cell_size) + 1
        grid_params_cuda[i, 7] = grid_params_cuda[i, 4] * grid_params_cuda[i, 5] * grid_params_cuda[i, 6] 
        if G < grid_params_cuda[i, 7]:
          G = int(grid_params_cuda[i, 7].item())
      # torch.save(grid_params_cuda, "data/grid_params_cuda/"+filename[:-4]+".pt")

      # insert points into the grid
      P2 = points2.shape[1]
      pc2_grid_cnt = torch.zeros((N, G), dtype=torch.int, device=points1.device)
      pc2_grid_cell = torch.full((N, P2), -1, dtype=torch.int, device=points1.device)
      pc2_grid_idx = torch.full((N, P2), -1, dtype=torch.int, device=points1.device)
      _C.insert_points_cuda(points2, lengths2, grid_params_cuda, pc2_grid_cnt, pc2_grid_cell, pc2_grid_idx, G)
      # torch.save(pc2_grid_cnt, "data/pc2_grid_cnt/"+filename[:-4]+".pt")
      # torch.save(pc2_grid_cell, "data/pc2_grid_cell/"+filename[:-4]+".pt")
      # torch.save(pc2_grid_idx, "data/pc2_grid_idx/"+filename[:-4]+".pt")


      # compute the offset for each grid
      pc2_grid_off = _C.prefix_sum_cuda(pc2_grid_cnt, grid_params_cuda.cpu())
      # torch.save(pc2_grid_off, "data/pc2_grid_off/"+filename[:-4]+".pt")

      # sort points according to their grid positions and insertion orders
      sorted_points2 = torch.zeros((N, P2, 3), dtype=torch.float, device=points1.device)
      sorted_points2_idxs = torch.full((N, P2), -1, dtype=torch.int, device=points1.device)
      _C.counting_sort_cuda(
        points2,
        lengths2,
        pc2_grid_cell,
        pc2_grid_idx,
        pc2_grid_off,
        sorted_points2,
        sorted_points2_idxs
      )
      # torch.save(sorted_points2, "data/sorted_points2/"+filename[:-4]+".pt")
      # torch.save(sorted_points2_idxs, "data/sorted_points2_idxs/"+filename[:-4]+".pt")

    assert(sorted_points2 is not None and pc2_grid_off is not None and sorted_points2_idxs is not None and grid_params_cuda is not None)

    G = pc2_grid_off.shape[1]

    # also sort the points1 for faster search
    P1 = points1.shape[1]
    pc1_grid_cnt = torch.zeros((N, G), dtype=torch.int, device=points1.device)
    pc1_grid_cell = torch.full((N, P1), -1, dtype=torch.int, device=points1.device)
    pc1_grid_idx = torch.full((N, P1), -1, dtype=torch.int, device=points1.device)
    _C.insert_points_cuda(points1, lengths1, grid_params_cuda, pc1_grid_cnt, pc1_grid_cell, pc1_grid_idx, G)
    # torch.save(pc1_grid_cnt, "data/pc1_grid_cnt/"+filename[:-4]+".pt")
    # torch.save(pc1_grid_cell, "data/pc1_grid_cell/"+filename[:-4]+".pt")
    # torch.save(pc1_grid_idx, "data/pc1_grid_idx/"+filename[:-4]+".pt")

    pc1_grid_off = _C.prefix_sum_cuda(pc1_grid_cnt, grid_params_cuda.cpu())
    # torch.save(pc1_grid_off, "data/pc1_grid_off/"+filename[:-4]+".pt")

    # print("last offset for pc1: ", pc1_grid_off[1, int(grid_params_cuda[1, 7].item())-2])
    
    sorted_points1 = torch.zeros((N, P1, 3), dtype=torch.float, device=points1.device)
    sorted_points1_idxs = torch.full((N, P1), -1, dtype=torch.int, device=points1.device)
    _C.counting_sort_cuda(
      points1,
      lengths1,
      pc1_grid_cell,
      pc1_grid_idx,
      pc1_grid_off,
      sorted_points1,
      sorted_points1_idxs
    )
    # torch.save(sorted_points1, "data/sorted_points1/"+filename[:-4]+".pt")
    # torch.save(sorted_points1_idxs, "data/sorted_points1_idxs/"+filename[:-4]+".pt")
    # print("sorted idxs max: ", sorted_points1_idxs[1, :lengths1[1]].max())
    # print("sorted idxs max: ", sorted_points1_idxs[1].max())
    # print("unique idxs: ", torch.unique(sorted_points1_idxs[1, :lengths1[1]]).shape[0])


      # cache the grid structure for reusing 
      # grid = _GRID(
      #     sorted_points=sorted_points2, # (N, P , 3) 
      #     grid_off=grid_off,  # (N, G)
      #     sorted_points_idxs=sorted_points2_idxs,  # (N, P)
      #     grid_params=grid_params_cuda) #(N, 8)

    #  # perform search on the grid
    #  idxs, dists = _C.find_nbrs_cuda(
    #    points1,
    #    sorted_points2,
    #    lengths1,
    #    lengths2,
    #    grid_off,
    #    sorted_points2_idxs,
    #    grid_params_cuda,
    #    K,
    #    r
    #  )
    #else:
    #  # use cached grid to search 
    #  idxs, dists = _C.find_nbrs_cuda(
    #    points1,
    #    sorted_points2,
    #    lengths1,
    #    lengths2,
    #    grid_off,
    #    sorted_points2_idxs,
    #    grid_params_cuda,
    #    K,
    #    r
    #  )
    # perform search on the grid
    idxs, dists = _C.find_nbrs_cuda(
      sorted_points1,
      sorted_points2,
      lengths1,
      lengths2,
      pc2_grid_off,
      sorted_points1_idxs,
      sorted_points2_idxs,
      grid_params_cuda,
      K,
      r
    )
    
    # TODO: compare which is faster: sort here or inside kernel function
    # if K > 1 and return_sorted:
    #   mask = dists < 0
    #   dists[mask] = float("inf")
    #   dists, sort_idxs = dists.sort(dim=2)
    #   dists[mask] = -1
    #   idxs = idxs.gather(2, sort_idxs)
      
    ctx.save_for_backward(points1, points2, lengths1, lengths2, idxs)
    ctx.mark_non_differentiable(idxs)
    ctx.mark_non_differentiable(sorted_points2)
    ctx.mark_non_differentiable(pc2_grid_off)
    ctx.mark_non_differentiable(sorted_points2_idxs)
    ctx.mark_non_differentiable(grid_params_cuda)

    return idxs, dists, sorted_points2, pc2_grid_off, sorted_points2_idxs, grid_params_cuda


  @staticmethod
  @once_differentiable
  def backward(ctx, grad_idxs, grad_dists, grad_sorted_points2, grad_pc2_grid_off,
               grad_sorted_points2_idxs, grad_grid_params_cuda):
    points1, points2, lengths1, lengths2, idxs = ctx.saved_tensors
    grad_points1, grad_points2 = _C.frnn_backward_cuda(
      points1, points2, lengths1, lengths2, idxs, grad_dists
    )
    return grad_points1, grad_points2, None, None, None, None, None, None, None, None, None, None, None


def frnn_grid_points(
  points1: torch.Tensor,
  points2: torch.Tensor,
  lengths1: Union[torch.Tensor, None] = None,
  lengths2: Union[torch.Tensor, None] = None,
  K: int = -1,
  r: float = -1,
  grid: Union[_GRID, None] = None,
  return_nn: bool = False,
  return_sorted: bool = True,     # for now we always sort the neighbors by dist
  radius_cell_ratio: float = 2.0,
  # filename: str = None,
  # return_grid: bool = False,      # for reusing grid structure
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
  P2 = points2.shape[1]
  # print(P1, P2)

  if lengths1 is None:
    lengths1 = torch.full((points1.shape[0],), P1, dtype=torch.long, device=points1.device)
  if lengths2 is None:
    lengths2 = torch.full((points2.shape[0],), P2, dtype=torch.long, device=points2.device)

  if grid is not None: 
    idxs, dists, sorted_points2, pc2_grid_off, sorted_points2_idxs, grid_params_cuda = _frnn_grid_points.apply(
      points1, points2, lengths1, lengths2, K, r, grid[0], grid[1], grid[2], grid[3], return_sorted, radius_cell_ratio
    )
  else:
    idxs, dists, sorted_points2, pc2_grid_off, sorted_points2_idxs, grid_params_cuda = _frnn_grid_points.apply(
      points1, points2, lengths1, lengths2, K, r, None, None, None, None, return_sorted, radius_cell_ratio # filename
    )

  grid = _GRID(
      sorted_points2=sorted_points2, # (N, P , 3) 
      pc2_grid_off=pc2_grid_off,  # (N, G)
      sorted_points2_idxs=sorted_points2_idxs,  # (N, P)
      grid_params=grid_params_cuda) #(N, 8)

  points2_nn = None
  if return_nn:
    points2_nn = frnn_gather(points2, idxs, lengths2)
  
  # follow pytorch3d.ops.knn_points' conventions to return dists frist
  # TODO: also change this in the c++/cuda code?  
  return dists, idxs, points2_nn, grid

# TODO: probably do this in cuda?
def frnn_gather(
  x: torch.Tensor,
  idxs: torch.Tensor,
  lengths: Union[torch.Tensor, None] = None
):
  """
  TODO: add docs
  """
  N, P2, D = x.shape     # M: number of points 
  _N, P1, K = idxs.shape

  if N != _N:
    raise ValueError("x and idxs must have same batch dimension")

  if lengths is None:
    lengths = torch.full((N,), P2, dtype=torch.long, device=x.device)

  # invalid indices are marked with -1
  # for broadcasting we set it to zero temporarily
  tmp_idxs = idxs.clone().detach()
  tmp_idxs[idxs < 0] = 0

  # N x P1 x K x D
  idxs_expanded = tmp_idxs[:, :, :, None].expand(-1, -1, -1, D)

  x_out = x[:, :, None].expand(-1, -1, K, -1).gather(1, idxs_expanded)

  mask = (idxs < 0)[:, :, :, None].expand(-1, -1, -1, D)

  x_out[mask] = 0.0

  return x_out



# def frnn_grid_points_with_timing(
#   points1: torch.Tensor,
#   points2: torch.Tensor,
#   lengths1: Union[torch.Tensor, None] = None,
#   lengths2: Union[torch.Tensor, None] = None,
#   grid: Union[_GRID, None] = None,
#   # sorted_points2: Union[torch.Tensor, None] = None,
#   # sorted_points2_idxs: Union[torch.Tensor, None] = None,
#   # grid_off: Union[torch.Tensor, None] = None,
#   K: int = -1,
#   r: float = -1,
#   return_nn: bool = False,
#   return_sorted: bool = True,     # for now we always sort the neighbors by dist
#   return_grid: bool = False,      # for reusing grid structure
# ):
#   """
#   TODO: add docs here
#   """
# 
#   if points1.shape[0] != points2.shape[0]:
#     raise ValueError("points1 and points2 must have the same batch  dimension")
#   if points1.shape[2] != 3 or points2.shape[2] != 3:
#     raise ValueError("for now only grid in 3D is supported")
#   if not points1.is_cuda or not points2.is_cuda:
#     raise TypeError("for now only cuda version is supported")
# 
#   points1 = points1.contiguous()
#   points2 = points2.contiguous()
# 
#   P1 = points1.shape[1]
#   P2 = points2.shape[1]
#   # print(P1, P2)
# 
#   if lengths1 is None:
#     lengths1 = torch.full((points1.shape[0],), P1, dtype=torch.long, device=points1.device)
#   if lengths2 is None:
#     lengths2 = torch.full((points2.shape[0],), P2, dtype=torch.long, device=points2.device)
# 
#   if grid is None:
#     # setup grid params
# 
# 
#     N = points1.shape[0]
#     grid_params_cuda = torch.zeros((N, GRID_PARAMS_SIZE), dtype=torch.float, device=points1.device)
# 
#     G = -1
#     for i in range(N):
#       # 0-2 grid_min; 3 grid_delta; 4-6 grid_res; 7 grid_total
#       start = torch.cuda.Event(enable_timing=True)
#       end = torch.cuda.Event(enable_timing=True)
#       start.record()
#       grid_min = points2[i, :lengths2[i]].min(dim=0)[0]
#       grid_max = points2[i, :lengths2[i]].max(dim=0)[0]
#       end.record()
#       torch.cuda.synchronize()
#       setup_time = start.elapsed_time(end)
#       grid_params_cuda[i, :3] = grid_min
#       grid_size = grid_max - grid_min
#       cell_size = r
#       if cell_size < grid_size.min()/MAX_RES:
#         cell_size = grid_size.min() / MAX_RES
#       grid_params_cuda[i, 3] = 1 / cell_size
#       grid_params_cuda[i, 4:7] = torch.floor(grid_size / cell_size) + 1
#       grid_params_cuda[i, 7] = grid_params_cuda[i, 4] * grid_params_cuda[i, 5] * grid_params_cuda[i, 6] 
#       if G < grid_params_cuda[i, 7]:
#         G = int(grid_params_cuda[i, 7].item())
#         # print(G)
# 
# 
#     # test setup_grid_params  
#     # print("Grid Params:\n", grid_params_cuda)
#     start = torch.cuda.Event(enable_timing=True)
#     end = torch.cuda.Event(enable_timing=True)
#     start.record()
# 
#     grid_cnt = torch.zeros((N, G), dtype=torch.int, device=points1.device)
#     grid_cell = torch.full((N, P2), -1, dtype=torch.int, device=points1.device)
#     grid_idx = torch.full((N, P2), -1, dtype=torch.int, device=points1.device)
# 
#     _C.insert_points_cuda(points2, lengths2, grid_params_cuda, grid_cnt, grid_cell, grid_idx, G)
# 
#     end.record()
#     torch.cuda.synchronize()
#     insert_points_time = start.elapsed_time(end)
# 
#     # test insert_points
#     # return grid_cnt, grid_cell, grid_idx
# 
#     start = torch.cuda.Event(enable_timing=True)
#     end = torch.cuda.Event(enable_timing=True)
#     start.record()
#     grid_off = _C.prefix_sum_cuda(grid_cnt, grid_params_cuda.cpu())
#     # test_prefix_sum
#     # return grid_off
#     # test_counting_sort (need to output grid_idx for comparison)
#     # return grid_off, grid_cnt, grid_cell, grid_idx
#     end.record()
#     torch.cuda.synchronize()
#     prefix_sum_time = start.elapsed_time(end)
# 
#     start = torch.cuda.Event(enable_timing=True)
#     end = torch.cuda.Event(enable_timing=True)
#     start.record()
#     sorted_points2 = torch.zeros((N, P2, 3), dtype=torch.float, device=points1.device)
#     sorted_points2_idxs = torch.full((N, P2), -1, dtype=torch.int, device=points1.device)
# 
#     _C.counting_sort_cuda(
#       points2,
#       lengths2,
#       grid_cell,
#       grid_idx,
#       grid_off,
#       sorted_points2,
#       sorted_points2_idxs
#     )
#     end.record()
#     torch.cuda.synchronize()
#     counting_sort_time = start.elapsed_time(end)
# 
#     start = torch.cuda.Event(enable_timing=True)
#     end = torch.cuda.Event(enable_timing=True)
#     start.record()
#     idxs, dists = _C.find_nbrs_cuda(
#       points1,
#       sorted_points2,
#       lengths1,
#       lengths2,
#       grid_off,
#       sorted_points2_idxs,
#       grid_params_cuda,
#       K,
#       r
#     )
#     end.record()
#     torch.cuda.synchronize()
#     find_nbrs_time = start.elapsed_time(end)
#   else:
#     idxs, dists = _C.find_nbrs_cuda(
#       points1,
#       grid[0],        # sorted_points2
#       lengths1,
#       lengths2,
#       grid[1],        # grid_off
#       grid[2],        # sorted_points2_idxs
#       grid[3],        # grid_params_cuda
#       K,
#       r
#     )
#     # use cached grid
#     
#   # for now we don't gather here
#   nn = None
# 
#   if return_grid:
#     if grid is not None:
#       return idxs, dists, nn, grid
#     else:
#       return idxs, dists, nn, \
#         _GRID(
#           sorted_points=sorted_points2, 
#           grid_off=grid_off, 
#           sorted_points_idxs=sorted_points2_idxs, 
#           grid_params=grid_params_cuda)
#   else:
#     return idxs, dists, nn, None, setup_time, insert_points_time, \
#       prefix_sum_time, counting_sort_time, find_nbrs_time
