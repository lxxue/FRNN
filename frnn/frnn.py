# create an interface similar to pytorch3d's knn
from collections import namedtuple
from typing import Union

import torch

from frnn import _C
from torch.autograd import Function
from torch.autograd.function import once_differentiable

from prefix_sum import prefix_sum_cuda

_GRID = namedtuple(
    "GRID", "sorted_points2 pc2_grid_off sorted_points2_idxs grid_params")


class _frnn_grid_points(Function):
    """
    Torch autograd Function wrapper for FRNN CUDA implementation
    """

    @staticmethod
    def forward(ctx,
                points1,
                points2,
                lengths1,
                lengths2,
                K: int,
                r: torch.Tensor,
                sorted_points2=None,
                pc2_grid_off=None,
                sorted_points2_idxs=None,
                grid_params_cuda=None,
                return_sorted: bool = True,
                radius_cell_ratio: float = 2.0):
        """
        TODO: add docs
        """
        use_cached_grid = (sorted_points2 is not None and
                           pc2_grid_off is not None and
                           sorted_points2_idxs is not None and
                           grid_params_cuda is not None)
        N = points1.shape[0]
        D = points1.shape[2]
        # assert D == 2 or D == 3, "For now only 2D/3D is supported"
        # assert D >= 2 and D <= 32
        # setup grid params
        # for D > 3, still use 3D grid
        # TODO: use PCA
        if D >= 3:
            # 0-2 grid_min; 3 grid_delta; 4-6 grid_res; 7 grid_total
            grid_params_size = 8
            grid_delta_idx = 3
            grid_total_idx = 7
            grid_max_res = 64
            grid_dim = 3
        else:
            # 0-1 grid_min; 2 grid_delta; 3-4 grid_res; 5 grid_total
            grid_params_size = 6
            grid_delta_idx = 2
            grid_total_idx = 5
            grid_max_res = 512
            grid_dim = 2

        if not use_cached_grid:
            # create grid from scratch
            # TODO: use struct of array instead of array of struct?
            grid_params_cuda = torch.zeros((N, grid_params_size),
                                           dtype=torch.float,
                                           device=points1.device)
            G = -1
            for i in range(N):
                grid_min = points2[i, :lengths2[i], :grid_dim].min(dim=0)[0]
                grid_max = points2[i, :lengths2[i], :grid_dim].max(dim=0)[0]
                grid_params_cuda[i, :grid_delta_idx] = grid_min
                grid_size = grid_max - grid_min
                cell_size = r[i].item() / radius_cell_ratio
                if cell_size < grid_size.max() / grid_max_res:
                    cell_size = grid_size.max() / grid_max_res
                grid_params_cuda[i, grid_delta_idx] = 1 / cell_size
                grid_params_cuda[i, grid_delta_idx +
                                 1:grid_total_idx] = torch.floor(
                                     grid_size / cell_size) + 1
                grid_params_cuda[i, grid_total_idx] = torch.prod(
                    grid_params_cuda[i, grid_delta_idx + 1:grid_total_idx])
                if G < grid_params_cuda[i, grid_total_idx]:
                    G = int(grid_params_cuda[i, grid_total_idx].item())

            # insert points into the grid
            P2 = points2.shape[1]
            pc2_grid_cnt = torch.zeros((N, G),
                                       dtype=torch.int,
                                       device=points1.device)
            pc2_grid_cell = torch.full((N, P2),
                                       -1,
                                       dtype=torch.int,
                                       device=points1.device)
            pc2_grid_idx = torch.full((N, P2),
                                      -1,
                                      dtype=torch.int,
                                      device=points1.device)
            _C.insert_points_cuda(points2, lengths2, grid_params_cuda,
                                  pc2_grid_cnt, pc2_grid_cell, pc2_grid_idx, G)

            # compute the offset for each grid
            # pc2_grid_off = _C.prefix_sum_cuda(pc2_grid_cnt, grid_params_cuda.cpu())

            # use prefix_sum from Matt Dean
            grid_params = grid_params_cuda.cpu()
            pc2_grid_off = torch.full((N, G),
                                      0,
                                      dtype=torch.int,
                                      device=points1.device)
            for i in range(N):
                prefix_sum_cuda(pc2_grid_cnt[i], grid_params[i,
                                                             grid_total_idx],
                                pc2_grid_off[i])

            # sort points according to their grid positions and insertion orders
            sorted_points2 = torch.zeros_like(points2)
            sorted_points2_idxs = torch.full((N, P2),
                                             -1,
                                             dtype=torch.int,
                                             device=points1.device)
            _C.counting_sort_cuda(points2, lengths2, pc2_grid_cell,
                                  pc2_grid_idx, pc2_grid_off, sorted_points2,
                                  sorted_points2_idxs)

        assert (sorted_points2 is not None and pc2_grid_off is not None and
                sorted_points2_idxs is not None and
                grid_params_cuda is not None)

        G = pc2_grid_off.shape[1]

        # also sort the points1 for faster search
        P1 = points1.shape[1]
        pc1_grid_cnt = torch.zeros((N, G),
                                   dtype=torch.int,
                                   device=points1.device)
        pc1_grid_cell = torch.full((N, P1),
                                   -1,
                                   dtype=torch.int,
                                   device=points1.device)
        pc1_grid_idx = torch.full((N, P1),
                                  -1,
                                  dtype=torch.int,
                                  device=points1.device)
        _C.insert_points_cuda(points1, lengths1, grid_params_cuda,
                              pc1_grid_cnt, pc1_grid_cell, pc1_grid_idx, G)

        # pc1_grid_off = _C.prefix_sum_cuda(pc1_grid_cnt, grid_params_cuda.cpu())
        grid_params = grid_params_cuda.cpu()

        pc1_grid_off = torch.full((N, G),
                                  0,
                                  dtype=torch.int,
                                  device=points1.device)
        for i in range(N):
            prefix_sum_cuda(pc1_grid_cnt[i], grid_params[i, grid_total_idx],
                            pc1_grid_off[i])

        sorted_points1 = torch.zeros((N, P1, D),
                                     dtype=torch.float,
                                     device=points1.device)
        sorted_points1_idxs = torch.full((N, P1),
                                         -1,
                                         dtype=torch.int,
                                         device=points1.device)
        _C.counting_sort_cuda(points1, lengths1, pc1_grid_cell, pc1_grid_idx,
                              pc1_grid_off, sorted_points1,
                              sorted_points1_idxs)

        # perform search on the grid
        idxs, dists = _C.find_nbrs_cuda(sorted_points1, sorted_points2,
                                        lengths1, lengths2, pc2_grid_off,
                                        sorted_points1_idxs,
                                        sorted_points2_idxs, grid_params_cuda,
                                        K, r, r * r)

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

        return (idxs, dists, sorted_points2, pc2_grid_off, sorted_points2_idxs,
                grid_params_cuda)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_idxs, grad_dists, grad_sorted_points2,
                 grad_pc2_grid_off, grad_sorted_points2_idxs,
                 grad_grid_params_cuda):
        points1, points2, lengths1, lengths2, idxs = ctx.saved_tensors
        grad_points1, grad_points2 = _C.frnn_backward_cuda(
            points1, points2, lengths1, lengths2, idxs, grad_dists)
        return (grad_points1, grad_points2, None, None, None, None, None, None,
                None, None, None, None)


def frnn_grid_points(
    points1: torch.Tensor,
    points2: torch.Tensor,
    lengths1: Union[torch.Tensor, None] = None,
    lengths2: Union[torch.Tensor, None] = None,
    K: int = -1,
    r: Union[float, torch.Tensor] = -1,
    grid: Union[_GRID, None] = None,
    return_nn: bool = False,
    # TODO: add non-sorted version?
    # for now we always sort the neighbors by dist
    return_sorted: bool = True,
    radius_cell_ratio: float = 2.0,
    # TODO: maybe add a flag here indicating whether points1 and points2 are the same?
    # also think of ways for further speedup in this scenario
):
    """
  Fixed Radius nearest neighbors search on CUDA with uniform grid for point clouds

  Args:
    points1: Tensor of shape (N, P1, D) giving a batch of N point clouds,
             each containing up to P1 points of dimension 2 or 3.
    points2: Tensor of shape (N, P2, D) giving a batch of N point clouds,
             each containing up to P2 points of dimension 2 or 3.
    lengths1: LongTensor of shape (N,) of values in the range [0, P1], giving the
              length of each pointcloud in p1. Or None to indicate that every cloud has
              length P1.
    lengths2: LongTensor of shape (N,) of values in the range [0, P2], giving the
              length of each pointcloud in p2. Or None to indicate that every cloud has
              length P2.
    K: Integer giving the number of nearest neighbors to return.
    r: Float or a Tensor of shape (1,) or (N,) giving the search radius for the query points.
       If it is not a Tensor of shape(N,), then all point clouds in the batch will have the same search radius.
       (neighbors with distance > search radius would be discarded)
    grid: A tuple of tensors consisting of cached grid structure. 
          If the points2 have been used as reference points before,
          we can reuse the constructed grid to avoid building it for a second time
    return_nn: If set to True returns the nearest neighbors in points2 for each point in points1.
    return_sorted: (bool) whether to return the nearest neighbors sorted in ascending order of distance. 
                   For now this function always return sorted nn no matter the value of this flag.
    radius_cell_ratio: A hyperparameter for grid cell size. Users can just use the defualt value 2.

    Returns:
      dists: Tensor of shape (N, P1, K) giving the squared distances to
             the nearest neighbors. This is padded with zeros both where a cloud in p2
             has fewer than K points and where a cloud in p1 has fewer than P1 points.
      idxs: LongTensor of shape (N, P1, K) giving the indices of the at most K nearest neighbors 
            from points in points1 to points in points2. Concretely, 
            if `idxs[n, i, k] = j` then `points2[n, j]` is the k-th nearest neighbors to 
            `points1[n, i]` in `points2[n]`. This is padded with -1 both where a cloud
            in points2 has fewer than K points and where a cloud in points1 has fewer than P1
            points. Also, when a point has less than K neighbors inside the query ball
            defined by the search radius, it is also padded with -1
      nn: Tensor of shape (N, P1, K, 3) giving the K nearest neighbors in points2 for
            each point in points1. Concretely, `nn[n, i, k]` gives the k-th nearest neighbor
            for `points1[n, i]`. Returned if `return_nn` is True.
            The nearest neighbors are collected using `frnn_gather`

            .. code-block::

                nn = frnn_gather(points2, idxs, lengths2)

            which is a helper function that allows indexing any tensor of shape (N, P2, U) with
            the indices `idxs` returned by `frnn_points`. The outout is a tensor of shape (N, P1, K, U).
      grid: A namedtuple of 4 tensors for reusing the grid if necessary/possible.
            Users don't need to know what's inside.
            _GRID = namedtuple("GRID", "sorted_points2 pc2_grid_off sorted_points2_idxs grid_params")
              sorted_points2 is a spatially sorted version of points2;
              sorted_points2_idxs records the mapping between points2 and sorted_points2;
              grid_params characterizes the constructed grid;
              pc2_grid_off records the start and the end indices of sorted_points2 for each grid
  """

    if points1.shape[0] != points2.shape[0]:
        raise ValueError(
            "points1 and points2 must have the same batch  dimension")
    if points1.shape[2] != points2.shape[2]:
        raise ValueError(
            f"dimension mismatch: points1 of dimension {points1.shape[2]} while points2 of dimension {points2.shape[2]}"
        )
    # if points1.shape[2] != 2 and points1.shape[2] != 3:
    #     raise ValueError("for now only grid in 2D/3D is supported")
    # if points1.shape[2] < 2 or points1.shape[2] > 32:
    #     raise ValueError(
    #         "for now only point clouds of dimension 2-32 is supported")
    if not points1.is_cuda or not points2.is_cuda:
        raise TypeError("for now only cuda version is supported")

    points1 = points1.contiguous()
    points2 = points2.contiguous()

    P1 = points1.shape[1]
    P2 = points2.shape[1]

    if lengths1 is None:
        lengths1 = torch.full((points1.shape[0],),
                              P1,
                              dtype=torch.long,
                              device=points1.device)
    if lengths2 is None:
        lengths2 = torch.full((points2.shape[0],),
                              P2,
                              dtype=torch.long,
                              device=points2.device)

    N = points1.shape[0]
    if isinstance(r, float) or isinstance(r, int):
        r = torch.ones((N,), dtype=torch.float32) * r
    if isinstance(r, torch.Tensor):
        assert (len(r.shape) == 1 and (r.shape[0] == 1 or r.shape[0] == N))
        if r.shape[0] == 1:
            r = r * torch.ones((N,), dtype=r.dtype, device=r.device)
    r = r.type(torch.float32)
    if not r.is_cuda:
        r = r.cuda()

    if grid is not None:
        idxs, dists, sorted_points2, pc2_grid_off, sorted_points2_idxs, grid_params_cuda = _frnn_grid_points.apply(
            points1, points2, lengths1, lengths2, K, r, grid[0], grid[1],
            grid[2], grid[3], return_sorted, radius_cell_ratio)
    else:
        idxs, dists, sorted_points2, pc2_grid_off, sorted_points2_idxs, grid_params_cuda = _frnn_grid_points.apply(
            points1, points2, lengths1, lengths2, K, r, None, None, None, None,
            return_sorted, radius_cell_ratio)

    grid = _GRID(
        sorted_points2=sorted_points2,  # (N, P, D)
        pc2_grid_off=pc2_grid_off,  # (N, G)
        sorted_points2_idxs=sorted_points2_idxs,  # (N, P)
        grid_params=grid_params_cuda)  # (N, 6) or (N, 8)

    points2_nn = None
    if return_nn:
        points2_nn = frnn_gather(points2, idxs, lengths2)

    # follow pytorch3d.ops.knn_points' conventions to return dists frist
    # TODO: also change this in the c++/cuda code?
    return dists, idxs, points2_nn, grid


# TODO: probably do this in cuda?
def frnn_gather(x: torch.Tensor,
                idxs: torch.Tensor,
                lengths: Union[torch.Tensor, None] = None):
    """
  A helper function for frnn that allows indexing a tensor x with the indices `idxs`
  returned by `frnn_points`.

  For example, if `dists, idxs, nn, grid = frnn_points(p, x, lengths_p, lengths, K, r)`
  where p is a tensor of shape (N, L, D) and x a tensor of shape (N, M, D),
  then one can compute the K nearest neighbors of p with `nn = knn_gather(x, idxs, lengths)`.
  It can also be applied for any tensor x of shape (N, M, U) where U != D.

  Args:
      x: Tensor of shape (N, M, U) containing U-dimensional features to be gathered.
      idxs: LongTensor of shape (N, L, K) giving the indices returned by `frnn_points`.
      lengths: LongTensor of shape (N,) of values in the range [0, M], giving the
               length of each example in the batch in x. Or None to indicate that every
               example has length M.
  Returns:
      x_out: Tensor of shape (N, L, K, U) resulting from gathering the elements of x
             with idx, s.t. `x_out[n, l, k] = x[n, idx[n, l, k]]`.
             If `k > lengths[n]` then `x_out[n, l, k]` is filled with 0.0.
  """
    N, P2, D = x.shape  # M: number of points
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


class _frnn_bf_points(Function):
    """
    Torch autograd Functio wrapper for FRNN CUDA brute force implementation
    (validation only)
    """

    @staticmethod
    def forward(ctx, points1, points2, lengths1, lengths2, K, r):
        idxs, dists = _C.frnn_bf_cuda(points1, points2, lengths1, lengths2, K,
                                      r)

        ctx.save_for_backward(points1, points2, lengths1, lengths2, idxs)
        ctx.mark_non_differentiable(idxs)

        return idxs, dists

    @staticmethod
    def backward(ctx, grad_idxs, grad_dists):
        points1, points2, lengths1, lengths2, idxs = ctx.saved_tensors
        grad_points1, grad_points2 = _C.frnn_backward_cuda(
            points1, points2, lengths1, lengths2, idxs, grad_dists)
        return grad_points1, grad_points2, None, None, None, None


def frnn_bf_points(points1,
                   points2,
                   lengths1,
                   lengths2,
                   K,
                   r,
                   return_nn=False):
    idxs, dists = _frnn_bf_points.apply(points1, points2, lengths1, lengths2,
                                        K, r)
    points2_nn = None
    if return_nn:
        points2_nn = frnn_gather(points2, idxs, lengths2)
    return idxs, dists, points2_nn
