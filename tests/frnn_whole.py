import unittest
import glob
import torch

import frnn
from fvcore.common.benchmark import benchmark
from pytorch3d.ops import knn_points


# import faiss

class Compare(unittest.TestCase):
  def setUp(self) -> None:
    super().setUp()
    torch.manual_seed(1)

  @staticmethod
  def frnn(N, fname, K, ratio=1):
    r=0.1
    ragged = False
    print(fname, N, K)
    points1 = torch.load("data/pc/"+fname)
    points2 = torch.load("data/pc/"+fname)
    print(points1.min(dim=1)[0], points1.max(dim=1)[0])
    if N > 1:
      points1 = points1.repeat(N, 1, 1)
      points2 = points2.repeat(N, 1, 1)
    if ragged:
      lengths1 = torch.randint(low=1, high=points1.shape[1], size=(N,), dtype=torch.long, device=points1.device)
      lengths2 = torch.randint(low=1, high=points2.shape[1], size=(N,), dtype=torch.long, device=points2.device)
    else:
      lengths1 = torch.ones((N,), dtype=torch.long, device=points1.device) * points1.shape[1]
      lengths2 = torch.ones((N,), dtype=torch.long, device=points2.device) * points2.shape[1]
    torch.cuda.synchronize()

    def output():
      dists, idxs, nn, grid = frnn.frnn_grid_points(
        points1, points2, lengths1, lengths2, K, r, grid=None, return_nn=False, return_sorted=True, radius_cell_ratio=ratio
      )
      torch.cuda.synchronize()
    
    return output

  @staticmethod
  def knn(N, fname, K):
    print(fname, N, K)
    ragged = False
    points1 = torch.load("data/pc/"+fname)
    points2 = torch.load("data/pc/"+fname)
    print(points1.min(dim=1)[0], points1.max(dim=1)[0])
    if N > 1:
      points1 = points1.repeat(N, 1, 1)
      points2 = points2.repeat(N, 1, 1)
    if ragged:
      lengths1 = torch.randint(low=1, high=points1.shape[1], size=(N,), dtype=torch.long, device=points1.device)
      lengths2 = torch.randint(low=1, high=points2.shape[1], size=(N,), dtype=torch.long, device=points2.device)
    else:
      lengths1 = torch.ones((N,), dtype=torch.long, device=points1.device) * points1.shape[1]
      lengths2 = torch.ones((N,), dtype=torch.long, device=points2.device) * points2.shape[1]
    torch.cuda.synchronize()

    def output():
      dists, idxs, nn = knn_points(
        points1, points2, lengths1, lengths2, K, version=-1, return_nn=False, return_sorted=True
      ) 
      torch.cuda.synchronize()

    return output
