import torch
from frnn_whole import Compare
import frnn

import glob
from itertools import product
import numpy as np

def frnn_r(fname, N, K, r):
  ragged = False
  print(fname, N, K, r)
  points1 = torch.load(fname)
  points2 = torch.load(fname)
  # print(points1.min(dim=1)[0], points1.max(dim=1)[0])
  if N > 1:
    points1 = points1.repeat(N, 1, 1)
    points2 = points2.repeat(N, 1, 1)
  if ragged:
    lengths1 = torch.randint(low=1, high=points1.shape[1], size=(N,), dtype=torch.long, device=points1.device)
    lengths2 = torch.randint(low=1, high=points2.shape[1], size=(N,), dtype=torch.long, device=points2.device)
  else:
    lengths1 = torch.ones((N,), dtype=torch.long, device=points1.device) * points1.shape[1]
    lengths2 = torch.ones((N,), dtype=torch.long, device=points2.device) * points2.shape[1]

  dists, idxs, nn, grid = frnn.frnn_grid_points(
    points1, points2, lengths1, lengths2, K, r, grid=None, return_nn=False, return_sorted=True
  )

  return
  

if __name__ == "__main__":
  fnames = sorted(glob.glob("data/pc/*.pt"))
  Ns = [8]
  Ks = [8]
  rs = [0.1, 0.1*torch.ones((1,)), 0.1*torch.ones((1,)).cuda(),
        0.1*torch.ones(8), 0.1*torch.ones(8,).cuda(),
        0.1*torch.FloatTensor(np.arange(1, 9))]
  test_cases = product(fnames, Ns, Ks, rs)

  for case in test_cases:
    fname, N, K, r = case
    # kwarg_dict = {"fname":fname, "N":N, "K":K, "r":r}
    # frnn_r(**kwarg_dict)
    frnn_r(fname, N, K , r)
    break