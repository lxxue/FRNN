import torch
import frnn
import unittest
from pytorch3d.structures import Pointclouds
from pytorch3d.ops.knn import knn_points

# from pytorch3d.io import load_ply
from pytorch_points.utils.pc_utils import read_ply

import argparse
import glob
import csv

class ValidateFRNN:
  def __init__(self, fname, num_pcs=1, K=5, r=0.1):
    if 'random' in fname:
      # fname format: random_{num_points}
      num_points = int(fname.split('_')[1])
      pc1 = torch.rand((num_pcs, num_points, 3), dtype=torch.float)
      pc2 = torch.rand((num_pcs, num_points, 3), dtype=torch.float)
      for i in range(num_pcs):
        for j in range(3):
          pc1[i, :, j] *= torch.rand(1)+0.5
          pc2[i, :, j] *= torch.rand(1)+0.5
    else:
      pc1 = torch.FloatTensor(read_ply(fname)[None, :, :3])  # no need for normals
      # pc2 = pc1
      pc2 = torch.FloatTensor(read_ply(fname)[None, :, :3])  # no need for normals
      normalize_pc(pc1)
      normalize_pc(pc2)
      # print("pc1 bbox: ", pc1.min(dim=1)[0], pc1.max(dim=1)[0])
      num_points = pc1.shape[1]
      if num_pcs > 1:
        pc1 = pc1.repeat(num_pcs, 1, 1)
        pc2 = pc2.repeat(num_pcs, 1, 1)
    self.num_pcs = num_pcs
    self.fname = fname.split('/')[-1]
    self.K = K
    self.r = r
    self.num_points = num_points
    self.pc1_cuda = pc1.cuda()
    self.pc2_cuda = pc2.cuda()
    lengths1 = torch.ones((num_pcs,), dtype=torch.long) * num_points
    lengths2 = torch.ones((num_pcs,), dtype=torch.long) * num_points
    self.lengths1_cuda = lengths1.cuda()
    self.lengths2_cuda = lengths2.cuda()
    print("{}: #points: {}".format(self.fname, num_points))
    self.grid = None

  def frnn_grid(self): 
    idxs_cuda, dists_cuda, nn, grid = frnn.frnn_grid_points(
      self.pc1_cuda,
      self.pc2_cuda,
      self.lengths1_cuda,
      self.lengths2_cuda,
      K=self.K,
      r=self.r,
      return_grid=True
    )
    if self.grid is None:
      self.grid = grid
    return idxs_cuda, dists_cuda

  def frnn_grid_reuse(self):
    idxs_cuda_2, dists_cuda_2, nn, _ = frnn.frnn_grid_points(
      self.pc1_cuda,
      self.pc2_cuda,
      self.lengths1_cuda,
      self.lengths2_cuda,
      self.grid,
      K = self.K,
      r = self.r
    )
    return idxs_cuda_2, dists_cuda_2

  def knn(self):
    knn_results = knn_points(
      self.pc1_cuda, 
      self.pc2_cuda, 
      self.lengths1_cuda,
      self.lengths2_cuda,
      self.K
    )
    return knn_results

  def frnn_bf(self):
    idxs_cuda_bf, dists_cuda_bf = frnn._C.frnn_bf_cuda(
      self.pc1_cuda,
      self.pc2_cuda,
      self.lengths1_cuda,
      self.lengths2_cuda,
      self.K,
      self.r
    )
    return idxs_cuda_bf, dists_cuda_bf

  def compare(self):
    if self.num_points > 1000000:
      print("\tnumber of points for exceed 1 million; skip")
      return
    idxs_bf, dists_bf = self.frnn_bf()
    idxs_grid, dists_grid = self.frnn_grid()

    diff_keys_percentage = torch.sum(idxs_grid == idxs_bf).type(torch.float).item() / self.K / self.num_points / self.num_pcs
    dists_all_close = torch.allclose(dists_bf, dists_grid)
    return [self.fname, self.num_points, "{:.4f}".format(diff_keys_percentage), dists_all_close]
      

def normalize_pc(pc):
  # convert pc to the unit box so that we don't need to manually set raidus for each mesh
  # pc should be 1 x P x 3
  # [0, 1] x [0, 1] x [0, 1]
  assert pc.shape[0] == 1 and pc.shape[2] == 3
  pc -= pc.min(dim=1)[0]
  pc /= torch.max(pc)
  # print(pc.min(dim=1), pc.max(dim=1))
  return

if __name__ == "__main__":
  fnames = sorted(glob.glob('data/*.ply') + glob.glob('data/*/*.ply'))
  fnames += ['random_10000', 'random_100000', 'random_1000000']
  print(fnames)
  with open("frnn_validation.csv", 'w') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Point cloud', 'Num points', 'Different key percentage', 'Dists all close'])
    for fname in fnames:
      if 'xyz' in fname or 'lucy' in fname:
        continue
      validator = ValidateFRNN(fname)
      results = validator.compare()
      writer.writerow(results)