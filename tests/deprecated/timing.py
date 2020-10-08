import torch
import frnn
import unittest
from pytorch3d.structures import Pointclouds
from pytorch3d.ops.knn import knn_points

# from pytorch3d.io import load_ply
from pytorch_points.utils.pc_utils import read_ply

import time
import argparse
import glob
import csv

class TimeFRNN:
  def __init__(self, fname, num_pcs=1, K=5, r=0.1, same=False):
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
      num_points = pc2.shape[1]
      if num_pcs > 1:
        pc1 = pc1.repeat(num_pcs, 1, 1)
        pc2 = pc2.repeat(num_pcs, 1, 1)
    if not same:
      pc1 = torch.rand((num_pcs, 100000, 3), dtype=torch.float)
    self.fname = fname.split('/')[-1]
    self.K = K
    self.r = r
    self.num_points = num_points
    self.pc1_cuda = pc1.cuda()
    self.pc2_cuda = pc2.cuda()
    if same:
      lengths1 = torch.ones((num_pcs,), dtype=torch.long) * num_points
    else:
      lengths1 = torch.ones((num_pcs,), dtype=torch.long) * 100000
    lengths2 = torch.ones((num_pcs,), dtype=torch.long) * num_points
    self.lengths1_cuda = lengths1.cuda()
    self.lengths2_cuda = lengths2.cuda()
    print("{}: #points: {}".format(self.fname, num_points))
    self.grid = None

  def frnn_grid(self): 
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
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
    end.record()
    torch.cuda.synchronize()
    grid_time = start.elapsed_time(end)
    return grid_time

  def frnn_grid_reuse(self):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    idxs_cuda_2, dists_cuda_2, nn, _ = frnn.frnn_grid_points(
      self.pc1_cuda,
      self.pc2_cuda,
      self.lengths1_cuda,
      self.lengths2_cuda,
      K = self.K,
      r = self.r
      self.grid,
    )
    end.record()
    torch.cuda.synchronize()
    grid_search_time = start.elapsed_time(end)
    return grid_search_time

  def knn(self):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    knn_results = knn_points(
      self.pc1_cuda, 
      self.pc2_cuda, 
      self.lengths1_cuda,
      self.lengths2_cuda,
      self.K
    )
    end.record()
    torch.cuda.synchronize()
    knn_time = start.elapsed_time(end)
    return knn_time

  def frnn_bf(self):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    idxs_cuda_bf, dists_cuda_bf = frnn._C.frnn_bf_cuda(
      self.pc1_cuda,
      self.pc2_cuda,
      self.lengths1_cuda,
      self.lengths2_cuda,
      self.K,
      self.r
    )
    end.record()
    torch.cuda.synchronize()
    bf_time = start.elapsed_time(end)
    return bf_time

  def compare(self, num_exp=10):
    if self.num_points > 1000000:
      print("\tnumber of points for exceed 1 million; skip")
      return
    knn_time = 0
    frnn_bf_time = 0
    frnn_grid_time = 0
    frnn_grid_search_time = 0

    for i in range(num_exp):
      knn_time += self.knn()
      frnn_bf_time += self.frnn_bf()
      frnn_grid_time += self.frnn_grid()
      frnn_grid_search_time += self.frnn_grid_reuse()
      
    knn_time /= num_exp
    frnn_bf_time /= num_exp
    frnn_grid_time /= num_exp
    frnn_grid_search_time /= num_exp
    # print("\tknn time: {:.2f}; bf time: {:.2f}; grid time: {:.2f}; grid search time: {:.2f}".format(knn_time, frnn_bf_time, frnn_grid_time, frnn_grid_search_time))
    return [self.fname, self.num_points, "{:.2f}".format(knn_time), "{:.2f}".format(frnn_bf_time), "{:.2f}".format(frnn_grid_time), "{:.2f}".format(frnn_grid_search_time)]

# def TimeFRNN(fname, num_pcs=1, K=5, r=0.1):
# 
#   print("{} #pcs: {:d}; #points {:d};".format(fname, num_pcs, num_points))
#   print("\tgrid vs bf # diff keys: ", torch.sum(idxs_cuda != idxs_cuda_bf).item())
#   print("\tgrid from scratch vs grid reuse # diff keys: ", torch.sum(idxs_cuda != idxs_cuda_2).item())
#   print("\tknn time: {:.2f}; bf time: {:.2f}; grid time: {:.2f}; grid search time: {:.2f}".format(knn_time, bf_time, grid_time, grid_search_time))


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
  with open("tests/output/timing.csv", 'w') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Point cloud', 'Num points', 'knn time', 'frnn bf time', 'frnn grid time', 'frnn grid search time'])
    for fname in fnames:
      if 'xyz' in fname or 'lucy' in fname:
        continue
      timer = TimeFRNN(fname, same=False)
      results = timer.compare(num_exp=10)
      writer.writerow(results)
  
  with open("tests/output/timing_same.csv", 'w') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Point cloud', 'Num points', 'knn time', 'frnn bf time', 'frnn grid time', 'frnn grid search time'])
    for fname in fnames:
      if 'xyz' in fname or 'lucy' in fname:
        continue
      timer = TimeFRNN(fname, same=True)
      results = timer.compare(num_exp=10)
      writer.writerow(results) 