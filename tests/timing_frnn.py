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
    # pc1 = torch.rand((num_pcs, 100000, 3), dtype=torch.float)
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
    if not same:
      lengths1 = torch.ones((num_pcs,), dtype=torch.long) * 100000
    else:
      lengths1 = torch.ones((num_pcs,), dtype=torch.long) * num_points
    lengths2 = torch.ones((num_pcs,), dtype=torch.long) * num_points
    self.lengths1_cuda = lengths1.cuda()
    self.lengths2_cuda = lengths2.cuda()
    print("{}: #points: {}".format(self.fname, num_points))

  def frnn_grid(self): 
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    idxs_cuda, dists_cuda, nn, grid, \
      setup_time, insert_points_time, prefix_sum_time, counting_sort_time, find_nbrs_time  = frnn.frnn_grid_points_with_timing(
      self.pc1_cuda,
      self.pc2_cuda,
      self.lengths1_cuda,
      self.lengths2_cuda,
      K=self.K,
      r=self.r,
    )
    end.record()
    torch.cuda.synchronize()
    total_time = start.elapsed_time(end)
    return total_time, setup_time, insert_points_time, prefix_sum_time, counting_sort_time, find_nbrs_time 

  def compare(self, num_exp=10):
    if self.num_points > 1000000:
      print("\tnumber of points for exceed 1 million; skip")
      return

    total = 0
    setup = 0
    insert_points = 0
    prefix_sum = 0
    counting_sort = 0
    find_nbrs = 0
    for i in range(num_exp):
      total_time, setup_time, insert_points_time, prefix_sum_time, counting_sort_time, find_nbrs_time = self.frnn_grid()
      total += total_time
      setup += setup_time
      insert_points += insert_points_time
      prefix_sum += prefix_sum_time
      counting_sort += counting_sort_time
      find_nbrs += find_nbrs_time
      
    # print("\ttotal time: {:.2f}; \n\tsetup time: {:.2f}; \n\tinsert_points time: {:.2f}; \n\tprefix_sum time: {:.2f}; \n\tcounting_sort time: {:.2f}; \n\tfind_nbrs time: {:.2f}".format(total, setup, insert_points, prefix_sum, counting_sort, find_nbrs))
    return [self.fname, self.num_points, total, setup, insert_points, prefix_sum, counting_sort, find_nbrs]

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
  # parser = argparse.ArgumentParser()
  # parser.add_argument("--pc", type=str, default=None)
  # args = parser.parse_args()
  # TimeFindNbrs(1, 10000, args.pc)
  # TimeFindNbrs(1, 100000, args.pc)
  # TimeFindNbrs(1, 1000000, args.pc)
  # TimeFindNbrs(10, 10000, args.pc)
  # TimeFindNbrs(10, 100000, args.pc)
  # TimeFindNbrs(10, 1000000, args.pc)
  fnames = sorted(glob.glob('data/*.ply') + glob.glob('data/*/*.ply'))
  fnames += ['random_10000', 'random_100000', 'random_1000000']
  print(fnames)
  with open("tests/output/timing_frnn.csv", 'w') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Point cloud', 'Num points', 'total time', 'setup time', 'insert points time', 'prefix sum time', 'counting sort time', 'grid search time'])
    for fname in fnames:
      if 'xyz' in fname or 'lucy' in fname:
        continue
      timer = TimeFRNN(fname, same=False)
      results = timer.compare(num_exp=10)
      writer.writerow(results)
      # TimeFRNN('data/lucy.ply')
      # TimeFRNN('data/drill/drill_shaft_vrip.ply')
      # break

  with open("tests/output/timing_frnn_same.csv", 'w') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Point cloud', 'Num points', 'total time', 'setup time', 'insert points time', 'prefix sum time', 'counting sort time', 'grid search time'])
    for fname in fnames:
      if 'xyz' in fname or 'lucy' in fname:
        continue
      timer = TimeFRNN(fname, same=True)
      results = timer.compare(num_exp=10)
      writer.writerow(results)