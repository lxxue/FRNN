import glob
import csv

import torch
import frnn

num_points_fixed_query = 100

class TestFRNN:
  def __init__(self, num_pcs=2, r=0.1):
    self.pc1 = torch.rand((num_pcs, num_points_fixed_query, 2), dtype=torch.float).cuda()
    self.pc2 = torch.rand((num_pcs, num_points_fixed_query, 2), dtype=torch.float).cuda()
    self.num_pcs = num_pcs
    self.r = r
    self.num_points = num_points_fixed_query
    lengths1 = torch.ones((num_pcs,), dtype=torch.long) * num_points_fixed_query 
    lengths2 = torch.ones((num_pcs,), dtype=torch.long) * num_points_fixed_query
    self.lengths1_cuda = lengths1.cuda()
    self.lengths2_cuda = lengths2.cuda()

  def frnn_sort_2d(self):
    sorted_points2, sorted_points2_idxs, pc2_grid_off, grid_params = frnn._frnn_sort_points(
        self.pc1,
        self.pc2,
        self.lengths1_cuda,
        self.lengths2_cuda,
        self.r,
        radius_cell_ratio=1.
    )
    for i in range(sorted_points2.shape[1]):
        print(sorted_points2[1][i])


if __name__ == "__main__":
  validator = TestFRNN()
  validator.frnn_sort_2d()