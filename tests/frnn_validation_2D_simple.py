import glob
import csv

import torch
import frnn

num_points_fixed_query = 10000

class TestFRNN:
  def __init__(self, num_pcs=1, K=5, r=0.1):
    self.pc1 = torch.rand((num_pcs, num_points_fixed_query, 2), dtype=torch.float).cuda()
    self.pc2 = torch.rand((num_pcs, num_points_fixed_query, 2), dtype=torch.float).cuda()
    self.num_pcs = num_pcs
    self.r = r
    self.K = K
    self.num_points = num_points_fixed_query
    lengths1 = torch.ones((num_pcs,), dtype=torch.long) * num_points_fixed_query 
    lengths2 = torch.ones((num_pcs,), dtype=torch.long) * num_points_fixed_query
    self.lengths1_cuda = lengths1.cuda()
    self.lengths2_cuda = lengths2.cuda()

  def frnn_2d(self):
    dists, idxs, nn, grid = frnn.frnn_grid_points(
      self.pc1, 
      self.pc2, 
      self.lengths1_cuda,
      self.lengths2_cuda,
      self.K,
      self.r,
      radius_cell_ratio=1.0
    )
    sorted_points2 = grid.sorted_points2
    sorted_points2_idxs = grid.sorted_points2_idxs[:, :, None].long().expand(-1, -1, 2)
    idxs_pc2 = torch.gather(self.pc2, 1, sorted_points2_idxs)
    print(torch.allclose(sorted_points2, idxs_pc2))



if __name__ == "__main__":
  validator = TestFRNN()
  validator.frnn_2d()