import torch
import frnn
import frnn._C
import unittest
from pytorch3d.structures import Pointclouds

GRID_PARAMS_SIZE = 8
MAX_RES = 100

class Test(unittest.TestCase):
  K = 5
  r = 0.1
  r2 = r * r
  num_pcs = 5
  pc1_max_num_points = 10000000
  pc2_max_num_points = 1000000 

  pc1 = torch.rand((num_pcs, pc1_max_num_points, 3), dtype=torch.float)
  pc2 = torch.rand((num_pcs, pc2_max_num_points, 3), dtype=torch.float)
  for i in range(num_pcs):
    for j in range(3):
      pc1[i, :, j] *= torch.rand(1) + 0.5
      pc2[i, :, j] *= torch.rand(1) + 0.5
  
  pc1_cuda = pc1.cuda()
  pc2_cuda = pc2.cuda()
  lengths1 = torch.randint(low=K, high=pc1_max_num_points, size=(num_pcs,), dtype=torch.long)
  lengths2 = torch.randint(low=K, high=pc2_max_num_points, size=(num_pcs,), dtype=torch.long)
  lengths1_cuda = lengths1.cuda()
  lengths2_cuda = lengths2.cuda()
  G = -1

  def _grid_params(self):
    self.grid_params_cuda = torch.zeros((self.num_pcs, GRID_PARAMS_SIZE), dtype=torch.float, device=self.pc1_cuda.device)
    self.G = -1
    for i in range(self.num_pcs):
      grid_min = self.pc2_cuda[i, :self.lengths2_cuda[i]].min(dim=0)[0]
      grid_max = self.pc2_cuda[i, :self.lengths2_cuda[i]].max(dim=0)[0]
      print(grid_min)
      print(grid_max)
      self.grid_params_cuda[i, :3] = grid_min
      grid_size = grid_max - grid_min
      cell_size = self.r
      if cell_size < grid_size.min()/MAX_RES:
        cell_size = grid_size.min() / MAX_RES
      self.grid_params_cuda[i, 3] = 1 / cell_size
      self.grid_params_cuda[i, 4:7] = torch.floor(grid_size / cell_size) + 1
      self.grid_params_cuda[i, 7] = self.grid_params_cuda[i, 4] * self.grid_params_cuda[i, 5] * self.grid_params_cuda[i, 6]
      if self.G < self.grid_params_cuda[i, 7]:
        self.G = int(self.grid_params_cuda[i, 7].item())
    
    print(self.G)
    print(self.grid_params_cuda)
    print(self.grid_params_cuda.cpu())

  def _insert_points(self):
    self.grid_cnt = torch.zeros((self.num_pcs, self.G), dtype=torch.int, device=self.pc1_cuda.device)
    self.grid_cell = torch.full((self.num_pcs, self.pc2_max_num_points), -1, dtype=torch.int, device=self.pc1_cuda.device)
    self.grid_idx = torch.full((self.num_pcs, self.pc2_max_num_points), -1, dtype=torch.int, device=self.pc1_cuda.device)

    frnn._C.insert_points_cuda(self.pc2_cuda, self.lengths2_cuda, self.grid_params_cuda, self.grid_cnt, self.grid_cell, self.grid_idx, self.G)

    print(self.grid_cnt)
    print(self.grid_cell)
    print(self.grid_idx)

  def _prefix_sum(self):
    self.grid_off = frnn._C.prefix_sum_cuda(self.grid_cnt, self.grid_params_cuda.cpu())
    print(self.grid_off)

  def _counting_sort(self):
    self.sorted_points2 = torch.zeros((self.num_pcs, self.pc2_max_num_points, 3), dtype=torch.float, device=self.pc2_cuda.device)
    self.sorted_points2_idxs = torch.full((self.num_pcs, self.pc2_max_num_points), -1, dtype=torch.int, device=self.pc2_cuda.device)
    frnn._C.counting_sort_cuda(
      self.pc2_cuda,
      self.lengths2_cuda,
      self.grid_cell,
      self.grid_idx,
      self.grid_off,
      self.sorted_points2,
      self.sorted_points2_idxs
    )
    print(self.sorted_points2)
    print(self.sorted_points2_idxs)

  def _find_nbrs(self):
    self.idxs, self.dists = frnn._C.find_nbrs_cuda(
      self.pc1_cuda,
      self.sorted_points2,
      self.lengths1_cuda,
      self.lengths2_cuda,
      self.grid_off,
      self.sorted_points2_idxs,
      self.grid_params_cuda,
      self.K,
      self.r
    )
    print(self.idxs)
    print(self.dists)

  # def test_separately(self):
  #   self._grid_params()
  #   self._insert_points()
  #   self._prefix_sum()
  #   self._counting_sort()
  #   self._find_nbrs()

  def test_together(self):
    idxs, dists, _, grid = frnn.frnn_grid_points(self.pc1_cuda, self.pc2_cuda, self.lengths1_cuda, self.lengths2_cuda, self.K, self.r, None, return_grid=True)
    # print(idxs)
    # print(dists)
    # print(grid[0])
    # print(grid[1])
    # print(grid[2])
    # print(grid[3])

if __name__ == "__main__":
  unittest.main() 

