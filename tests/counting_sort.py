import torch
import frnn
import unittest
from pytorch3d.structures import Pointclouds

class Test(unittest.TestCase):
  K = 5
  r = 0.5
  r2 = r * r
  num_pcs = 1
  max_num_points = 10

  pc = torch.rand((num_pcs, max_num_points, 3), dtype=torch.float)
  for i in range(num_pcs):
    for j in range(3):
      # pc[i, :, j] *= torch.rand(1)+0.5
      pc[i, :, j] *= 1
  pc_cuda = pc.cuda()
  lengths = torch.randint(low=K, high=max_num_points, size=(num_pcs,), dtype=torch.int)
  lengths_cuda = lengths.cuda()

  def test_counting_sort_cuda(self):
    pc = Pointclouds(self.pc)
    bboxes = pc.get_bounding_boxes() 
    grid_cnt_cpu, grid_cell_cpu = frnn.test_insert_points_cpu(bboxes, self.pc, self.lengths, self.r)
    grid_cnt_cuda, grid_cell_cuda, grid_idx_cuda = frnn.test_insert_points_cuda(bboxes, self.pc_cuda, self.lengths_cuda, self.r)
    grid_idx_cpu = grid_idx_cuda.cpu()
    print(torch.allclose(grid_cell_cpu, grid_cell_cuda.cpu()))
    print(torch.allclose(grid_cnt_cpu, grid_cnt_cuda.cpu()))
    grid_off_cpu = frnn.prefix_sum_cpu(grid_cnt_cpu)
    grid_off_cuda = frnn.prefix_sum_cuda(grid_cnt_cuda)
    print(torch.allclose(grid_off_cpu, grid_off_cuda.cpu()))

    sorted_points_cpu = torch.zeros_like(self.pc)
    sorted_points_cuda = torch.zeros_like(self.pc_cuda)
    sorted_grid_cell_cpu = -torch.ones_like(grid_cell_cpu)
    sorted_grid_cell_cuda = -torch.ones_like(grid_cell_cuda)
    sorted_point_idx_cpu = -torch.ones_like(grid_cell_cpu)
    sorted_point_idx_cuda = -torch.ones_like(grid_cell_cuda)

    # print(self.pc, self.lengths)
    # print(grid_cell_cpu)
    # print(grid_idx_cpu)
    # print(grid_off_cpu)

    frnn.counting_sort_cpu(
      self.pc,
      self.lengths,
      grid_cell_cpu,
      grid_idx_cpu,
      grid_off_cpu,
      sorted_points_cpu,
      sorted_grid_cell_cpu,
      sorted_point_idx_cpu
    )

    frnn.counting_sort_cuda(
      self.pc_cuda,
      self.lengths_cuda,
      grid_cell_cuda,
      grid_idx_cuda,
      grid_off_cuda,
      sorted_points_cuda,
      sorted_grid_cell_cuda,
      sorted_point_idx_cuda
    )

    print(sorted_points_cpu[0, :10, :])
    print(sorted_points_cuda[0, :10, :])
    print(sorted_grid_cell_cpu[0, :10])
    print(sorted_grid_cell_cuda[0, :10])
    print(sorted_point_idx_cpu[0, :10])
    print(sorted_point_idx_cuda[0, :10])

    print(torch.allclose(sorted_points_cpu, sorted_points_cuda.cpu()))
    print(torch.allclose(sorted_grid_cell_cpu, sorted_grid_cell_cuda.cpu()))
    print(torch.allclose(sorted_point_idx_cpu, sorted_point_idx_cuda.cpu()))
  

if __name__ == "__main__":
  unittest.main()