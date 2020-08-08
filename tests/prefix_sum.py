import torch
import frnn
import unittest
from pytorch3d.structures import Pointclouds

class Test(unittest.TestCase):
  K = 5
  r = 0.1
  r2 = r * r
  num_pcs = 10
  max_num_points = 1000

  pc = torch.rand((num_pcs, max_num_points, 3), dtype=torch.float)
  for i in range(num_pcs):
    for j in range(3):
      pc[i, :, j] *= torch.rand(1)+0.5
      # pc[i, :, j] *= 1
  pc_cuda = pc.cuda()
  lengths = torch.randint(low=K, high=max_num_points, size=(num_pcs,), dtype=torch.long)
  lengths_cuda = lengths.cuda()

  def test_insert_points_cuda(self):
    pc = Pointclouds(self.pc)
    bboxes = pc.get_bounding_boxes() 
    print(self.lengths)
    # grid_cnt_cpu, grid_cell_cpu, grid_idx_cpu = frnn.test_insert_points_cpu(bboxes, self.pc, self.lengths, self.r)
    # grid_cnt_cuda, grid_cell_cuda, grid_idx_cuda = frnn.test_insert_points_cuda(bboxes, self.pc_cuda, self.lengths_cuda, self.r)
    # print(torch.allclose(grid_cell_cpu, grid_cell_cuda.cpu()))
    # print(torch.allclose(grid_cnt_cpu, grid_cnt_cuda.cpu()))
    # grid_off_cpu = frnn.prefix_sum_cpu(grid_cnt_cpu)
    # grid_off_cuda = frnn.prefix_sum_cuda(grid_cnt_cuda)
    grid_off_cpu = frnn.test_prefix_sum_cpu(
      bboxes,
      self.pc,
      self.lengths,
      self.r
    )
    print("cpu done")
    grid_off_cuda = frnn.test_prefix_sum_cuda(
      bboxes,
      self.pc_cuda,
      self.lengths_cuda,
      self.r
    )
    print("cuda done")
    print(grid_off_cpu[0, :10])
    print(grid_off_cuda[0, :10])
    print(grid_off_cpu[1, :10])
    print(grid_off_cuda[1, :10])
    print(torch.allclose(grid_off_cpu, grid_off_cuda.cpu()))

if __name__ == "__main__":
  unittest.main()