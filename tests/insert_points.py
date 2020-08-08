import torch
import frnn
import unittest
from pytorch3d.structures import Pointclouds

class Test(unittest.TestCase):
  K = 5
  r = 0.1
  r2 = r * r
  num_pcs = 10
  max_num_points = 10000

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
    grid_cnt_cpu, grid_cell_cpu, grid_idx_cpu = frnn.test_insert_points_cpu(bboxes, self.pc, self.lengths, self.r)
    grid_cnt_cuda, grid_cell_cuda, grid_idx_cuda = frnn.test_insert_points_cuda(bboxes, self.pc_cuda, self.lengths_cuda, self.r)
    # print(grid_cnt_cpu.shape)
    # print(grid_cnt_cuda.shape)
    # print(grid_cell_cpu.shape)
    # print(grid_cell_cuda.shape)
    # print(grid_cnt_cpu[1, :10])
    # print(grid_cnt_cuda[1, :10])
    # print(grid_cell_cpu[1, :10])
    # print(grid_cell_cuda[1, :10])
    print(self.pc[1])
    print(grid_cnt_cpu[1])
    print(grid_cnt_cuda[1])
    print(grid_cell_cpu[1])
    print(grid_cell_cuda[1])
    print(torch.allclose(grid_cell_cpu[0], grid_cell_cuda[0].cpu()))
    print(torch.allclose(grid_cnt_cpu[0], grid_cnt_cuda[0].cpu()))
    print(torch.allclose(grid_cell_cpu[1], grid_cell_cuda[1].cpu()))
    print(torch.allclose(grid_cnt_cpu[1], grid_cnt_cuda[1].cpu()))
    print(torch.allclose(grid_cell_cpu, grid_cell_cuda.cpu()))
    print(torch.allclose(grid_cnt_cpu, grid_cnt_cuda.cpu()))

    print("grid_idx can be quite different since the insertion order might be different")
    print("the precentage will decrease as #points increases")
    print(torch.sum(grid_idx_cpu == grid_idx_cuda.cpu()).type(torch.float).item() / grid_idx_cpu.shape[0] / grid_idx_cpu.shape[1])

    # print(grid_cell_cpu[0, self.lengths[0]:])
    # print(grid_cell_cuda[0, self.lengths[0]:])

if __name__ == "__main__":
  unittest.main()