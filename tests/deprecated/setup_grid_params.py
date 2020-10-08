import torch
import frnn
import unittest
from pytorch3d.structures import Pointclouds

class Test(unittest.TestCase):
  K = 5
  r = 0.1
  r2 = r * r
  num_pcs = 5
  max_num_points = 1000

  pc = torch.rand((num_pcs, max_num_points, 3), dtype=torch.float)
  for i in range(num_pcs):
    for j in range(3):
      pc[i, :, j] *= torch.rand(1)+0.5
  pc_cuda = pc.cuda()
  lengths = torch.randint(low=K, high=max_num_points, size=(num_pcs,), dtype=torch.long)
  lengths_cuda = lengths.cuda()

  def test_setup_grid_params_cuda(self):
    pc_list = [self.pc[i, :self.lengths[i]] for i in range(len(self.pc))]
    pc = Pointclouds(pc_list)
    bboxes = pc.get_bounding_boxes() 
    frnn._C.test_setup_grid_params_cuda(bboxes, 0.1)
    # need to modify frnn_grid_points to print params
    frnn.frnn_grid_points(self.pc_cuda, self.pc_cuda, self.lengths_cuda, self.lengths_cuda, K=5, r=0.1)


if __name__ == "__main__":
  unittest.main()