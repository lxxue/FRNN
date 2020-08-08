import torch
import frnn
import unittest
from pytorch3d.structures import Pointclouds

class Test(unittest.TestCase):
  K = 5
  r = 0.1
  r2 = r * r
  num_pcs = 5
  max_num_points = 10000

  pc = torch.rand((num_pcs, max_num_points, 3), dtype=torch.float)
  for i in range(num_pcs):
    for j in range(3):
      pc[i, :, j] *= torch.rand(1)+0.5
      # pc[i, :, j] *= 1
  pc_cuda = pc.cuda()
  lengths = torch.randint(low=K, high=max_num_points, size=(num_pcs,), dtype=torch.long)
  lengths_cuda = lengths.cuda()

  def test_counting_sort_cuda(self):
    pc = Pointclouds(self.pc)
    bboxes = pc.get_bounding_boxes() 
    idxs_cuda, dists_cuda = frnn.test_find_nbrs_cuda(
    # grid_off, sorted_points = frnn.test_find_nbrs_cuda(
      bboxes, 
      self.pc_cuda, 
      self.pc_cuda, 
      self.lengths_cuda,
      self.lengths_cuda,
      self.K,
      self.r)

    idxs_cpu_bf, dists_cpu_bf = frnn.frnn_bf_cpu(
      self.pc,
      self.pc,
      self.lengths,
      self.lengths,
      self.K,
      self.r
    )
    idxs_cuda_bf, dists_cuda_bf = frnn.frnn_bf_cuda(
      self.pc_cuda,
      self.pc_cuda,
      self.lengths_cuda,
      self.lengths_cuda,
      self.K,
      self.r
    )
    print(idxs_cuda[0, :10])
    print(idxs_cuda_bf[0, :10])
    print(dists_cuda[0, :10])
    print(dists_cuda_bf[0, :10])
    print(idxs_cuda[1, :10])
    print(idxs_cuda_bf[1, :10])
    print(dists_cuda[1, :10])
    print(dists_cuda_bf[1, :10])
    print("cpu vs cuda idx bf: ", torch.allclose(idxs_cpu_bf, idxs_cuda_bf.cpu()))
    print("cpu vs cuda dists bf: ", torch.allclose(dists_cpu_bf, dists_cuda_bf.cpu()))

    print("idx same: ", torch.allclose(idxs_cuda, idxs_cuda_bf))
    print("idx same percentage: ", torch.sum(idxs_cuda == idxs_cuda_bf).type(torch.float).item() / self.K / self.max_num_points / self.num_pcs)
    print("dist same: ", torch.allclose(dists_cuda, dists_cuda_bf))


if __name__ == "__main__":
  unittest.main()