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

  pc1 = torch.rand((num_pcs, max_num_points, 3), dtype=torch.float)
  pc2 = torch.rand((num_pcs, max_num_points, 3), dtype=torch.float)
  for i in range(num_pcs):
    for j in range(3):
      pc1[i, :, j] *= torch.rand(1)+0.5
      pc2[i, :, j] *= torch.rand(1)+0.5
  pc1_cuda = pc1.cuda()
  pc2_cuda = pc2.cuda()
  lengths1 = torch.randint(low=K, high=max_num_points, size=(num_pcs,), dtype=torch.long)
  lengths2 = torch.randint(low=K, high=max_num_points, size=(num_pcs,), dtype=torch.long)
  lengths1_cuda = lengths1.cuda()
  lengths2_cuda = lengths2.cuda()

  def test_find_nbrs_cuda(self):
    pc2 = Pointclouds(self.pc2)
    bboxes2 = pc2.get_bounding_boxes() 

    idxs_cpu, dists_cpu = frnn.test_find_nbrs_cpu(
      bboxes2,
      self.pc1,
      self.pc2,
      self.lengths1,
      self.lengths2,
      self.K,
      self.r
    )

    idxs_cuda, dists_cuda = frnn.test_find_nbrs_cuda(
      bboxes2, 
      self.pc1_cuda, 
      self.pc2_cuda, 
      self.lengths1_cuda,
      self.lengths2_cuda,
      self.K,
      self.r
    )

    idxs_cpu_bf, dists_cpu_bf = frnn.frnn_bf_cpu(
      self.pc1,
      self.pc2,
      self.lengths1,
      self.lengths2,
      self.K,
      self.r
    )

    idxs_cuda_bf, dists_cuda_bf = frnn.frnn_bf_cuda(
      self.pc1_cuda,
      self.pc2_cuda,
      self.lengths1_cuda,
      self.lengths2_cuda,
      self.K,
      self.r
    )
    print(idxs_cuda[0, :10])
    print(idxs_cpu[0, :10])
    print(dists_cuda[0, :10])
    print(dists_cpu[0, :10])
    # print(idxs_cuda_bf[0, :10])
    # print(idxs_cuda_bf[0, :10])
    # print(idxs_cuda[0, :10])
    # print(idxs_cuda_bf[0, :10])
    # print(dists_cuda[0, :10])
    # print(dists_cuda_bf[0, :10])
    # print(idxs_cuda[1, :10])
    # print(idxs_cuda_bf[1, :10])
    # print(dists_cuda[1, :10])
    # print(dists_cuda_bf[1, :10])
    print("cpu vs cuda idx bf: ", torch.allclose(idxs_cpu_bf, idxs_cuda_bf.cpu()))
    print("cpu vs cuda dists bf: ", torch.allclose(dists_cpu_bf, dists_cuda_bf.cpu()))

    print("cpu vs cuda idx grid: ", torch.allclose(idxs_cpu, idxs_cuda.cpu()))
    print("cpu vs cuda dists grid: ", torch.allclose(dists_cpu, dists_cuda.cpu()))

    print("grid vs bf idx cuda: ", torch.allclose(idxs_cuda, idxs_cuda_bf))
    print("grid vs bf dists cuda: ", torch.allclose(dists_cuda, dists_cuda_bf))
    print("idx same percentage: ", torch.sum(idxs_cuda == idxs_cuda_bf).type(torch.float).item() / self.K / self.max_num_points / self.num_pcs)
    print("dists same percentage: ", torch.sum(dists_cuda == dists_cuda_bf).type(torch.float).item() / self.K / self.max_num_points / self.num_pcs)


if __name__ == "__main__":
  unittest.main()