import faiss
import unittest
import torch
from pytorch_points.utils.pc_utils import read_ply
import numpy as np

class Compare(unittest.TestCase):
  @staticmethod
  def faiss_exact(N, fname, K):
    print(fname, N, K)
    # points1 = torch.load("data/pc/"+fname).cpu().numpy()
    # points2 = torch.load("data/pc/"+fname).cpu().numpy()
    points1 = np.ascontiguousarray(read_ply("data/mesh/"+fname))
    points2 = np.ascontiguousarray(read_ply("data/mesh/"+fname))
    res = faiss.StandardGpuResources()
    flat_config = faiss.GpuIndexFlatConfig()
    flat_config.device = 0
    index = faiss.GpuIndexFlatL2(res, 3, flat_config)
    index.add(points2)
    torch.cuda.synchronize()

    def output():
      for i in range(N):
        D, I = index.search(points1, K)
      torch.cuda.synchronize()
    
    return output

  @staticmethod 
  def faiss_approximate(N, fname, K):
    print(fname, N, K)
    res = faiss.StandardGpuResources()
    points1 = np.ascontiguousarray(read_ply("data/mesh/"+fname))
    points2 = np.ascontiguousarray(read_ply("data/mesh/"+fname))
    # index = faiss.index_factory(3, "IVF4096, PQ64")
    # index = faiss.index_factory(3, "IVF4096, Flat")
    index = faiss.index_factory(3, "IVF4096, Flat") 
    co = faiss.GpuClonerOptions
    co.useFloat16 = True
    index = faiss.index_cpu_to_gpu(res, 0, index) 
    torch.cuda.synchronize()
    def output():
      for i in range(N):
        index.train(points2)
        D, I = index.search(points1, K)
      torch.cuda.synchronize()
    return output
