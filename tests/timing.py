import torch
import frnn
import unittest
from pytorch3d.structures import Pointclouds
from pytorch3d.ops.knn import knn_points

# from pytorch3d.io import load_ply
from pytorch_points.utils.pc_utils import read_ply

import time
import argparse

def TimeFindNbrs(num_pcs, num_points, fname=None):
  K = 5
  r = 0.1
  r2 = r * r
  if fname is None:
    pc1 = torch.rand((num_pcs, num_points, 3), dtype=torch.float)
    pc2 = torch.rand((num_pcs, num_points, 3), dtype=torch.float)
    for i in range(num_pcs):
      for j in range(3):
        pc1[i, :, j] *= torch.rand(1)+0.5
        pc2[i, :, j] *= torch.rand(1)+0.5
    lengths1 = torch.ones((num_pcs,), dtype=torch.long) * num_points
    lengths2 = torch.ones((num_pcs,), dtype=torch.long) * num_points
  pc1_cuda = pc1.cuda()
  pc2_cuda = pc2.cuda()
  lengths1_cuda = lengths1.cuda()
  lengths2_cuda = lengths2.cuda()

  pointcloud2 = Pointclouds(pc2)
  bboxes2 = pointcloud2.get_bounding_boxes() 

  # t1 = time.time()
  # idxs_cpu, dists_cpu = frnn.test_find_nbrs_cpu(
  #   bboxes2,
  #   pc1,
  #   pc2,
  #   lengths1,
  #   lengths2,
  #   K,
  #   r
  # )
  # t2 = time.time()
  start = torch.cuda.Event(enable_timing=True)
  end = torch.cuda.Event(enable_timing=True)
  start.record()
  idxs_cuda, dists_cuda = frnn.test_find_nbrs_cuda(
    bboxes2, 
    pc1_cuda, 
    pc2_cuda, 
    lengths1_cuda,
    lengths2_cuda,
    K,
    r
  )
  end.record()
  torch.cuda.synchronize()
  grid_time = start.elapsed_time(end)


  
  start = torch.cuda.Event(enable_timing=True)
  end = torch.cuda.Event(enable_timing=True)
  start.record()
  knn_results = knn_points(
    pc1_cuda, 
    pc2_cuda, 
    lengths1_cuda,
    lengths2_cuda,
    K
  )
  end.record()
  torch.cuda.synchronize()
  knn_time = start.elapsed_time(end)

  start = torch.cuda.Event(enable_timing=True)
  end = torch.cuda.Event(enable_timing=True)
  start.record()
  idxs_cuda_bf, dists_cuda_bf = frnn.frnn_bf_cuda(
    pc1_cuda,
    pc2_cuda,
    lengths1_cuda,
    lengths2_cuda,
    K,
    r
  )
  end.record()
  torch.cuda.synchronize()
  bf_time = start.elapsed_time(end)



  # t4 = time.time()

  # print("cpu vs cuda idx grid: ", torch.allclose(idxs_cpu, idxs_cuda.cpu()))
  # print("cpu vs cuda dists grid: ", torch.allclose(dists_cpu, dists_cuda.cpu()))
  # print("grid vs bf dists cuda: ", torch.allclose(dists_cuda, dists_cuda_bf))
  
  # print("time grid cpu {:.8f}s".format(t2 - t1))
  # print("time bf cuda {:.8f}s".format(t3 - t2))
  # print("time grid cuda {:.8f}s".format(t4 - t3))
  print("#pcs: {:d}; #points {:d};".format(num_pcs, num_points))
  print("\tgrid vs bf # diff keys: ", torch.sum(idxs_cuda != idxs_cuda_bf).item())
  print("\tknn time: {:.2f}; bf time: {:.2f}; grid time: {:.2f}".format(knn_time, bf_time, grid_time))


if __name__ == "__main__":
  # parser = argparse.ArgumentParser()
  # parser.add_argument("--pc", type=str, default=None)
  # args = parser.parse_args()
  TimeFindNbrs(1, 10000, args.pc)
  TimeFindNbrs(1, 100000, args.pc)
  TimeFindNbrs(1, 1000000, args.pc)
  TimeFindNbrs(10, 10000, args.pc)
  TimeFindNbrs(10, 100000, args.pc)
  TimeFindNbrs(10, 1000000, args.pc)
  
  