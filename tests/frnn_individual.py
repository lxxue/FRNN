import unittest
import glob

import torch
import frnn
from frnn import _C

from pytorch_points.utils.pc_utils import read_ply
from fvcore.common.benchmark import benchmark

from frnn_validation import normalize_pc

def save_intermediate_results(fnames):
  num_pcs = 1
  k = 8
  for fname in fnames:
    # Problem here: our algorithm is problematic when 
    # there is only one grid in one dimension 
    # the drill is a long thin object which brings some trouble
    if "drill_shaft" in fname:
      continue
    pc1 = torch.FloatTensor(read_ply(fname)[None, :, :3]).cuda()  # no need for normals
    pc2 = torch.FloatTensor(read_ply(fname)[None, :, :3]).cuda()  # no need for normals
    torch.save(pc1, "data/pc/"+fname.split('/')[-1][:-4]+'.pt')
    print(fname)
    print(pc1.shape)
    num_points = pc1.shape[1]
    pc1 = normalize_pc(pc1)
    print(pc1.min(dim=1)[0], pc1.max(dim=1)[0])
    pc2 = normalize_pc(pc2)
    lengths1 = torch.ones((num_pcs,), dtype=torch.long).cuda() * num_points
    lengths2 = torch.ones((num_pcs,), dtype=torch.long).cuda() * num_points
    dists, idxs, nn, grid = frnn.frnn_grid_points(pc1, pc2, lengths1, lengths2, K=k, r=0.1, filename=fname.split('/')[-1])
    print(fname+" done!")


class TestFRNN(unittest.TestCase):
  def steUp(self) -> None:
    super().setUp()
    torch.manual_seed(1)

  @staticmethod
  def frnn_setup_grid(N, fname):
    ragged = False
    r = 0.1
    MAX_RES = 100
    points2 = torch.load("data/pc/"+fname)
    if N > 1:
      points2 = points2.repeat(N, 1, 1)
    if ragged:
      lengths2 = torch.randint(low=1, high=points2.shape[1], size=(N,), dtype=torch.long, device=points2.device)
    else:
      lengths2 = torch.ones((N,), dtype=torch.long, device=points2.device) * points2.shape[1]
    GRID_PARAMS_SIZE = 8
    grid_params_cuda = torch.zeros((N, GRID_PARAMS_SIZE), dtype=torch.float, device=points2.device)
    torch.cuda.synchronize()

    def output():
      G = -1
      # print(points2.shape)
      # print(lengths2)
      # print(points2[0, :lengths2[0]].shape)
      # print(points2[0, :lengths2[0]].min(dim=0)[0].shape)
      for i in range(N):
        grid_min = points2[i, :lengths2[i]].min(dim=0)[0]
        grid_max = points2[i, :lengths2[i]].max(dim=0)[0]
        grid_params_cuda[i, :3] = grid_min
        grid_size = grid_max - grid_min
        cell_size = r
        if cell_size < grid_size.min() / MAX_RES:
          cell_size = grid_size.min() / MAX_RES
        grid_params_cuda[i, 3] = 1 / cell_size
        grid_params_cuda[i, 4:7] = torch.floor(grid_size / cell_size) + 1
        grid_params_cuda[i, 7] = grid_params_cuda[i, 4] * grid_params_cuda[i, 5] * grid_params_cuda[i, 6] 
        if G < grid_params_cuda[i, 7]:
          G = int(grid_params_cuda[i, 7].item())
      torch.cuda.synchronize()
    return output

  @staticmethod
  def frnn_insert_points(N, fname):
    ragged = False
    points1 = torch.load("data/pc/"+fname)
    points2 = torch.load("data/pc/"+fname)
    grid_params_cuda = torch.load("data/grid_params_cuda/"+fname)
    if N > 1:
      points1 = points1.repeat(N, 1, 1)
      points2 = points2.repeat(N, 1, 1)
      grid_params_cuda = grid_params_cuda.repeat(N, 1)
    if ragged:
      lengths1 = torch.randint(low=1, high=points1.shape[1], size=(N,), dtype=torch.long, device=points1.device)
      lengths2 = torch.randint(low=1, high=points2.shape[1], size=(N,), dtype=torch.long, device=points2.device)
    else:
      lengths1 = torch.ones((N,), dtype=torch.long, device=points1.device) * points1.shape[1]
      lengths2 = torch.ones((N,), dtype=torch.long, device=points2.device) * points2.shape[1]
 
    G = int(grid_params_cuda[:, 7].max().item())
    N = points1.shape[0]
    P1 = points1.shape[1]
    pc1_grid_cnt = torch.zeros((N, G), dtype=torch.int, device=points1.device)
    pc1_grid_cell = torch.full((N, P1), -1, dtype=torch.int, device=points1.device)
    pc1_grid_idx = torch.full((N, P1), -1, dtype=torch.int, device=points1.device)
    P2 = points2.shape[1]
    pc2_grid_cnt = torch.zeros((N, G), dtype=torch.int, device=points1.device)
    pc2_grid_cell = torch.full((N, P2), -1, dtype=torch.int, device=points1.device)
    pc2_grid_idx = torch.full((N, P2), -1, dtype=torch.int, device=points1.device)
    torch.cuda.synchronize()

    def output():
      _C.insert_points_cuda(points1, lengths1, grid_params_cuda, pc1_grid_cnt, pc1_grid_cell, pc1_grid_idx, G)
      _C.insert_points_cuda(points2, lengths2, grid_params_cuda, pc2_grid_cnt, pc2_grid_cell, pc2_grid_idx, G)
      torch.cuda.synchronize()
    return output
  
  @staticmethod
  def frnn_prefix_sum(N, fname):
    pc1_grid_cnt = torch.load("data/pc1_grid_cnt/"+fname)
    pc2_grid_cnt = torch.load("data/pc2_grid_cnt/"+fname)
    grid_params_cuda = torch.load("data/grid_params_cuda/"+fname)
    if N > 1:
      pc1_grid_cnt = pc1_grid_cnt.repeat(N, 1)
      pc2_grid_cnt = pc2_grid_cnt.repeat(N, 1)
      grid_params_cuda = grid_params_cuda.repeat(N, 1)

    torch.cuda.synchronize()
    def output():
      pc1_grid_off = _C.prefix_sum_cuda(pc1_grid_cnt, grid_params_cuda.cpu())
      pc2_grid_off = _C.prefix_sum_cuda(pc2_grid_cnt, grid_params_cuda.cpu())
      torch.cuda.synchronize()
    return output
  
  @staticmethod
  def frnn_counting_sort(N, fname):
    ragged = False
    points1 = torch.load("data/pc/"+fname)
    points2 = torch.load("data/pc/"+fname)
    pc1_grid_cell = torch.load("data/pc1_grid_cell/"+fname) 
    pc2_grid_cell = torch.load("data/pc2_grid_cell/"+fname) 
    pc1_grid_idx = torch.load("data/pc1_grid_idx/"+fname) 
    pc2_grid_idx = torch.load("data/pc2_grid_idx/"+fname) 
    pc1_grid_off = torch.load("data/pc1_grid_off/"+fname) 
    pc2_grid_off = torch.load("data/pc2_grid_off/"+fname) 
    if N > 1:
      points1 = points1.repeat(N, 1, 1)
      points2 = points2.repeat(N, 1, 1)
      pc1_grid_cell = pc1_grid_cell.repeat(N, 1)
      pc2_grid_cell = pc2_grid_cell.repeat(N, 1)
      pc1_grid_idx = pc1_grid_idx.repeat(N, 1)
      pc2_grid_idx = pc2_grid_idx.repeat(N, 1)
      pc1_grid_off = pc1_grid_off.repeat(N, 1)
      pc2_grid_off = pc2_grid_off.repeat(N, 1)
    if ragged:
      lengths1 = torch.randint(low=1, high=points1.shape[1], size=(N,), dtype=torch.long, device=points1.device)
      lengths2 = torch.randint(low=1, high=points2.shape[1], size=(N,), dtype=torch.long, device=points2.device)
    else:
      lengths1 = torch.ones((N,), dtype=torch.long, device=points1.device) * points1.shape[1]
      lengths2 = torch.ones((N,), dtype=torch.long, device=points2.device) * points2.shape[1]
    
    P1 = points1.shape[1]
    P2 = points2.shape[1]
    sorted_points1 = torch.zeros((N, P1, 3), dtype=torch.float, device=points1.device)
    sorted_points1_idxs = torch.full((N, P1), -1, dtype=torch.int, device=points1.device)
    sorted_points2 = torch.zeros((N, P2, 3), dtype=torch.float, device=points1.device)
    sorted_points2_idxs = torch.full((N, P2), -1, dtype=torch.int, device=points1.device)
    torch.cuda.synchronize()

    def output():
      _C.counting_sort_cuda(
        points1,
        lengths1,
        pc1_grid_cell,
        pc1_grid_idx,
        pc1_grid_off,
        sorted_points1,
        sorted_points1_idxs
      )
      _C.counting_sort_cuda(
        points2,
        lengths2,
        pc2_grid_cell,
        pc2_grid_idx,
        pc2_grid_off,
        sorted_points2,
        sorted_points2_idxs
      )
      torch.cuda.synchronize()

    return output

  @staticmethod
  def frnn_find_nbrs(N, fname, K):
    # print(N, fname, K)
    r = 0.1*torch.ones((N,), dtype=torch.float32, device=torch.device("cuda:0"))
    ragged = False
    
    sorted_points1 = torch.load("data/sorted_points1/"+fname) 
    sorted_points2 = torch.load("data/sorted_points2/"+fname) 
    sorted_points1_idxs = torch.load("data/sorted_points1_idxs/"+fname) 
    sorted_points2_idxs = torch.load("data/sorted_points2_idxs/"+fname) 
    pc2_grid_off = torch.load("data/pc2_grid_off/"+fname) 
    grid_params_cuda = torch.load("data/grid_params_cuda/"+fname)
    if N > 1:
      sorted_points1 = sorted_points1.repeat(N, 1, 1)
      sorted_points2 = sorted_points2.repeat(N, 1, 1)
      sorted_points1_idxs = sorted_points1_idxs.repeat(N, 1)
      sorted_points2_idxs = sorted_points2_idxs.repeat(N, 1)
      pc2_grid_off = pc2_grid_off.repeat(N, 1)
      grid_params_cuda = grid_params_cuda.repeat(N, 1)

    if ragged:
      lengths1 = torch.randint(low=1, high=points1.shape[1], size=(N,), dtype=torch.long, device=points1.device)
      lengths2 = torch.randint(low=1, high=points2.shape[1], size=(N,), dtype=torch.long, device=points2.device)
    else:
      lengths1 = torch.ones((N,), dtype=torch.long, device=sorted_points1.device) * sorted_points1.shape[1]
      lengths2 = torch.ones((N,), dtype=torch.long, device=sorted_points2.device) * sorted_points2.shape[1]
    torch.cuda.synchronize()

    def output():
      idxs, dists = _C.find_nbrs_cuda(
        sorted_points1,
        sorted_points2,
        lengths1,
        lengths2,
        pc2_grid_off,
        sorted_points1_idxs,
        sorted_points2_idxs,
        grid_params_cuda,
        K,
        r,
        r*r,
      )
      torch.cuda.synchronize()
    
    return output


if __name__ == "__main__":
  fnames = sorted(glob.glob('data/mesh/*.ply') + glob.glob('data/mesh/*/*.ply'))
  print(fnames)
  # fnames = ['data/mesh/lucy.ply']
  # fnames = ['data/mesh/drill/drill_shaft_zip.ply'] + fnames
  # save_intermediate_results(fnames)
  for fname in fnames:
    pc = torch.FloatTensor(read_ply(fname)[None, :, :3]).cuda()  # no need for normals
    pc -= pc.min(dim=1)[0]
    pc /= pc.max()
    print(pc.min(dim=1)[0], pc.max(dim=1)[0])
    torch.save(pc, "data/pc/"+fname.split('/')[-1][:-4]+'.pt')
