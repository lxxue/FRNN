import unittest
import glob

import torch
import frnn

from pytorch_points.utils.pc_utils import read_ply

def save_intermediate_results(fnames):
  num_pcs = 1
  k = 8
  for fname in fnames:
    if "drill_shaft" in fname:
      continue
    print(fname)
    pc1 = torch.FloatTensor(read_ply(fname)[None, :, :3]).cuda()  # no need for normals
    pc2 = torch.FloatTensor(read_ply(fname)[None, :, :3]).cuda()  # no need for normals
    print(pc1.shape)
    num_points = pc1.shape[1]
    normalize_pc(pc1)
    normalize_pc(pc2)
    lengths1 = torch.ones((num_pcs,), dtype=torch.long).cuda() * num_points
    lengths2 = torch.ones((num_pcs,), dtype=torch.long).cuda() * num_points
    dists, idxs, nn, grid = frnn.frnn_grid_points(pc1, pc2, lengths1, lengths2, K=k, r=0.1, filename=fname.split('/')[-1])
    print(fname+" done!")

def normalize_pc(pc):
  # convert pc to the unit box so that we don't need to manually set raidus for each mesh
  # pc should be 1 x P x 3
  # [0, 1] x [0, 1] x [0, 1]
  assert pc.shape[0] == 1 and pc.shape[2] == 3
  pc = pc - torch.min(pc, dim=1)[0]
  pc /= torch.max(pc)
  # print(pc.min(dim=1), pc.max(dim=1))
  return




if __name__ == "__main__":
  fnames = sorted(glob.glob('data/mesh/*.ply') + glob.glob('data/mesh/*/*.ply'))
  # fnames = ['data/mesh/drill/drill_shaft_zip.ply'] + fnames
  save_intermediate_results(fnames)