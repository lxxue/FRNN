import torch
import frnn
from pytorch_points.utils.pc_utils import read_ply

if __name__ == "__main__":
  gpu = torch.device("cuda:0")
  pc = torch.cuda.FloatTensor(read_ply("drill/drill_shaft_vrip.ply")[None, ...])
  # print(pc.shape)
  # print(pc.min(dim=1)[0], pc.max(dim=1)[0])
  # pc -= pc.min(dim=1)[0]
  # pc /= pc.max()
  # print(pc.min(dim=1)[0], pc.max(dim=1)[0])
  lengths = torch.ones((1,), dtype=torch.long, device=gpu) * pc.shape[1]
  # print(lengths)
  dists, idxs, nn, grid = frnn.frnn_grid_points(pc, pc, lengths, lengths, 8, 0.1, None, False, True, 2.0)
  print(dists)
  print(idxs)
  
  