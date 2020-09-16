import glob
import csv

import torch
import frnn
from pytorch3d.ops.knn import knn_gather, knn_points
from pytorch_points.utils.pc_utils import read_ply

num_points_fixed_query = 100000

class TestFRNN:
  def __init__(self, fname, num_pcs=1, K=5, r=0.1, same=False):
    if 'random' in fname:
      # fname format: random_{num_points}
      num_points = int(fname.split('_')[1])
      pc1 = torch.rand((num_pcs, num_points, 3), dtype=torch.float)
      pc2 = torch.rand((num_pcs, num_points, 3), dtype=torch.float)
      for i in range(num_pcs):
        for j in range(3):
          pc1[i, :, j] *= torch.rand(1)+0.5
          pc2[i, :, j] *= torch.rand(1)+0.5
    else:
      pc1 = torch.FloatTensor(read_ply(fname)[None, :, :3])  # no need for normals
      # pc2 = pc1
      pc2 = torch.FloatTensor(read_ply(fname)[None, :, :3])  # no need for normals
      normalize_pc(pc1)
      normalize_pc(pc2)
      # print("pc1 bbox: ", pc1.min(dim=1)[0], pc1.max(dim=1)[0])
      num_points = pc2.shape[1]
      if num_pcs > 1:
        pc1 = pc1.repeat(num_pcs, 1, 1)
        pc2 = pc2.repeat(num_pcs, 1, 1)

    if not same:
      pc1 = torch.rand((num_pcs, num_points_fixed_query, 3), dtype=torch.float)
    self.num_pcs = num_pcs
    self.fname = fname.split('/')[-1]
    self.K = K
    self.r = r
    self.num_points = num_points
    self.same = same
    self.pc1_knn = pc1.clone().detach().cuda()
    self.pc2_knn = pc2.clone().detach().cuda()
    self.pc1_frnn = pc1.clone().detach().cuda()
    self.pc2_frnn = pc2.clone().detach().cuda()
    self.pc1_frnn_reuse = pc1.clone().detach().cuda()
    self.pc2_frnn_reuse = pc2.clone().detach().cuda()
    # self.pc1_knn.requires_grad_(True)
    # self.pc2_knn.requires_grad_(True)
    # self.pc1_frnn.requires_grad_(True)
    # self.pc2_frnn.requires_grad_(True)
    # self.pc1_frnn_reuse.requires_grad_(True)
    # self.pc2_frnn_reuse.requires_grad_(True)
    if same:
      lengths1 = torch.ones((num_pcs,), dtype=torch.long) * num_points
    else:
      lengths1 = torch.ones((num_pcs,), dtype=torch.long) * num_points_fixed_query 
    lengths2 = torch.ones((num_pcs,), dtype=torch.long) * num_points
    self.lengths1_cuda = lengths1.cuda()
    self.lengths2_cuda = lengths2.cuda()
    print("{}: #points: {}".format(self.fname, num_points))
    self.grid = None
    self.grad_dists = torch.ones((num_pcs, pc1.shape[0], K), dtype=torch.float32).cuda()

  def frnn_grid(self):
    dists, idxs, nn, grid = frnn.frnn_grid_points(
      self.pc1_frnn,
      self.pc2_frnn,
      self.lengths1_cuda,
      self.lengths2_cuda,
      K=self.K,
      r=self.r,
      grid=None,
      return_nn=True,
      return_sorted=True
    )
    if self.grid is None:
      self.grid = grid
    return dists, idxs, nn

  def frnn_grid_reuse(self):
    dists, idxs, nn, _ = frnn.frnn_grid_points(
      self.pc1_frnn_reuse,
      self.pc2_frnn_reuse,
      self.lengths1_cuda,
      self.lengths2_cuda,
      K=self.K,
      r=self.r,
      grid=self.grid,
      return_nn=True,
      return_sorted=True
    )
    return dists, idxs, nn 
  
  def knn(self):
    dists, idxs, nn = knn_points(
      self.pc1_knn,
      self.pc2_knn,
      self.lengths1_cuda,
      self.lengths2_cuda,
      K=self.K,
      version=-1,
      return_nn=True,
      return_sorted=True
    ) 
    # for backward, assume all we have k neighbors within the radius
    mask = dists > self.r * self.r
    idxs[mask] = -1
    dists[mask] = -1
    nn[mask] = 0.

    # print(dists.shape)
    # print(mask.shape)
    # print(idxs.shape)
    return dists, idxs, nn

  # def frnn_bf(self):
  #   idxs, dists = frnn._C.frnn_bf_cuda(
  #     self.pc1_cuda,
  #     self.pc2_cuda,
  #     self.lengths1_cuda,
  #     self.lengths2_cuda,
  #     self.K,
  #     self.r
  #   )
  #   return dists, idxs

  def compare_frnn_knn(self):
    if self.num_points > 10000000:
      print("\tnumber of points for exceed 1 million; skip")
      return None

    # forward    
    dists_knn, idxs_knn, nn_knn = self.knn()
    dists_frnn, idxs_frnn, nn_frnn = self.frnn_grid()
    dists_frnn_reuse, idxs_frnn_reuse, nn_frnn_reuse = self.frnn_grid_reuse()
    # dists_frnn_bf, idxs_frnn_bf = self.frnn_bf()

    # backward
    # loss_knn = (dists_knn * self.grad_dists).sum()
    # loss_knn.backward()
    # loss_frnn = (dists_frnn * self.grad_dists).sum()
    # loss_frnn.backward()
    # loss_frnn_reuse = (dists_frnn_reuse * self.grad_dists).sum()
    # loss_frnn_reuse.backward()

    # for i in range(self.pc2_knn.shape[1]):
    #   if not torch.allclose(self.pc2_knn.grad[0, i], self.pc2_frnn.grad[0, i]):
    #     print(self.pc2_knn.grad[0, i])
    #     print(self.pc2_frnn.grad[0, i])

    idxs_all_same = torch.all(idxs_frnn == idxs_knn).item()
    idxs_all_same_reuse = torch.all(idxs_frnn_reuse == idxs_knn).item()
    diff_keys_percentage = torch.sum(idxs_frnn == idxs_knn).type(torch.float).item() / self.K / self.pc1_knn.shape[1] / self.num_pcs
    diff_keys_percentage_reuse = torch.sum(idxs_frnn_reuse == idxs_knn).type(torch.float).item() / self.K / self.pc1_knn.shape[1] / self.num_pcs
    dists_all_close = torch.allclose(dists_frnn, dists_knn)
    dists_all_close_reuse = torch.allclose(dists_frnn_reuse, dists_knn)
    nn_all_close = torch.allclose(nn_frnn, nn_knn)
    nn_all_close_reuse = torch.allclose(nn_frnn_reuse, nn_knn)
    return [self.fname, self.num_points, idxs_all_same, idxs_all_same_reuse, 
            "{:.4f}".format(diff_keys_percentage), "{:.4f}".format(diff_keys_percentage_reuse),
            dists_all_close, dists_all_close_reuse, nn_all_close, nn_all_close_reuse]
    # pc1_grad_all_close = torch.allclose(self.pc1_frnn.grad, self.pc1_knn.grad, atol=5e-6)
    # # pc1_grad_all_close_reuse = torch.allclose(self.pc1_frnn_reuse.grad, self.pc1_knn.grad)
    # pc1_grad_all_close_reuse = True
    # pc2_grad_all_close = torch.allclose(self.pc2_frnn.grad, self.pc2_knn.grad, atol=5e-6)
    # # pc2_grad_all_close_reuse = torch.allclose(self.pc2_frnn_reuse.grad, self.pc2_knn.grad)
    # pc2_grad_all_close_reuse = True
    # return [self.fname, self.num_points, idxs_all_same, idxs_all_same_reuse, 
    #         "{:.4f}".format(diff_keys_percentage), "{:.4f}".format(diff_keys_percentage_reuse),
    #         dists_all_close, dists_all_close_reuse, nn_all_close, nn_all_close_reuse, pc1_grad_all_close,
    #         pc1_grad_all_close_reuse, pc2_grad_all_close, pc2_grad_all_close_reuse]

  # def compare_frnnreuse_knn(self):
  #   if self.num_points > 10000000:
  #     print("\tnumber of points for exceed 1 million; skip")
  #     return None
  #   dists_knn, idxs_knn, nn_knn = self.knn()
  #   dists_frnn, idxs_frnn, nn_frnn = self.frnn_grid_reuse()

  #   diff_keys_percentage = torch.sum(idxs_frnn == idxs_knn).type(torch.float).item() / self.K / self.pc1_cuda.shape[1] / self.num_pcs
  #   dists_all_close = torch.allclose(dists_frnn, dists_knn)
  #   nn_all_close = torch.allclose(nn_frnn, nn_knn)
  #   return [self.fname, self.num_points, "{:.4f}".format(diff_keys_percentage), dists_all_close, nn_all_close]

def normalize_pc(pc):
  # convert pc to the unit box so that we don't need to manually set raidus for each mesh
  # pc should be 1 x P x 3
  # [0, 1] x [0, 1] x [0, 1]
  assert pc.shape[0] == 1 and pc.shape[2] == 3
  pc -= pc.min(dim=1)[0]
  pc /= torch.max(pc)
  # print(pc.min(dim=1), pc.max(dim=1))
  return


if __name__ == "__main__":
  fnames = sorted(glob.glob('data/*.ply') + glob.glob('data/*/*.ply'))
  fnames += ['random_10000', 'random_100000', 'random_1000000']
  print(fnames)
  with open("tests/output/frnn_validation.csv", 'w') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Point cloud', 'Num points', 'Different key percentage', 'Dists all close'])
    for fname in fnames:
      if 'xyz' in fname or 'lucy' in fname:
        continue
      validator = TestFRNN(fname, same=False)
      results = validator.compare_frnn_knn()
      print(results)
      writer.writerow(results)
      # results = validator.compare_frnnreuse_knn()
      # print(results)
      # writer.writerow(results)

  with open("tests/output/frnn_validation_same.csv", 'w') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Point cloud', 'Num points', 'Different key percentage', 'Dists all close'])
    for fname in fnames:
      if 'xyz' in fname or 'lucy' in fname:
        continue
      validator = TestFRNN(fname, same=True)
      results = validator.compare_frnn_knn()
      print(results)
      writer.writerow(results)
      # results = validator.compare_frnnreuse_knn()
      # print(results)
      # writer.writerow(results)
