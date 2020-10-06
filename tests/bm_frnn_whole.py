from itertools import product
from fvcore.common.benchmark import benchmark
from frnn_whole import Compare

import glob
import torch

def bm_frnn(fnames):
  Ns = [1, 2, 4, 8]
  Ks = [1, 2, 4, 8, 16] #, 32]
  # Ns = [8]
  # Ks = [32]
  # Ks = [16]
  test_cases = product(fnames, Ns, Ks)
  kwargs_list = []

  for case in test_cases:
    fname, N, K = case
    kwargs_list.append({"fname":fname.split("/")[-1], "N":N, "K":K})
  
  benchmark(Compare.frnn, "frnn", kwargs_list, num_iters=5, warmup_iters=1)

def bm_knn(fnames):
  Ns = [1, 2, 4, 8]
  Ks = [1, 2, 4, 8, 16]
  # Ns = [8]
  # Ks = [32]
  test_cases = product(fnames, Ns, Ks)
  kwargs_list = []

  for case in test_cases:
    fname, N, K = case
    kwargs_list.append({"fname":fname.split("/")[-1], "N":N, "K":K})
  
  benchmark(Compare.knn, "knn", kwargs_list, num_iters=5, warmup_iters=1)

if __name__ == "__main__":
  fnames = sorted(glob.glob('data/pc/*.pt'))
  # fnames = ['data/pc/random_1000000.pt']
  # for fname in fnames:
    # pc = torch.load(fname)
    # pc = pc - pc.min(axis=1)[0]
    # print(pc.max())
    # pc /= pc.max()
    # print(pc.max(axis=1)[0], pc.min(axis=1)[0])
    # torch.save(pc, fname)
  bm_frnn(fnames)
  bm_knn(fnames)
