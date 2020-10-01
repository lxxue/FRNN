from itertools import product
from fvcore.common.benchmark import benchmark
from frnn_individual import TestFRNN 

import glob
import torch

from frnn_validation import normalize_pc

# all experiments done at search radius 0.1 (point cloud normalized)

def bm_frnn_setup_grid(fnames):
  Ns = [1, 2, 4, 8, 16]
  # raggeds = [False, True]
  test_cases = product(fnames, Ns)
  kwargs_list = []

  for case in test_cases:
    fname, N = case
    kwargs_list.append({"N": N, "fname": fname.split("/")[-1]})
  
  benchmark(TestFRNN.frnn_setup_grid, "setup_grid_params", kwargs_list, warmup_iters=1)

def bm_frnn_insert_points(fnames):
  Ns = [1, 2, 4, 8, 16]
  test_cases = product(fnames, Ns)

  kwargs_list = []
  for case in test_cases:
    fname, N = case
    kwargs_list.append({"N": N, "fname": fname.split("/")[-1]})

  benchmark(TestFRNN.frnn_insert_points, "insert_points", kwargs_list, warmup_iters=1)

def bm_frnn_prefix_sum(fnames):
  Ns = [1, 2, 4, 8, 16]
  test_cases = product(fnames, Ns)

  kwargs_list = []
  for case in test_cases:
    fname, N = case
    kwargs_list.append({"N": N, "fname": fname.split("/")[-1]})

  benchmark(TestFRNN.frnn_prefix_sum, "prefix_sum", kwargs_list, warmup_iters=1)

def bm_frnn_counting_sort(fnames):
  Ns = [1, 2, 4, 8, 16]
  test_cases = product(fnames, Ns)

  kwargs_list = []
  for case in test_cases:
    fname, N = case
    kwargs_list.append({"N": N, "fname": fname.split("/")[-1]})

  benchmark(TestFRNN.frnn_counting_sort, "counting_sort", kwargs_list, warmup_iters=1)

def bm_frnn_find_nbrs(fnames):
  # Ns = [1, 2, 4, 8, 16]
  Ns = [1, 2, 4, 8]
  Ks = [1, 2, 4, 8, 16, 32]
  test_cases = product(fnames, Ns, Ks)

  kwargs_list = []
  for case in test_cases:
    fname, N, K = case
    kwargs_list.append({"N": N, "fname": fname.split("/")[-1], "K":K})

  benchmark(TestFRNN.frnn_find_nbrs, "find_nbrs", kwargs_list, warmup_iters=1)





if __name__ == "__main__":
  # fnames = sorted(glob.glob('data/mesh/*.ply') + glob.glob('data/mesh/*/*.ply'))
  fnames = sorted(glob.glob('data/pc/*.pt'))
  # bm_frnn_setup_grid(fnames)
  # bm_frnn_insert_points(fnames)
  # bm_frnn_prefix_sum(fnames)
  # bm_frnn_counting_sort(fnames)
  # fnames = ['random_1000000.pt']
  bm_frnn_find_nbrs(fnames)


