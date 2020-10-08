from itertools import product
from fvcore.common.benchmark import benchmark
from frnn_individual import TestFRNN 

import glob
import torch
import csv

from frnn_validation import normalize_pc

# all experiments done at search radius 0.1 (point cloud normalized)

def bm_frnn_setup_grid(fnames):
  Ns = [1, 2, 4, 8] #, 16]
  # raggeds = [False, True]
  test_cases = product(fnames, Ns)
  kwargs_list = []

  for case in test_cases:
    fname, N = case
    kwargs_list.append({"N": N, "fname": fname.split("/")[-1]})
  
  outputs = benchmark(TestFRNN.frnn_setup_grid, "setup_grid_params", kwargs_list, num_iters=5, warmup_iters=1)
  return outputs

def bm_frnn_insert_points(fnames):
  Ns = [1, 2, 4, 8] #, 16]
  test_cases = product(fnames, Ns)

  kwargs_list = []
  for case in test_cases:
    fname, N = case
    kwargs_list.append({"N": N, "fname": fname.split("/")[-1]})

  outputs = benchmark(TestFRNN.frnn_insert_points, "insert_points", kwargs_list, num_iters=5, warmup_iters=1)
  return outputs

def bm_frnn_prefix_sum(fnames):
  Ns = [1, 2, 4, 8]# , 16]
  test_cases = product(fnames, Ns)

  kwargs_list = []
  for case in test_cases:
    fname, N = case
    kwargs_list.append({"N": N, "fname": fname.split("/")[-1]})

  outputs = benchmark(TestFRNN.frnn_prefix_sum, "prefix_sum", kwargs_list, num_iters=5, warmup_iters=1)
  return outputs

def bm_frnn_counting_sort(fnames):
  Ns = [1, 2, 4, 8] # , 16]
  test_cases = product(fnames, Ns)

  kwargs_list = []
  for case in test_cases:
    fname, N = case
    kwargs_list.append({"N": N, "fname": fname.split("/")[-1]})

  outputs = benchmark(TestFRNN.frnn_counting_sort, "counting_sort", kwargs_list, num_iters=5, warmup_iters=1)
  return outputs

def bm_frnn_find_nbrs(fnames):
  # Ns = [1, 2, 4, 8, 16]
  Ns = [1, 2, 4, 8] #, 16]
  # Ks = [1, 2, 4, 8, 16, 32]
  Ks = [8]
  test_cases = product(fnames, Ns, Ks)

  kwargs_list = []
  for case in test_cases:
    fname, N, K = case
    kwargs_list.append({"N": N, "fname": fname.split("/")[-1], "K":K})

  outputs = benchmark(TestFRNN.frnn_find_nbrs, "find_nbrs", kwargs_list, num_iters=5, warmup_iters=1)
  return outputs





if __name__ == "__main__":
  # fnames = sorted(glob.glob('data/mesh/*.ply') + glob.glob('data/mesh/*/*.ply'))
  fnames = sorted(glob.glob('data/pc/*.pt'))
  # fnames = ['random_10000.pt']
  outputs_1 = bm_frnn_setup_grid(fnames)
  outputs_2 = bm_frnn_insert_points(fnames)
  outputs_3 = bm_frnn_prefix_sum(fnames)
  outputs_4 = bm_frnn_counting_sort(fnames)
  outputs_5 = bm_frnn_find_nbrs(fnames)

  Ns = [1, 2, 4, 6]
  assert(len(outputs_1) == len(fnames)*len(Ns))

  with open("tests/output/frnn_individual.csv", "w") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["fname", "setup", "insert", "prefix_sum", "sort", "search"])
    for i in range(len(fnames)*len(Ns)):
      row = [outputs_5[i][0]]
      row.append("{:.0f}".format(float(outputs_1[i][1])))
      row.append("{:.0f}".format(float(outputs_2[i][1])))
      row.append("{:.0f}".format(float(outputs_3[i][1])))
      row.append("{:.0f}".format(float(outputs_4[i][1])))
      row.append("{:.0f}".format(float(outputs_5[i][1])))
      writer.writerow(row)



