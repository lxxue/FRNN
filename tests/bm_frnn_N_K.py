from itertools import product
from fvcore.common.benchmark import benchmark
from frnn_whole import Compare

import glob
import torch
import csv
import numpy as np

def bm_frnn_N(fnames):
  Ns = [1, 2, 3, 4, 5, 6, 7, 8]
  Ks = [8]
  test_cases = product(fnames, Ns, Ks)
  kwargs_list = []

  for case in test_cases:
    fname, N, K = case
    kwargs_list.append({"fname":fname.split("/")[-1], "N":N, "K":K, "ratio":2.0})
  
  outputs = benchmark(Compare.frnn, "frnn", kwargs_list, num_iters=5, warmup_iters=1)
  with open("tests/output/frnn_N.csv", "w") as csvfile:
    writer = csv.writer(csvfile)
    idx = 0
    writer.writerow(["N", 1,2,3,4,5,6,7,8])
    for fname_i in range(len(fnames)):
      row = [fnames[fname_i]]
      for N_i in range(len(Ns)):
        # fname = fnames[fname_i].split('/')[-1] + '_' + str(Ns[N_i]) + '_' + str(Ks[K_i])
        row.append("{:.0f}".format(float(outputs[idx][1])))
        idx += 1
      writer.writerow(row)

def bm_frnn_K(fnames):
  Ns = [1]
  Ks = np.arange(1, 33, dtype=np.int)
  test_cases = product(fnames, Ns, Ks)
  kwargs_list = []

  for case in test_cases:
    fname, N, K = case
    kwargs_list.append({"fname":fname.split("/")[-1], "N":N, "K":K, "ratio":2.0})
  
  outputs = benchmark(Compare.frnn, "frnn", kwargs_list, num_iters=5, warmup_iters=1)
  with open("tests/output/frnn_K.csv", "w") as csvfile:
    writer = csv.writer(csvfile)
    idx = 0
    writer.writerow(["K"]+[i for i in Ks])
    for fname_i in range(len(fnames)):
      row = [fnames[fname_i]]
      for N_i in range(len(Ks)):
        # fname = fnames[fname_i].split('/')[-1] + '_' + str(Ns[N_i]) + '_' + str(Ks[K_i])
        row.append("{:.0f}".format(float(outputs[idx][1])))
        idx += 1
      writer.writerow(row)


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
  # fnames = fnames[0:1]
  bm_frnn_N(fnames)
  bm_frnn_K(fnames)
