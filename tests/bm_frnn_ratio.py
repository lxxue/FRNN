from itertools import product
from fvcore.common.benchmark import benchmark
from frnn_whole import Compare

import glob
import torch
import numpy as np
import csv

def bm_frnn_ratio(fnames):
  # Ns = [1]
  # Ks = [8]
  # ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 
  #           2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4.0]
  # Ns = [1, 2, 4, 8]
  Ns = [1]
  # Ks = [1, 2, 4, 8, 16]
  Ks = [8]
  # ratios = [1.0, 2.0, 3.0, 4.0]
  # ratios = [1.0]
  ratios = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 
            2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4.0]
  test_cases = product(fnames, Ns, Ks, ratios)
  kwargs_list = []

  for case in test_cases:
    fname, N, K, ratio = case
    kwargs_list.append({"fname":fname.split("/")[-1], "N":N, "K":K, "ratio":ratio})
  
  # here I modify the source code for benchmark to get outputs directly, then we can easily export it to csv
  outputs = benchmark(Compare.frnn, "frnn", kwargs_list, num_iters=5, warmup_iters=1)
  with open("tests/output/frnn_ratio_whole.csv", "w") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["fname"] + ratios)
    idx = 0
    for fname_i in range(len(fnames)):
      for N_i in range(len(Ns)):
        for K_i in range(len(Ks)):
          fname = fnames[fname_i].split('/')[-1] + '_' + str(Ns[N_i]) + '_' + str(Ks[K_i])
          row = [fname]
          for ratio_i in range(len(ratios)):
            row.append("{:.0f}".format(float(outputs[idx][1])))
            idx += 1
          writer.writerow(row)
  
      




if __name__ == "__main__":
  fnames = sorted(glob.glob('data/pc/*.pt'))
  bm_frnn_ratio(fnames)
