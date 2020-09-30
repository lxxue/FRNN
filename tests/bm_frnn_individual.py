from itertools import product
from fvcore.common.benchmark import benchmark
from frnn_individual import TestFRNN 

import glob
import torch

from frnn_validation import normalize_pc

# all experiments done at search radius 0.1 (point cloud normalized)

def bm_frnn_setup_grid(fnames):
  Ns = [1, 2, 4, 8, 16]
  raggeds = [False, True]
  test_cases = product(Ns, fnames, raggeds)
  kwargs_list = []

  for case in test_cases:
    N, fname, ragged = case
    if 'lucy' in fname:
      continue
    kwargs_list.append({"N": N, "fname": fname.split("/")[-1], "ragged": ragged})
  
  benchmark(TestFRNN.frnn_setup_grid, "frnn setup grid", kwargs_list, warmup_iters=1)



if __name__ == "__main__":
  # fnames = sorted(glob.glob('data/mesh/*.ply') + glob.glob('data/mesh/*/*.ply'))
  fnames = sorted(glob.glob('data/pc/*.pt'))
  bm_frnn_setup_grid(fnames)

