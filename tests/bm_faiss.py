from itertools import product
from fvcore.common.benchmark import benchmark
from faiss_whole import Compare

import glob

def bm_faiss_exact(fnames):
  Ns = [1, 2, 4, 8]
  Ks = [1, 2, 4, 8, 16, 32]
  test_cases = product(fnames, Ns, Ks)
  kwargs_list = []

  for case in test_cases:
    fname, N, K = case
    kwargs_list.append({"fname":fname.split("/")[-1], "N":N, "K":K})

  benchmark(Compare.faiss_exact, "faiss_exact", kwargs_list, warmup_iters=1)


def bm_faiss_approximate(fnames):
  Ns = [1, 2, 4, 8]
  Ks = [1, 2, 4, 8, 16]
  test_cases = product(fnames, Ns, Ks)
  kwargs_list = []

  for case in test_cases:
    fname, N, K = case
    kwargs_list.append({"fname":fname.split("/")[-1], "N":N, "K":K})

  benchmark(Compare.faiss_approximate, "faiss_approximate", kwargs_list, warmup_iters=1)

if __name__ == "__main__":
  fnames = sorted(glob.glob('data/mesh/*.ply'))
  # fnames = ['data/pc/random_1000000.pt']
  # bm_faiss_exact(fnames)
  bm_faiss_approximate(fnames)