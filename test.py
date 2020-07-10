import torch
import FRNN
import unittest

class TestFRNN(unittest.TestCase):
    def test_frnn_bf_cpu():
        # randomly create a point clouds
        K = 5
        # a large enough dist so that FRNN should return the same result as 
        r2 = 1.0
        max_num_points = 10
        num_pcs = 1
        pc = torch.rand((num_pcs, max_num_points, 3), dtype=torch.float)
        lengths = torch.randint(low=K, high=max_num_points, size=(num_pcs,), dtype=torch.long)
        knn_idxs, knn_dists = FRNN.knn_bf_cpu(pc, pc, lengths, lengths, K)
        frnn_idxs, frnn_dists = FRNN.frnn_bf_cpu(pc, pc, lengths, lengths, K, r2)
        print(knn_idxs)
        print(frnn_idxs)
        print(torch.allclose(knn_idxs, frnn_idxs))
        print(torch.allclose(knn_dists, frnn_dists))


if __name__ == "__main__":
    unnitest.main()
