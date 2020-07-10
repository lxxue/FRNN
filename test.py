import torch
import FRNN
import unittest

class TestFRNN(unittest.TestCase):
    def test_frnn_bf_cpu(self):
        # randomly create a point clouds
        K = 5
        # a large enough dist so that FRNN should return the same result as 
        r2 = 1.0
        max_num_points = 1000
        num_pcs = 5
        pc = torch.rand((num_pcs, max_num_points, 3), dtype=torch.float)
        lengths = torch.randint(low=K, high=max_num_points, size=(num_pcs,), dtype=torch.long)
        knn_idxs, knn_dists = FRNN.knn_bf_cpu(pc, pc, lengths, lengths, K)
        frnn_idxs, frnn_dists = FRNN.frnn_bf_cpu(pc, pc, lengths, lengths, K, r2)
        # print(knn_idxs)
        # print(frnn_idxs)
        print("same idxs for large r2: ", torch.allclose(knn_idxs, frnn_idxs))
        print("same dists for large r2: ", torch.allclose(knn_dists, frnn_dists))

        r2 = 0.01
        frnn_idxs, frnn_dists = FRNN.frnn_bf_cpu(pc, pc, lengths, lengths, K, r2)
        mask = knn_dists >= r2
        knn_dists[mask] = -1.
        knn_idxs[mask] = -1
        print("same idxs for small r2: ", torch.allclose(knn_idxs, frnn_idxs))
        print("same dists for small r2: ", torch.allclose(knn_dists, frnn_dists))


if __name__ == "__main__":
    unittest.main()
