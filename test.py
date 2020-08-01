import torch
import FRNN.cpu
import FRNN.cuda
import unittest

class TestFRNN(unittest.TestCase):
    '''
    def test_frnn_bf_cpu(self):
        # randomly create a point clouds
        K = 5
        # a large enough dist so that FRNN should return the same result as 
        r2 = 1.0
        max_num_points = 1000
        num_pcs = 5
        pc = torch.rand((num_pcs, max_num_points, 3), dtype=torch.float)
        lengths = torch.randint(low=K, high=max_num_points, size=(num_pcs,), dtype=torch.long)
        knn_idxs, knn_dists = FRNN.cpu.knn_bf_cpu(pc, pc, lengths, lengths, K)
        frnn_idxs, frnn_dists = FRNN.cpu.frnn_bf_cpu(pc, pc, lengths, lengths, K, r2)
        # print(knn_idxs)
        # print(frnn_idxs)
        print("same idxs for large r2: ", torch.allclose(knn_idxs, frnn_idxs))
        print("same dists for large r2: ", torch.allclose(knn_dists, frnn_dists))

        r2 = 0.01
        frnn_idxs, frnn_dists = FRNN.cpu.frnn_bf_cpu(pc, pc, lengths, lengths, K, r2)
        mask = knn_dists >= r2
        knn_dists[mask] = -1.
        knn_idxs[mask] = -1
        print("same idxs for small r2: ", torch.allclose(knn_idxs, frnn_idxs))
        print("same dists for small r2: ", torch.allclose(knn_dists, frnn_dists))
    '''
    '''
    def test_frnn_bf_gpu(self):
        K = 5
        r2 = 0.01
        max_num_points = 1000
        num_pcs = 5
        pc = torch.rand((num_pcs, max_num_points, 3), dtype=torch.float)
        lengths = torch.randint(low=K, high=max_num_points, size=(num_pcs,), dtype=torch.long)
        frnn_idxs_cpu, frnn_dists_cpu = FRNN.cpu.frnn_bf_cpu(pc, pc, lengths, lengths, K, r2)
        pc = pc.cuda()
        lengths = lengths.cuda()
        frnn_idxs_gpu, frnn_dists_gpu = FRNN.cuda.frnn_bf_gpu(pc, pc, lengths, lengths, K, r2)
        # print(torch.sum(torch.eq(frnn_idxs_cpu, frnn_idxs_gpu.cpu())))
        # print(frnn_idxs_cpu[1][:5])
        # print(frnn_idxs_gpu[1][:5])
        # print(frnn_dists_cpu[1][:5])
        # print(frnn_dists_gpu[1][:5])
        print("same idxs for small r2: ", torch.allclose(frnn_idxs_cpu, frnn_idxs_gpu.cpu()))
        print("same dists for small r2: ", torch.allclose(frnn_dists_cpu, frnn_dists_gpu.cpu()))
    '''
    '''
    def test_frnn_grid_cpu(self):
        K = 5
        r2 = 0.01
        r = 0.1
        max_num_points = 50000
        num_pcs = 1
        pc = torch.rand((num_pcs, max_num_points, 3), dtype=torch.float)
        # pc[0, 0, 0] = 0
        # pc[0, 0, 1] = 0
        # pc[0, 0, 2] = 0
        # pc[0, 1, 0] = 1
        # pc[0, 1, 1] = 1
        # pc[0, 1, 2] = 1
        # lengths = torch.randint(low=K, high=max_num_points, size=(num_pcs,), dtype=torch.long)
        lengths = torch.LongTensor([max_num_points])
        frnn_idxs_cpu, frnn_dists_cpu = FRNN.cpu.frnn_bf_cpu(pc, pc, lengths, lengths, K, r2)
        idxs, dists = FRNN.cpu.grid_test(pc[0], K, r)
        print("same idxs: ", torch.allclose(frnn_idxs_cpu[0], idxs))
        print("same dists: ", torch.allclose(frnn_dists_cpu[0], dists))
        # print(idxs[:5])
        # print(frnn_idxs_cpu[0, :5])
        # print(dists[:5])
        # print(frnn_dists_cpu[0, :5])
    '''
    def test_frnn_grid_gpu(self):
        K = 5
        r = 0.1
        r2 = 0.01
        max_num_points = 1000
        num_pcs = 1
        pc = torch.rand((num_pcs, max_num_points, 3), dtype=torch.float)
        lengths = torch.LongTensor([max_num_points])
        print("cpu done")
        gridCnt_cpu = FRNN.cpu.grid_test(pc[0], K, r)
        bbox_max = torch.max(pc[0], 0)[0] 
        bbox_min = torch.min(pc[0], 0)[0]
        gridCnt_gpu = FRNN.cuda.grid_test_gpu(pc[0].cuda(), bbox_max, bbox_min, K, r)
        # gridCnt_gpu = FRNN.cuda.tensor_test(pc[0].cuda())
        print("gpu done")
        print(gridCnt_cpu.shape)
        print(gridCnt_gpu.shape)
        print(gridCnt_cpu.view(-1)[:20])
        print(gridCnt_gpu[:20])

        
        



if __name__ == "__main__":
    unittest.main()
