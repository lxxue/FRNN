import torch
import FRNN.cpu
import FRNN.cuda
import unittest
import time

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
    '''
    '''
    def test_prefix_sum(self):
        GridCnt = torch.randint(low=0, high=1000, size=(100000,), device=torch.device('cuda:0'), dtype=torch.int)
        cumsum = torch.cumsum(GridCnt, dim=0, dtype=torch.int)
        gt = torch.zeros_like(cumsum)
        gt[1:] = cumsum[:-1]
        print(GridCnt.cpu()[:10])
        print(gt.cpu()[:10])
        GridOff = FRNN.cuda.prefix_sum(GridCnt)
        print(GridOff[:10])
        print(torch.allclose(GridOff, gt))
    '''
    '''
    def test_counting_sort(self):
        K = 5
        r = 0.1
        r2 = 0.01
        max_num_points = 1000
        num_pcs = 1
        pc = torch.rand((num_pcs, max_num_points, 3), dtype=torch.float)
        pc[0, 0, 0] = 0
        pc[0, 0, 1] = 0
        pc[0, 0, 2] = 0
        pc[0, 1, 0] = 1
        pc[0, 1, 1] = 1
        pc[0, 1, 2] = 1
        # lengths = torch.LongTensor([max_num_points])
        # print("cpu done")
        # gridCnt_cpu = FRNN.cpu.grid_test(pc[0], K, r)
        bbox_max = torch.max(pc[0], 0)[0] 
        bbox_min = torch.min(pc[0], 0)[0]
        SortedPoints, SortedGridCell = FRNN.cuda.grid_test_gpu(pc[0].cuda(), bbox_max, bbox_min, K, r)
        print(SortedPoints[:10])
        print(SortedGridCell[:10])
    '''

    def test_frnn_grid_gpu(self):
        K = 5
        r2 = 0.01
        r = 0.1
        max_num_points = 2000000
        num_pcs = 1
        pc = torch.rand((num_pcs, max_num_points, 3), dtype=torch.float)
        # pc[0, 0, 0] = 0
        # pc[0, 0, 1] = 0
        # pc[0, 0, 2] = 0
        # pc[0, 1, 0] = 1
        # pc[0, 1, 1] = 1
        # pc[0, 1, 2] = 1
        bbox_max = torch.max(pc[0], 0)[0] 
        bbox_min = torch.min(pc[0], 0)[0]
        # lengths = torch.randint(low=K, high=max_num_points, size=(num_pcs,), dtype=torch.long)
        lengths = torch.LongTensor([max_num_points])
        # frnn_idxs_cpu, frnn_dists_cpu = FRNN.cpu.frnn_bf_cpu(pc, pc, lengths, lengths, K, r2)
        start = time.time()
        # idxs_cpu, dists_cpu = FRNN.cpu.grid_test(pc[0], K, r)
        pc = pc.cuda()
        lengths = lengths.cuda()
        cpu_end = time.time()
        print("bf start")
        frnn_idxs_gpu, frnn_dists_gpu = FRNN.cuda.frnn_bf_gpu(pc, pc, lengths, lengths, K, r2)
        print("bf end")
        gpu_bf_end = time.time()
        print("grid start")
        idxs_gpu, dists_gpu = FRNN.cuda.grid_test_gpu(pc[0], bbox_max, bbox_min, K, r)
        print("grid end")
        gpu_end = time.time()
        # print("same idxs: ", float(torch.sum(frnn_idxs_cpu[0] == idxs_gpu.cpu())) / K / max_num_points)
        # print("same dists: ", float(torch.sum(torch.isclose(frnn_dists_cpu[0], dists_gpu.cpu()))) / K / max_num_points)
        # print("same idxs: ", float(torch.sum(idxs_cpu == idxs_gpu.cpu())) / K / max_num_points)
        # print("same dists: ", float(torch.sum(torch.isclose(dists_cpu, dists_gpu.cpu()))) / K / max_num_points)
        print("lose itself: ", torch.sum(idxs_gpu.cpu()[:, 0] != torch.arange(max_num_points)))
        print("lose itself: ", idxs_gpu.cpu()[:, 0])
        print(idxs_gpu[-5:])
        # print(idxs_cpu[-5:])
        # print("same gridcnt: ", float(torch.sum(idxs_gpu.cpu()==idxs_cpu.view(-1)))/idxs_gpu.shape[0])
        # print(frnn_idxs_cpu[0, :5])
        print(dists_gpu[-5:])
        # print(dists_cpu[-5:])
        print(cpu_end - start)
        print(gpu_bf_end - cpu_end)
        print(gpu_end - gpu_bf_end)
        # dists_cpu_linear = (dists_cpu[:, 0] * 11 + dists_cpu[:, 1])*11 + dists_cpu[:, 2]
        # print("same gridcell: ", float(torch.sum(dists_gpu.cpu()==dists_cpu_linear))/dists_gpu.shape[0])
        # convert gpu to cpu 
        # print(frnn_dists_cpu[0, :5])

        
        



if __name__ == "__main__":
    unittest.main()
