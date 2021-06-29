import frnn
import torch

if __name__ == '__main__':
    device = torch.device('cuda')
    points = torch.load('pts.pth')[None, :, :].to(device).float()
    n_points = torch.tensor([points.size(1)]).to(device).long()
    K = 10
    radius = 0.05
    print(points.size(), n_points)
    print(points.max(axis=1))
    print(points.min(axis=1))
    _, idxs, _, _ = frnn.frnn_grid_points(points,
                                          points,
                                          n_points,
                                          n_points,
                                          K,
                                          radius,
                                          grid=None,
                                          return_nn=False,
                                          return_sorted=False,
                                          radius_cell_ratio=2)
