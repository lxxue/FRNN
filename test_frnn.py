import torch
from frnn import frnn_grid_points
from pytorch3d.ops import knn_points

pts_world = torch.load("pts_world.pt")[:2]
num_points_per_cloud = torch.load("num_points_per_cloud.pt")[:2]

dists_knn, idxs_knn, _ = knn_points(pts_world, pts_world, num_points_per_cloud, num_points_per_cloud, K=3)
dists_frnn, idxs_frnn, _, _ = frnn_grid_points(pts_world, pts_world, num_points_per_cloud, num_points_per_cloud, K=3, r=0.5)

# mask = dists_knn > 0.5*0.5
# dists_knn[mask] = -1
# idxs_knn[mask] = -1
N = dists_knn.shape[0]
# for i in range(N):
print(num_points_per_cloud[0])
print(num_points_per_cloud[1])
# print(num_points_per_cloud[2])
print(pts_world.shape[1])
# 
print(dists_knn[1, :num_points_per_cloud[1], 0])
print(torch.sum(dists_frnn[0, :3999, 0]))
print(torch.sum(dists_frnn[1, :num_points_per_cloud[1], 0]))
# print(torch.sum(dists_frnn[2, :num_points_per_cloud[2], 0]))
print(torch.allclose(dists_frnn[1, :num_points_per_cloud[1]], dists_knn[1, :num_points_per_cloud[1]]))
