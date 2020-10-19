# FRNN
A Fixed Nearest Neighbors Search implemented on CUDA with similar interface as [pytorch3d.ops.knn_points](https://pytorch3d.readthedocs.io/en/latest/modules/ops.html#pytorch3d.ops.knn_points).

## Performance

![Performance](./images/teaser.png)

## Algorithm Walkthrough & Experiment Results
[FRNN Presentation](./frnn_pre.pdf)

## Depenency
Tested with cuda 10.2, python 3.8 and pytorch 1.6.0 on ubuntu 18.04.

Should be also fine other versions of cuda/python/pytorch.

## Install
```
git clone --recursive https://github.com/lxxue/FRNN.git
# install a prefix_sum routine first
cd FRNN/external/prefix_sum
python setup.py install

# install FRNN
cd ../../ # back to the {FRNN} directory
python setup.py install
```

## Usage
For fixed nearest neighbors search:
[doc](https://github.com/lxxue/FRNN/blob/59a4c8fdc786c64afd991919af39f1a65d4ec2ff/frnn/frnn.py#L154-L224)
```
  # first time there is no cached grid
  dists, idxs, nn, grid = frnn.frnn_grid_points(
        points1, points2, lengths1, lengths2, K, r, grid=None, return_nn=False, return_sorted=True
  )
  # if points2 and r don't change, we can reuse the grid
  dists, idxs, nn, grid = frnn.frnn_grid_points(
        points1, points2, lengths1, lengths2, K, r, grid=grid, return_nn=False, return_sorted=True
  )
```
For manually gather nearest neighbors from idxs generated via frnn_grid_points:
[doc](https://github.com/lxxue/FRNN/blob/59a4c8fdc786c64afd991919af39f1a65d4ec2ff/frnn/frnn.py#L268-L292)
```
  nn = frnn.frnn_gather(points2, idxs, lengths2)
```

## Note
For now this function only supports D=3 (point clouds) and K <= 32. Would add more supports for arbitrary D & K in the near future. For point clouds with fewer than 10,000 points, the speedup might not be that much.


## Acknowledgement

The code is build on the [algorithm](https://on-demand.gputechconf.com/gtc/2014/presentations/S4117-fast-fixed-radius-nearest-neighbor-gpu.pdf) introduced by Rama C. Hoetzlein. I use the [parallel prefix_sum](https://github.com/lxxue/prefix_sum) routines implemented by [mattdean1](https://github.com/mattdean1/cuda). I also learn (copy & paste) a lot from [Pytorch3D's KNN](https://github.com/facebookresearch/pytorch3d/blob/master/pytorch3d/csrc/knn/knn.cu) implementations.

<!--
## TODO

1. Fix the problem of error for long thin objects
2. Support dimensions for arbitrary D
3. Support K > 32
4. KNN grid implementations
-->
