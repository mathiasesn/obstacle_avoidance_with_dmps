# DenseFusion: 6D Object Pose Estimation by Iterative Dense Fusion

The following work is based upon: [DenseFusion](https://github.com/j96w/DenseFusion).

For more information about DenseFusion [ARCHITECTURE.md](ARCHITECTURE.md).

## Requirements

- python 3.6.10
- pytorch 1.0.0
- PIL
- scipy
- numpy
- pyyaml
- logging
- matplotlib
- CUDA 10
- opencv 4.2

For further information see [requirements.txt](requirements.txt).

## Code Structure

- lib
    - lib/knn: CUDA K-nearest neighbours library adapted from [pytorch_knn_cuda](https://github.com/chrischoy/pytorch_knn_cuda)
    - lib/extractors.py: Encoder network architecture adapted from [pspnet-pytorch](https://github.com/Lextal/pspnet-pytorch)
    - lib/loss_refiner.py: Loss calculation for iterative refinement model
    - lib/loss.py: Loss calculation for DenseFusion model
    - lib/network.py: Network architecture
    - lib/pspnet.py: Decoder network architecture
    - lib/transformation.py: [Transformation Function Library](https://pypi.org/project/transformations/)
    - lib/utils.py: Logger code and K-nearest neighbours class
- logs: Training log files
- trained_models
    - trained_models/linemod: Checkpoints of linemod dataset