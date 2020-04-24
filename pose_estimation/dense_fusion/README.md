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
- trained_checkpoints
    - trained_checkpoints/linemod: Checkpoints of linemod from author found [here](https://drive.google.com/drive/folders/19ivHpaKm9dOrr12fzC8IDFczWRPFxho7)
- eval.py: Evaluation script
- train.py: Training script

## Training

To the train the network run the python file train.py. An example is given below:

```bash
python pose_estimation/dense_fusion/train.py --resume_posenet pose_model_current.pth --resume_refinenet pose_refine_model_current.pth --start_epoch 11
```

The default dataset root is set to 'pose_estimation/dataset/linemod/Linemod_preprocessed', thus if a new dataset is implemented this will have to be changed. This can be set by adding the parameter '--dataset_root'.

## Evaluation

```bash
python pose_estimation/dense_fusion/eval.py --model pose_estimation/dense_fusion/trained_models/linemod/pose_model_4_0.012838995820563148.pth --refine_model pose_estimation/dense_fusion/trained_models/linemod/pose_refine_model_9_0.012751821958854472.pth
```

The default dataset root is also set to 'pose_estimation/dataset/linemod/Linemod_preprocessed' and can be changed by setting the parameter '--dataset_root'.
