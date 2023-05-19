# MinkowskiNet implementation

This folder contains code for the MinkowskiNet experiments, based on the [Minkowski Engine](https://github.com/NVIDIA/MinkowskiEngine).
This implementation is an adaptation of the  [Spatio-Temporal Segmentation](https://github.com/chrischoy/SpatioTemporalSegmentation)
repository.

## Installation

### Requirements
The present code was tested on the following environment:

- Ubuntu 20.04
- CUDA 10.2.89
- cuDNN 7.6.5
- PyTorch 1.8.1
- Python 3.8

### Conda environment
You can use the ```environment.yml``` in order to initialize a working environment via ```conda```:
```bash
conda env create -f environment.yml
```
The Minkowski Engine **v0.5.4** needs to also be installed, either [manually](https://github.com/NVIDIA/MinkowskiEngine)
or through [pip](https://pypi.org/project/MinkowskiEngine/).
