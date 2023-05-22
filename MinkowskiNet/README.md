# MinkowskiNet implementation
This folder contains code for the MinkowskiNet experiments based on the [Minkowski Engine](https://github.com/NVIDIA/MinkowskiEngine).

## Installation

### Requirements
The present code was tested on the following environment:

- Ubuntu 20.04
- CUDA 10.2.89
- cuDNN 7.6.5
- PyTorch 1.8.1
- Python 3.8
- Minkowski Engine 0.5.4

### Conda environment
You can use the ```environment.yml``` in order to initialize a working environment via ```conda```:
```bash
conda env create -f environment.yml
```
The Minkowski Engine **v0.5.4** needs to also be installed, either [manually](https://github.com/NVIDIA/MinkowskiEngine)
or through [pip](https://pypi.org/project/MinkowskiEngine/).

## Part Segmentation

### Dataset
For the 3D part segmentation task we used the [PartNet](https://partnet.cs.stanford.edu/) dataset. First, download the 
PartNet v0 from [here](https://www.shapenet.org/download/parts). You will need to download the HDF5 files for the 
**semantic segmentation task** (Sec 5.1 of PartNet paper - 8GB). Create the following directory ```Dataset/PartNet```
and extract the contents of the downloaded file  ```sem_seg_h5.zip``` there. The data should be organized as follows:
```shell
CSN
└── MinkowskiNet
    └── Dataset
        └── PartNet
            └── sem_seg_h5
                ├── Bag-1
                ├── Bed-1
               ...
                └── Vase-3
```

### Training

#### MinkNetHRNet
This network is inspired by the [HRNet](https://github.com/HRNet/HRNet-Semantic-Segmentation) architecture that 
processes 2D images in a multi-resolution manner. You can train our sparse tensor HRNet variant by using the following
bash script:
```bash
./scripts/training_hrnet.sh <PartNet_category>

# <PartNet_category>: specify the PartNet category, e.g., Bed, Bottle, etc. You can view all the available categories
# by providing the '--show_categories' argument.
```
Adjust the batch size, according to your computational resources. For our experiments we set `batch_size=8`, and
the NVIDIA V100 GPU was used (```VRAM=32GB```).

### Evaluation

#### MinkNetHRNet
To evaluate a trained model on the PartNet test splits you can use the following:
```bash
./scripts/testing_hrnet.sh <PartNet_category>

# <PartNet_category>: specify the PartNet category, e.g., Bed, Bottle, etc. You can view all the available categories
# by providing the '--show_categories' argument.  You call also use the 'all' option to evaluate all categories.
```

**Optional:** After you evaluate all PartNet categories you can collect the results using:
```bash
python lib/collect_partnet_results.py <experiments_directory>

# <experiments_directory>: specify the experiments base directory, e.g.,  outputs/PartnetVoxelization0_05Dataset/HRNetSeg3S/ 
```

### Pretrained models

|    Model     | Voxel Size |  Input Features   | Batch Size | avg. part IoU | avg. shape IoU |                                              Link                                               |
|:------------:|:----------:|:-----------------:|:----------:|:-------------:|:--------------:|:-----------------------------------------------------------------------------------------------:|
| HRNetSeg3S   |    0.05    | Point coordinates |     8      |     48.0      |      54.4      |[download](https://drive.google.com/file/d/1WIOii5OzrzYfyg2mX40cQZjYOvaOdnWE/view?usp=share_link)|
You can download the pretrained models using the following script:
```bash
./scripts/download_pretrained_models.sh
```

