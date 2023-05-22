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

#### MinkHRNetCSN
This network implements our Cross-Shape Attention layer, and it utilizes the MinkHRNet as a backbone. Use
the following command to train our Cross-ShapeNet:
```bash
./scripts/training_csn.sh <PartNet_category> <num_neighbors>

# <PartNet_category>: specify the PartNet category, e.g., Bed, Bottle, etc. You can view all the available categories
# by providing the '--show_categories' argument.
# <num_neighbors>:  number of shapes to retrieve from the shape collection, e.g., 0 (only Self-Shape Attention - SSA), 
# 1, 2, 3 (CSA)
```
Adjust the batch size, according to your computational resources. You can also use the `ITER_SIZE` argument for gradient 
accumulation. If `ITER_SIZE>1` you will need to lower the batch size of the sub-iteration, so that the total batch size is 8
(e.g. `ITER_SIZE=2`, `BATCH_SIZE=4`, total `batch_size=8`). This is especially useful for experiments with `num_neighbors=2 or 3`.
For our experiments we used the following settings (see `scripts/training_csn.sh` lines 36-37 and `lib/trainer_csn` lines 194-210):
* `num_neighbors=0 (only SSA)`: `BATCH_SIZE=8, ITER_SIZE=1`
* `num_neighbors=1`: `BATCH_SIZE=8, ITER_SIZE=1`
* `num_neighbors=2`: `BATCH_SIZE=4, ITER_SIZE=2` (equivalent to `batch_size=8`)
* `num_neighbors=3`: `BATCH_SIZE=2, ITER_SIZE=4` (equivalent to `batch_size=8`)

The NVIDIA V100 GPU (```VRAM=32GB```) was used in all of our experiments.

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

#### MinkHRNetCSN
To evaluate a trained model on the PartNet test splits you can use the following:
```bash
./scripts/testing_csn.sh <PartNet_category> <num_neighbors> 

# <PartNet_category>: specify the PartNet category, e.g., Bed, Bottle, etc. You can view all the available categories
# by providing the '--show_categories' argument.  You call also use the 'all' option to evaluate all categories.
# <num_neighbors>:  number of shapes to retrieve from the shape collection, e.g., 0 (only Self-Shape Attention - SSA), 
# 1, 2, 3 (CSA)
```

**Optional:** After you evaluate all PartNet categories you can collect the results using:
```bash
python lib/collect_partnet_results.py <experiments_directory> <num_neighbors>

# <experiments_directory>: specify the experiments base directory, e.g.,  outputs/PartnetVoxelization0_05Dataset/HRNetSimCSN3S/
# <num_neighbors>: number of shape that were retrieved from the shape colleciton, e.g., 0, 1, 2 or 3  
```

### Pretrained models

|       Model       | Voxel Size |  Input Features   | #Neighbors | avg. part IoU | avg. shape IoU |                                              Link                                               |
|:-----------------:|:----------:|:-----------------:|:----------:|:-------------:|:--------------:|:-----------------------------------------------------------------------------------------------:|
|    HRNetSeg3S     |    0.05    | Point coordinates |     -      |     48.0      |      54.4      |[download](https://drive.google.com/file/d/1WIOii5OzrzYfyg2mX40cQZjYOvaOdnWE/view?usp=share_link)|
| HRNetSimCSN3S_SSA |    0.05    | Point coordinates |     0      |     48.7      |      56.0      |[download](https://drive.google.com/file/d/1MxD-7Gra09CCcGo59b6ogmjEy3ML4Kt9/view?usp=share_link)|
| HRNetSimCSN3S_K1  |    0.05    | Point coordinates |     1      |     49.9      |      56.2      |[download](https://drive.google.com/file/d/1TrlFsdUfqWcw-135hgLJMLbsoS1DULBQ/view?usp=share_link)|
| HRNetSimCSN3S_K2  |    0.05    | Point coordinates |     2      |     49.7      |      55.9      |[download](https://drive.google.com/file/d/1sTSGVlStY5Zx5iEyK8_NDA1hyzWxsFjW/view?usp=share_link)|
| HRNetSimCSN3S_K3  |    0.05    | Point coordinates |     3      |     47.2      |      53.6      |[download](https://drive.google.com/file/d/1YHh_qFSFJCWZliLbcGoEwlPGzSwIPmqW/view?usp=share_link)|

You can download the pretrained models using the following script:
```bash
./scripts/download_pretrained_models.sh
```
