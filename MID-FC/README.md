# CrossShapeNet

This repository contains the implementation of Cross-Shape Attention on top of the point-cloud features extracted from MID-FC architecture that led to state-of-the-art results on the PartNet Semantic Shape Segmentation benchmark. Please checkout the following links for details on our work. 

**Cross-Shape Attention for Part Segmentation of 3D Point Clouds | [Project Page](https://marios2019.github.io/CSN/) | [Paper](https://arxiv.org/abs/2003.09053)**

## Extracting PartNet features from MID-FC

To extract the features from the MID-Net (Octree based HRNet) architecture, first setup the Tensorflow implementation of the original O-CNN repo from [here](https://github.com/Microsoft/O-CNN). Specifically see [Installation](https://github.com/microsoft/O-CNN/blob/master/docs/installation.md) and set up [Octree](https://github.com/microsoft/O-CNN/blob/master/docs/installation.md) and [Tensorflow code](https://github.com/microsoft/O-CNN/blob/master/docs/installation.md). Please also set up the PartNet dataset following [this](https://github.com/microsoft/O-CNN/blob/master/docs/segmentation.md#shape-segmentation-on-partnet-with-tensorflow) and follow the steps 1 and 2 for setting up.

First copy the files from `ocnn_extraction` and paste (replace) them in the `O-CNN/tensorflow/script/` folder. 

We extracted the dense numpy features from the first fully-connected layer of the MID-FC HRNet network and save them on the disk. We also saved the corresponding point-clouds, point labels as well as logits from the final MID-FC layer. 

To extract the features from the training data, use

```
cd O-CNN/tensorflow/script/
python run_seg_partnet_test_cmd.py --phase=train --input_path=PATH_TO_TRAIN_TFRECORDS --logs_dir=PATH_TO_TRAIN_FEATURES_DIR 

# For testing
python run_seg_partnet_test_cmd.py --phase=test --input_path=PATH_TO_TEST_TFRECORDS --logs_dir=PATH_TO_TEST_FEATURES_DIR 
```

- `PATH_TO_TRAIN_TFRECORDS/PATH_TO_TEST_TFRECORDS`: Path to the level-3 tfrecords generated using [this](https://github.com/microsoft/O-CNN/blob/master/docs/segmentation.md#shape-segmentation-on-partnet-with-tensorflow).
- `logs_dir`: Directory path where all the features will be stored in subdirs fc_1, labels, preds, pts

## Training CrossShapeNet

### Self-Shape Attention 

First, the architecture is trained with conventional self-attention to generate meaningfull dense point-wise representation to help sample the most relevant shapes for Cross-Attention. 

To run the training for single-shape (eg Bed, Bottle, Chair, ....), use the following command

```
python run_training.py --logs_dir='PATH_TO_LOGS' --attention_type='ssa' --start=PART_INDEX --end=PART_INDEX --n_heads=ATTENTION_HEADS --batch_size=4 --lr=0.001 --cmd --num_workers=3
```

For more arguments please check the [run_training.py](https://github.com/gargsid/CrossShapeNet/blob/main/run_training.py).

- `logs_dir`: Path to the folder that will store the trained model and logs. Folder will be created if not already present
- `attention_type`: 'ssa' for self-attention and 'csa' for cross-attention
- `n_heads`: number of self-attention heads used in the training
- `start`: starting index in the range of categories for which we want to run the training. Please check [this](https://github.com/gargsid/CrossShapeNet/blob/988c1c480e1b5fb221b0521757fa00244dde3731/run_training.py#L7). 
- `end`: ending inde in the range of categories for which we want to run the training. So start=0 and end=0 will the run the training for Bed 
- `cmd`: initiating it will run the training on console
To submit the training to a GPU, please modify [these](https://github.com/gargsid/CrossShapeNet/blob/988c1c480e1b5fb221b0521757fa00244dde3731/run_training.py#L112C14-L125) lines according to your system configurations and replace `--cmd` with `--job` argument. 

**To submit jobs for all the shapes simultaneously use** `start=0` and `end=16` with `--job` flag. 

### Generating KNN graph 

Once the models are trained using self-attention, we might want to precompute the KNN graphs so that they are not computed everytime we initiate cross-shape attention training because some categories like Lamp, Chair takes a large amount of time for graph construction because of large number of point-cloud shapes in each categories. 

To generate the precomputed graphs use

`python run_save_knn.py --ssa_logs_dir=PATH_TO_LOGS_DIR --n_heads=ATTENTION_HEADS --batch_size=4 --num_workers=3 --start=0 --end=16 --job`

- `ssa_logs_dir`: Name of the directory (excluding the part-name) where self-attention trained models are saved
- `n_heads`: number of self-attention heads used in the training

To submit jobs for particular shapes modify the `start` and `end` flags according to [this](https://github.com/gargsid/CrossShapeNet/blob/988c1c480e1b5fb221b0521757fa00244dde3731/run_training.py#L7). 

To run the construction on console replace `--job` with `--cmd`. Note that this will run the construction for the shapes in range between start to end shapes one by one. 

The final graphs will be saved in the folder: `ssa_logs_dir/knn_graphs/`

### Cross-Shape Attention 

After running the self-attention training and generating the KNN graphs, we will train the network with cross-shape attention. We will also inialize our weights with the pretrained self-attention layers because that led to more stable training. 

```
python run_training.py --logs_dir='PATH_TO_LOGS'  --ssa_logs_dir='PATH_TO_SSA_LOGS' --attention_type='csa' --start=PART_INDEX --end=PART_INDEX --n_heads=ATTENTION_HEADS --batch_size=4 --num_workers=3 --lr=0.001 --start=0 --end=16 --job
```
- `logs_dir`: Path to the folder that will store the trained model and logs. Folder will be created if not already present
- `n_heads`: number of self-attention heads used in the training
- `ssa_logs_dir`: Path to the folder that saved the self-attention models with same number of attention heads. 
- `attention_type`: 'csa' for cross-attention 

Similar to graph construction, this will run the training for all shapes simultaneously. To run the construction on console replace `--job` with `--cmd` and change `--start` and `--end` flags according to your requirements. 


## Running Pretrained Model

Please download the pretrained models from [here](https://drive.google.com/file/d/1I71Yv3zS0DP75FwaM8l3ahNUia98f4mk/view?usp=sharing). Here N_HEADS=8 and K=4. Store the directory called `pretrained_models` in some folder (let's say LOGS). `pretrained_models` folder containes the CSA trained models as well as precomputed graphs. (But it will be better to generate the graphs again for your dataset.)

Now to run the predictions, use

```
python run_csa_pred.py --logs_dir=LOGS --start=0 --end=16
```

- `--logs_dir`: Path to the folder that contains `pretrained_models` directory

The results will be stored in the `logs_dir/pretrained_models/run_1/PARTNAME/part_IoU_summaries.csv`, where `PARTNAME` is individual shape category. 

## Results

<p align="center"><img src="https://github.com/gargsid/CrossShapeNet/blob/main/assets/midfc_results.png" width="100%"></p>

## Citation
Please also consider citing the corresponding papers.
```
@article{CSN:2023,
  author  = {Marios Loizou and Siddhant Garg and Dmitry Petrov and Melinos Averkiou and Evangelos Kalogerakis},
  title   = {{Cross-Shape Attention for Part Segmentation of 3D Point Clouds}},
  journal = {Computer Graphics Forum (Proc. SGP)},
  year    = {2023},
  volume  = {42},
  issue   = {5}
}
```