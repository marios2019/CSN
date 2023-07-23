# Cross-Shape Attention for Part Segmentation of 3D Point Clouds
**Computer Graphics Forum (Proc. SGP), 2023**

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/cross-shape-graph-convolutional-networks/3d-semantic-segmentation-on-partnet)](https://paperswithcode.com/sota/3d-semantic-segmentation-on-partnet?p=cross-shape-graph-convolutional-networks)

This is the official implementation of **Cross-ShapeNet**, a deep learning method that propagates point-wise feature 
representations across shapes within a collection for the purpose of 3D part segmentation. For more technical details,
please refer to:

**Cross-Shape Attention for Part Segmentation of 3D Point Clouds**

[Marios Loizou](https://marios2019.github.io/), [Siddhant Garg](https://gargsid.github.io/), [Dmitry Petrov](https://lodurality.github.io/),
[Melinos Averkiou](https://melinos.github.io/), [Evangelos Kalogerakis](https://people.cs.umass.edu/~kalo/)

[**Project Page**](https://marios2019.github.io/CSN/) | **Paper** ([arxiv](https://arxiv.org/abs/2003.09053), 
[CGF](https://diglib.eg.org/handle/10.1111/cgf14909)) | [**SGP 2023 presentation**](https://marios2019.github.io/CSN/assets/CSN_presentation.pdf)

## CSN pipeline
<p align="center"><img src="https://marios2019.github.io/CSN/assets/teaser.png" width="100%"></p>

_Left:_ Given an input shape collection, our method constructs a graph where each shape is represented as a node and 
edges indicate shape pairs that are deemed compatible for cross-shape feature propagation. _Middle:_ Our network is 
designed to compute point-wise feature representations for a given shape (grey shape) by enabling interactions between 
its own point-wise features and those of other shapes using our cross-shape attention mechanism. _Right:_ As a result, 
the point-wise features of the shape become more synchronized with ones of other relevant shapes leading to more 
accurate fine-grained segmentation.

## MinkowskiNet Experiments
Follow this [guide](MinkowskiNet/README.md) for the MinkowskiNet experiments on the PartNet dataset.

## MID-FC Experiments
To conduct the MID-FC experiments on the PartNet dataset, please follow the instructions in the following [guide](MID-FC/README.md).

## Acknowledgement

This repo is developed based on [Spatio-Temporal Segmentation](https://github.com/chrischoy/SpatioTemporalSegmentation)
and [O-CNN](https://github.com/microsoft/O-CNN).

Please also consider citing the corresponding papers.

## Citation

```latex
@article{CSN:2023,
  author  = {Marios Loizou and Siddhant Garg and Dmitry Petrov and Melinos Averkiou and Evangelos Kalogerakis},
  title   = {{Cross-Shape Attention for Part Segmentation of 3D Point Clouds}},
  journal = {Computer Graphics Forum (Proc. SGP)},
  year    = {2023},
  volume  = {42},
  issue   = {5}
}
```
