# Cross-Shape Attention for Part Segmentation of 3D Point Clouds

This is the official implementation of **Cross-ShapeNet**, deep learning method that propagates point-wise feature 
representations across shapes within a collection for the purpose of 3D part segmentation. For more technical details,
please refer to:

**Cross-Shape Attention for Part Segmentation of 3D Point Clouds**

[Marios Loizou](https://marios2019.github.io/), [Siddhant Garg](https://gargsid.github.io/), [Dmitry Petrov](https://lodurality.github.io/),
[Melinos Averkiou](https://melinos.github.io/), [Evangelos Kalogerakis](https://people.cs.umass.edu/~kalo/)

[**Project Page**](https://marios2019.github.io/CSN/) | **Paper** ([arxiv](https://arxiv.org/abs/2003.09053))

## CSN pipeline
<p align="center"><img src="https://marios2019.github.io/CSN/assets/teaser.png" width="100%"></p>

_Left:_ Given an input shape collection, our method constructs a graph where each shape is represented as a node and 
edges indicate shape pairs that are deemed compatible for cross-shape feature propagation. _Middle:_ Our network is 
designed to compute point-wise feature representations for a given shape (grey shape) by enabling interactions between 
its own point-wise features and those of other shapes using our cross-shape attention mechanism. _Right:_ As a result, 
the point-wise features of the shape become more synchronized with ones of other relevant shapes leading to more 
accurate fine-grained segmentation.


## MinkowskiNet Experiments
Follow this [guide](MinkowskiNet/README.md), for the MinkowskiNet experiments on the PartNet dataset.

## MID-FC Experiments
Code coming soon!
