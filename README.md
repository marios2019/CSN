# Cross-Shape Attention for Part Segmentation of 3D Point Clouds

[Project Page](https://marios2019.github.io/CSN/) | Paper ([arxiv](https://arxiv.org/abs/2003.09053))

We present a deep learning method that propagates point-wise feature representations across shapes within a collection for 
the purpose of 3D shape segmentation. We propose a cross-shape attention mechanism to enable interactions between a 
shape’s point-wise features and those of other shapes. The mechanism assesses both the degree of interaction between 
points and also mediates feature propagation across shapes, improving the accuracy and consistency of the resulting 
point-wise feature representations for shape segmentation. Our method also proposes a shape retrieval measure to select 
suitable shapes for cross-shape attention operations for each test shape. Our experiments demonstrate that our approach 
yields state-of-the-art results in the popular PartNet dataset.

## CSN pipeline
<p align="center"> <img src="https://marios2019.github.io/CSN/assets/teaser.png" width="100%"> </p>
**Left:** Given an input shape collection, our method constructs a graph where each shape is represented as a node and 
edges indicate shape pairs that are deemed compatible for cross-shape feature propagation. Middle: Our network is 
designed to compute point-wise feature representations for a given shape (grey shape) by enabling interactions between 
its own point-wise features and those of other shapes using our cross-shape attention mechanism. *Right:* As a result, 
the point-wise features of the shape become more synchronized with ones of other relevant shapes leading to more 
accurate fine-grained segmentation.


## MinkowskiNet Experiments
Follow this [guide](MinkowskiNet/README.md), for the MinkowskiNet experiments on the PartNet dataset.

## MID-FC Experiments
Code coming soon!
