# MvMM-RegNet



>  Implementation of the paper: MvMM-RegNet: A new image registration framework based on multivariate mixture model and neural network estimation, accepted to MICCAI 2020.

## Getting Started

### Project Structure

The project contains implementation of MvMM-RegNet on both 3D and 2D medical images. The project structure is:

```
MvMM-RegNet
|-- src_2d								# 2D version of MvMM-RegNet
|	|-- core							# dataset, network, trainer and utilities
|	|-- help							# data augmentation and evaluation
|	|-- preprocessing					# data preprocessing 	
|	|-- __init__.py
|	|-- save_test_prediction_2d.py		# model testing
|	|-- train_unified_seg_2d.py			# model training
|-- src_3d								# 3D version of MvMM-RegNet
|	|-- core							# dataset, network, trainer and utilities
|	|-- help							# data augmentation and preprocessing
|	|-- __init__.py
|	|-- save_label_fusion.py			# label fusion
|	|-- save_prediction_pairwise.py		# model testing
|	|-- train_unified_seg.py			# model training
|-- LICENSE
|-- README.md
|-- camera-ready.pdf					# camera-ready paper
```

### Requirement

Here are the essential pa