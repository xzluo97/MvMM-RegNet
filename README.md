# MvMM-RegNet



>  Implementation of the paper: MvMM-RegNet: A new image registration framework based on multivariate mixture model and neural network estimation, accepted to MICCAI 2020.

## Getting Started

### Project Structure

The project contains implementation of MvMM-RegNet on both 3D and 2D medical images. The project structure is:

```
MvMM-RegNet
|-- src_2d                              # 2D version of MvMM-RegNet
|	|-- core                            # dataset, network, trainer and utilities
|	|-- help                            # data augmentation and evaluation
|	|-- preprocessing                   # data preprocessing 	
|	|-- __init__.py
|	|-- save_test_prediction_2d.py      # model testing
|	|-- train_unified_seg_2d.py         # model training
|-- src_3d                              # 3D version of MvMM-RegNet
|	|-- core                            # dataset, network, trainer and utilities
|	|-- help                            # data augmentation and preprocessing
|	|-- __init__.py
|	|-- save_label_fusion.py            # label fusion
|	|-- save_prediction_pairwise.py     # model testing
|	|-- train_unified_seg.py            # model training
|-- LICENSE
|-- README.md
|-- camera-ready.pdf                    # camera-ready paper
```

### Requirement

Here are the essential modules that you need to install before getting started. Versions are also given for reproducibility.

```
matplotlib==3.1.1
nibabel==2.5.1
numba==0.46.0
opencv-python==4.1.1.26
pandas==0.25.3
pytorch==1.3.1
scikit-image==0.15.0
scikit-learn==0.21.3
scipy==1.3.1
tensorflow-gpu==1.14.0
tqdm==4.42.1
```

### Usage

Training the model is through `./src_3d/train_unified_seg.py` for 3D images and `./src_2d/train_unified_seg_2d.py` for 2D images. For example, the following code shows the usage for 3D-training:

```
python train_unified_seg.py
--cuda_device 0                                                     # specify GPU ID
--train_target_search_path #YOUR OWN TRAINING DATASET#              # training target dataset
--train_atlas_search_path #YOUR OWN TRAINING DATASET#               # training atlas dataset
--test_target_search_path #YOUR OWN VALIDATION/TEST DATASET#        # validation/test target dataset
--test_atlas_search_path #YOUR OWN VALIDATION/TEST DATASET#         # validation/test atlas dataset
--cost_function mvmm_net_mask                                       # specify loss function
--regularization_coefficient 0 0.1 0.1 0.001                        # specify regularization coefficient
--optimizer_name adam-clr                                           # specify optimizer    
--learning_rate 1e-5
--dropout 0.2
--batch_size 1
--epochs 50
```

For model testing, the following code shows how to perform inference for pairwise registration on 3D images:

```
python save_prediction_pairwise.py
--cuda_device 0
--atlas_search_path #YOUR OWN TEST DATASET#
--model_path #YOUR OWN CHECKPOINT FILE#
```

Some of the parameters are not included in the above illustration. However, one can delve into the running scripts for all the configurations and also for their clear explanation.

## Reproducibility