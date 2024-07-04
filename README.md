# MvMM-RegNet

[[paper](https://arxiv.org/abs/2006.15573)]

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

### Requirements

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

Training the model is through `./src_3d/train_unified_seg.py` for 3D images and `./src_2d/train_unified_seg_2d.py` for 2D images. For example, the following code shows the usage for 3D pairwise registration:

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
--epochs 50
--learning_rate 1e-5
--dropout 0.2
--batch_size 1
```

For model testing, the following code shows how to perform inference for pairwise registration on 3D images as an example:

```
python save_prediction_pairwise.py
--cuda_device 0
--atlas_search_path #YOUR OWN TEST DATASET#
--model_path #YOUR OWN CHECKPOINT FILE#
```

*Note* that some of the parameters are not included in the above illustration. However, one can delve into the running scripts for all the configurations and also for their clear explanation.

## Reproducibility

In the paper, we tested four variants of the MvMM-RegNet using difference appearance models. The average Dice and Hausdorff distance (HD) statistics for pairwise MR-to-MR registration on [MM-WHS](https://zmiclab.github.io/projects/mmwhs/) dataset are:

| Method        | Dice          | HD (mm)       |
| ------------- | ------------- | ------------- |
| Baseline-MoG  | 0.832 ± 0.027 | 19.65 ± 2.792 |
| Baseline-Mask | 0.840 ± 0.028 | 16.91 ± 2.374 |
| Baseline-ECC  | 0.844 ± 0.026 | 16.69 ± 2.355 |
| Baseline-NCC  | 0.847 ± 0.028 | 16.83 ± 2.422 |

The trained TensorFlow model checkpoint files are included in `./src_3d/baselines/`, which can be restored for reproducible experiments.

## Groupwise registration

In our paper, we explored a special case of groupwise registration by setting the common space onto a target image as the reference space. This specialization can be viewed as a natural framework for multi-atlas segmentation, during which multiple expert-annotated images with segmented labels, called atlases, are registered to the target space. These warped atlas labels are further combined by label fusion to produce the final segmentation on target image. 

We implemented a 2D version of MvMM-RegNet on [MS-CMRSeg](https://zmiclab.github.io/projects/mscmrseg19/) dataset to investigate deep-learning based groupwise registration. To train the model, one can run  `./src_2d/train_unified_seg_2d.py` using the following code:

```
python train_unified_seg_2d.py
--cuda_device 0                                                     # specify GPU ID
--train_target_search_path #YOUR OWN TRAINING DATASET#              # training target dataset
--train_atlas_search_path #YOUR OWN TRAINING DATASET#               # training atlas dataset
--test_target_search_path #YOUR OWN VALIDATION/TEST DATASET#        # validation/test target dataset
--test_atlas_search_path #YOUR OWN VALIDATION/TEST DATASET#         # validation/test atlas dataset
--num_atlases 3                                                     # number of atlases in the model
--cost_function mvmm_net_ncc                                        # specify loss function
--regularization_coefficient 0 0.1                                  # specify regularization coefficient
--optimizer_name adam-clr                                           # specify optimizer
--pretrain_epochs 1                                                 # pretrain the network 
--epochs 50
--learning_rate 1e-5
--dropout 0.2
--batch_size 4
```

*Note* that some of the parameters are not included in the above illustration. However, one can delve into the running scripts for all the configurations and also for their clear explanation.

## Acknowledgement

Some parts of our code were adapted from [VoxelMorph](https://github.com/voxelmorph/voxelmorph/tree/master) and [label-reg](https://github.com/YipengHu/label-reg), which are both excellent repositories for medical image registration.

## Citation

If you found the repository useful, please cite our work as below:

```bibtex
@inproceedings{Luo2020MvMM-RegNet,
  title={MvMM-RegNet: A new image registration framework based on multivariate mixture model and neural network estimation},
  author={Luo, Xinzhe and Zhuang, Xiahai},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={149--159},
  year={2020},
  organization={Springer}
}
```

## Contact

For any questions or problems please [open an issue](https://github.com/xzluo97/MvMM-RegNet/issues/new) on GitHub.
