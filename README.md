# FDConv

**[CVPR 2025]** Official implementation of *Frequency Dynamic Convolution for Dense Image Prediction*.  
FDConv enhances dynamic convolution by learning frequency-diverse weights in the Fourier domain, achieving state-of-the-art performance with minimal parameter overhead.

[![Paper](https://img.shields.io/badge/Paper-CVPR%202025-blue)](https://arxiv.org/abs/2503.18783) ←*click here to read the paper~*

![FDConv Overview](./assets/method.png)

## 📰News

- 2025.7.14 Code for training with mmdet using FDConv ([here](./FDConv_detection)).
- 2025.7.5 Code for converting ImageNet pretrained weight to FDConv weight ([here](./tools)).

## 🚀 Key Features

- **Fourier Disjoint Weight (FDW):** Constructs frequency-diverse kernels by learning disjoint spectral coefficients, eliminating parameter redundancy.
- **Kernel Spatial Modulation (KSM):** Dynamically adjusts filter responses at the element-wise level using local-global feature fusion.
- **Frequency Band Modulation (FBM):** Modulates spatial-frequency bands adaptively for context-aware feature extraction.
- **Plug-and-Play:** Seamlessly integrates into ConvNets and Transformers.

## 📈 Performance Highlights

| Task                  | Method                                                       | Metrics (Improvement)  | Params Cost |
| --------------------- | ------------------------------------------------------------ | ---------------------- | ----------- |
| Object Detection      | Faster R-CNN ([config](./FDConv_detection/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco_FDConv.py)) | AP↑2.2%                | **+3.6M**   |
| Instance Segmentation | Mask R-CNN ([config](./FDConv_detection/configs/mask_rcnn/mask_rcnn_r50_fpn_1x_coco_adamw_FDConv.py)) | AP<sup>mask</sup>↑2.2% | +3.6M       |
| Semantic Segmentation | UPerNet                                                      | mIoU↑3.1%              | +3.6M       |

**Outperforms CondConv (+90M), DY-Conv (+75M), and ODConv (+65M) with 1/20 parameters!**

## 🛠 Installation

You can install mmdet following the guidence of [mmdetection](https://github.com/open-mmlab/mmdetection/tree/dev-2.x) [Installation](https://mmdetection.readthedocs.io/en/v2.8.0/get_started.html#installation).

## 🏎️ Quick Start

```python
from FDConv import FDConv

# Replace standard convolution in your model
model.conv = FDConv(in_channels=64, out_channels=64, kernel_size=3, kernel_num=64)
```

## 🔄 Using Pre-trained Models with FDConv

The FDConv layer replaces standard nn.Conv2d or nn.Linear layers. To leverage official pre-trained weights (e.g., from ImageNet), you must first convert their standard spatial-domain weights into our Fourier-domain format (.dft_weight). We provide a versatile script to automate this process.

### The Conversion Script

The script tools/convert_to_fdconv.py handles the conversion. It loads a standard checkpoint, identifies target layers based on the specified model architecture, transforms their weights using 2D Fast Fourier Transform (FFT), and saves a new checkpoint compatible with models using FDConv.

### Usage

The general command is:

```
python tools/convert_to_fdconv.py \
    --model_type <MODEL_TYPE> \
    --weight_path <PATH_TO_ORIGINAL_WEIGHTS> \
    --save_path <PATH_TO_SAVE_CONVERTED_WEIGHTS>
```

- --model_type: Specify the architecture. Currently supported: resnet for ResNet-like models, and mit for Mix Transformer models (like SegFormer).
- --weight_path: Path to the downloaded official pre-trained weights.
- --save_path: Path where the new, converted weights will be saved.

### Examples

#### Example 1: Converting a ResNet-18 Model

To convert an official ImageNet pre-trained ResNet-18 model for use with FDConv:

1. Download the official ResNet-18 weights (e.g., resnet18-fbbb1da6.pth).
2. Run the conversion script:

Generated bash

```
python tools/convert_to_fdconv.py \
    --model_type resnet \
    --weight_path /path/to/your/resnet18-fbbb1da6.pth \
    --save_path /path/to/your/resnet18_fdconv.pth
```

This will find weights like layer1.0.conv1.weight, convert them to layer1.0.conv1.dft_weight, and save the complete modified state dictionary to resnet18_fdconv.pth.

#### Example 2: Converting a SegFormer (MiT-B0) Model

Transformer-based models like SegFormer use linear layers in their Feed-Forward Networks (FFNs), which can be replaced by FDConv (as 1x1 convolutions).

1. Download the pre-trained MiT-B0 weights (e.g., mit_b0.pth).
2. Run the script with mit type:

Generated bash

```
python tools/convert_to_fdconv.py \
    --model_type vit \
    --weight_path /path/to/your/mit_b0.pth \
    --save_path /path/to/your/mit_b0_fdconv.pth
```

This will target the FFN linear layers (e.g., block1.0.ffn.layers.0.0.weight), convert them, and save the new checkpoint.

After conversion, you can load the generated .pth file into your model architecture where standard layers have been replaced by FDConv.



## 📖 Citation

If you find this work useful, please cite:

```
@inproceedings{chen2025frequency,
  title={Frequency Dynamic Convolution for Dense Image Prediction},
  author={Chen, Linwei and Gu, Lin and Li, Liang and Yan, Chenggang and Fu, Ying},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages={30178--30188},
  year={2025}
}
```

## Acknowledgment

This code is built using [mmsegmentation](https://github.com/open-mmlab/mmsegmentation), [mmdetection](https://github.com/open-mmlab/mmdetection/tree/dev-2.x) libraries.

## Contact

If you encounter any problems or bugs, please don't hesitate to contact me at [chenlinwei@bit.edu.cn](chenlinwei@bit.edu.cn), [charleschen2013@163.com](charleschen2013@163.com). To ensure effective assistance, please provide a brief self-introduction, including your name, affiliation, and position. If you would like more in-depth help, feel free to provide additional information such as your personal website link. I would be happy to discuss with you and offer support.