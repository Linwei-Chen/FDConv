# Copyright (c) OpenMMLab. All rights reserved.
# This script is modified from the original version to support multiple model architectures
# and provide clearer instructions for converting pre-trained weights to the FDConv format.

import argparse
import copy
import math
import torch
from typing import Dict, Any

def parse_args() -> argparse.Namespace:
    """
    Parses and returns command-line arguments.
    
    (zh) 解析并返回命令行参数。
    """
    parser = argparse.ArgumentParser(
        description='Convert standard pre-trained weights to FDConv Fourier-domain weights.'
    )
    parser.add_argument(
        '--model_type',
        type=str,
        required=True,
        choices=['resnet', 'vit'],
        help="Type of the model architecture to convert. 'resnet' for ResNet-like CNNs, "
             "'vit' for Transformer backbones used in models like SegFormer."
    )
    parser.add_argument(
        '--weight_path',
        type=str,
        required=True,
        help='Path to the original pre-trained weight file (.pth).'
    )
    parser.add_argument(
        '--save_path',
        type=str,
        required=True,
        help='Path to save the converted FDConv weight file (.pth).'
    )
    args = parser.parse_args()
    return args

def get_fft2freq(d1: int, d2: int, use_rfft: bool = False) -> torch.Tensor:
    """
    Generates frequency coordinates and sorts them by their distance from the origin.
    This helps in grouping frequency components, which is a core concept of FDConv's
    Fourier Disjoint Weight (FDW).
    
    (zh) 生成频率坐标，并根据其到原点的距离进行排序。这有助于对频率分量进行分组，
    这是FDConv傅里叶不相交权重（FDW）的核心概念。

    Args:
        d1 (int): The size of the first dimension. (zh) 第一个维度的大小。
        d2 (int): The size of the second dimension. (zh) 第二个维度的大小。
        use_rfft (bool): Whether to use rfft (real-input FFT). (zh) 是否使用实数输入的FFT。

    Returns:
        torch.Tensor: A tensor of shape (2, N) containing sorted 2D frequency coordinates.
                      (zh) 形状为 (2, N) 的张量，包含排序后的2D频率坐标。
    """
    # Frequency components for rows and columns
    # (zh) 行和列的频率分量
    freq_h = torch.fft.fftfreq(d1)
    if use_rfft:
        freq_w = torch.fft.rfftfreq(d2)
    else:
        freq_w = torch.fft.fftfreq(d2)
    
    # Create a 2D grid of frequency coordinates
    # (zh) 创建频率坐标的2D网格
    freq_hw = torch.stack(torch.meshgrid(freq_h, freq_w, indexing='ij'), dim=-1)
    
    # Calculate the L2 norm (distance) from the origin (0,0) in the frequency space
    # (zh) 在频率空间中计算L2范数（到原点(0,0)的距离）
    dist = torch.norm(freq_hw, dim=-1)
    
    # Sort the distances and get the original indices
    # (zh) 对距离进行排序并获取原始索引
    _, indices = torch.sort(dist.view(-1))
    
    # Get the corresponding 2D coordinates for the sorted distances
    # (zh) 获取排序后距离对应的2D坐标
    if use_rfft:
        d2_eff = d2 // 2 + 1
    else:
        d2_eff = d2
        
    sorted_coords = torch.stack([indices // d2_eff, indices % d2_eff], dim=-1)
    
    return sorted_coords.permute(1, 0)

def main():
    """
    Main function to perform the weight conversion.
    (zh) 执行权重转换的主函数。
    """
    args = parse_args()
    print(f"Loading original weights from: {args.weight_path}")
    pth_dict = torch.load(args.weight_path, map_location=torch.device('cpu'))

    # The actual weights are often stored in a 'state_dict' or 'model' key
    # (zh) 实际的权重通常存储在 'state_dict' 或 'model' 键中
    model_weight = pth_dict
    if 'state_dict' in pth_dict:
        model_weight = pth_dict['state_dict']
    elif 'model' in pth_dict:
        model_weight = pth_dict['model']
        
    new_model_weight = copy.deepcopy(model_weight)
    
    print(f"Processing model type: {args.model_type}")
    if args.model_type == 'resnet':
        # Target convolution layers in ResNet blocks
        # (zh) 目标是ResNet块中的卷积层
        target_patterns = ['conv1.weight', 'conv2.weight', 'conv3.weight']
    elif args.model_type == 'vit':
        # Target linear layers in Mix Transformer's FFN, which act as 1x1 convolutions
        # (zh) 目标是Mix Transformer前馈网络中的线性层，它们的作用等同于1x1卷积
        target_patterns = ['.ffn.layers.0.0.weight', '.ffn.layers.1.weight'] # For vit
        # Add older patterns for compatibility if needed
        # target_patterns.extend(['.ffn.layers.0.weight','.ffn.layers.4.weight']) 
    else:
        raise ValueError(f"Unsupported model type: {args.model_type}")
        
    # Thresholds for applying FDConv, as described in the paper
    # (zh) 应用FDConv的阈值，与论文中的描述一致
    CH_THRES = 32  # Minimum channel size
    K_THRES = [1, 3] # Supported kernel sizes

    converted_keys = []

    for k, v in model_weight.items():
        # Check if the current layer's weight should be converted
        # (zh) 检查当前层的权重是否需要转换
        if any(pattern in k for pattern in target_patterns):
            print(f"Found target weight: {k} with shape {v.shape}")

            # Standardize weight tensor to 4D (cout, cin, k_h, k_w)
            # (zh) 将权重张量标准化为4D (cout, cin, k_h, k_w)
            if v.dim() == 2:  # For linear layers (treated as 1x1 conv)
                v = v[..., None, None]
            
            cout, cin, k_size, _ = v.shape

            # Apply conversion only if the layer meets the criteria (channel size, kernel size)
            # (zh) 仅当层符合标准（通道数、核尺寸）时才进行转换
            if min(cout, cin) >= CH_THRES and k_size in K_THRES:
                print(f"--> Converting '{k}' to Fourier domain weight...")
                
                # Reshape weight for 2D FFT: (cout * k_h, cin * k_w)
                # The permutation groups output channels with kernel height, and input channels with kernel width.
                # (zh) 为2D FFT重塑权重：(cout * k_h, cin * k_w)
                # 这个置换操作将输出通道与核高度、输入通道与核宽度组合在一起。
                weight_reshaped = v.permute(0, 2, 1, 3).reshape(cout * k_size, cin * k_size)
                
                # Apply 2D Real FFT
                # (zh) 应用2D实数傅里叶变换
                weight_rfft = torch.fft.rfft2(weight_reshaped, dim=(0, 1))
                print(f"    Shape after rfft2: {weight_rfft.shape}")

                # Stack real and imaginary parts into a new last dimension
                # The result is a real tensor representing complex values.
                # (zh) 将实部和虚部堆叠到一个新的维度，得到一个表示复数值的实数张量。
                dft_weight = torch.stack([weight_rfft.real, weight_rfft.imag], dim=-1)
                
                # Add a new dimension for kernel groups (as in the FDConv paper)
                # (zh) 为核组增加一个新的维度（如FDConv论文所述）
                dft_weight = dft_weight[None]

                # Normalization factor, likely empirically determined
                # (zh) 归一化因子，很可能是通过实验确定的
                norm_factor = (min(cout, cin) // 2)
                if norm_factor > 0:
                    dft_weight /= norm_factor

                # Replace the original weight with the new DFT weight
                # (zh) 用新的DFT权重替换原始权重
                new_key = k.replace('.weight', '.dft_weight')
                new_model_weight[new_key] = dft_weight
                new_model_weight.pop(k) # Remove the old weight
                converted_keys.append((k, new_key))
            else:
                print(f"--> Skipping '{k}': does not meet conversion criteria (channels/kernel size).")

    print("\nConversion summary:")
    if not converted_keys:
        print("No keys were converted. Please check your model_type and weight file.")
    for old_k, new_k in converted_keys:
        print(f"  - Replaced '{old_k}' with '{new_k}' (shape: {new_model_weight[new_k].shape})")

    # Save the new state dictionary
    # (zh) 保存新的状态字典
    print(f"\nSaving converted weights to: {args.save_path}")
    torch.save(new_model_weight, args.save_path)
    print("Done.")

if __name__ == '__main__':
    main()