import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.checkpoint import checkpoint


class Qwen2RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Qwen2RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"
    
class StarReLU(nn.Module):
    """
    StarReLU: s * relu(x) ** 2 + b
    """
    def __init__(self, scale_value=1.0, bias_value=0.0,
                 scale_learnable=True, bias_learnable=True,
                 inplace=False):
        super().__init__()
        self.inplace = inplace
        self.relu = nn.ReLU(inplace=inplace)
        self.scale = nn.Parameter(scale_value * torch.ones(1),
                                  requires_grad=scale_learnable)
        self.bias = nn.Parameter(bias_value * torch.ones(1),
                                 requires_grad=bias_learnable)
    def reset_parameters(self):
        pass

    def forward(self, x):
        return self.scale * self.relu(x) ** 2 + self.bias

def get_fft2freq(d1, d2, use_rfft=False):
    # 生成频域坐标，用于权重压缩索引
    freq_h = torch.fft.fftfreq(d1)
    if use_rfft:
        freq_w = torch.fft.rfftfreq(d2)
    else:
        freq_w = torch.fft.fftfreq(d2)
    
    freq_hw = torch.stack(torch.meshgrid(freq_h, freq_w, indexing='ij'), dim=-1)
    dist = torch.norm(freq_hw, dim=-1)
    sorted_dist, indices = torch.sort(dist.view(-1))
    
    if use_rfft:
        d2 = d2 // 2 + 1
    sorted_coords = torch.stack([indices // d2, indices % d2], dim=-1)
    return sorted_coords.permute(1, 0), freq_hw

class KernelSpectralModulation_Linear(nn.Module):
    """
    适配 Linear 的注意力生成模块
    输入: (B, C_in) 或 (B, N, C_in) 的全局特征
    输出: Channel Att, Filter Att, Kernel Att
    """
    def __init__(self, in_features, out_features, reduction=0.0625, kernel_num=4, min_channel=16, 
                 temp=1.0, kernel_temp=None, att_multi=2.0, act_type='sigmoid'):
        super().__init__()
        attention_channel = max(int(in_features * reduction), min_channel)
        self.act_type = act_type
        self.kernel_num = kernel_num
        self.temperature = temp
        self.kernel_temp = kernel_temp
        self.att_multi = att_multi

        # 压缩特征
        self.fc = nn.Linear(in_features, attention_channel, bias=False)
        # self.norm = nn.Identity()
        self.norm = Qwen2RMSNorm(attention_channel)
        # self.relu = StarReLU()
        self.relu = nn.ReLU(inplace=True)

        # 生成 Channel Attention (针对输入维度)
        self.channel_fc = nn.Linear(attention_channel, in_features, bias=True)
        
        # 生成 Filter Attention (针对输出维度)
        self.filter_fc = nn.Linear(attention_channel, out_features, bias=True)

        # 生成 Kernel Attention (针对动态基)
        # 这里的 kernel 其实是指 spectral basis component
        if kernel_num > 1:
            self.kernel_fc = nn.Linear(attention_channel, kernel_num, bias=True)
        else:
            self.func_kernel = self.skip

        self.reset_parameters()

    def reset_parameters(self):
        # 1. 先让子层进行默认初始化（防止 LayerNorm 等出问题）
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        # 2. 【核心修复】强制将你的关键层重置为 0
        # 无论 FSDP 何时调用此方法，都会执行这几行
        # print(f"DEBUG: Resetting KSM weights to ZERO for {self}")
        nn.init.constant_(self.channel_fc.weight, 0)
        nn.init.constant_(self.filter_fc.weight, 0)
        nn.init.constant_(self.kernel_fc.weight, 0)
        
        # 别忘了 Bias 也要设为 0，否则 Sigmoid(Bias) 可能是 0.5
        if self.channel_fc.bias is not None:
            nn.init.constant_(self.channel_fc.bias, 0)
        if self.filter_fc.bias is not None:
            nn.init.constant_(self.filter_fc.bias, 0)
        if self.kernel_fc.bias is not None:
            nn.init.constant_(self.kernel_fc.bias, 0)

    @staticmethod
    def skip(_):
        return 1.0

    def get_attention(self, fc_layer, x, temp, act_type, multi=1.0):
        out = fc_layer(x)
        if act_type == 'sigmoid':
            return torch.sigmoid(out / temp) * multi
        elif act_type == 'tanh':
            return 1 + torch.tanh(out / temp)
        elif act_type == 'softmax':
            return F.softmax(out / temp, dim=1)
        else:
            raise NotImplementedError

    def get_kernel_attention(self, x):
        out = self.kernel_fc(x)
        if self.act_type == 'softmax':
            return F.softmax(out / self.kernel_temp, dim=1) * self.kernel_num
        elif self.act_type == 'sigmoid':
            # print('kernel out:', out.shape)
            return torch.sigmoid(out / self.kernel_temp) * 2
        elif self.act_type == 'tanh':
            return (1 + torch.tanh(out / self.kernel_temp))
        return out

    def forward(self, x, use_checkpoint=False):
        if use_checkpoint:
            return checkpoint(self._forward, x)
        else:
            return self._forward(x)
        
    def _forward(self, x):
        # x shape: (B, C_in)
        avg_x = self.relu(self.norm(self.fc(x)))
        
        channel_att = self.get_attention(self.channel_fc, avg_x, self.temperature, self.act_type, self.att_multi)
        filter_att = self.get_attention(self.filter_fc, avg_x, self.temperature, self.act_type, self.att_multi)
        
        if hasattr(self, 'kernel_fc'):
            kernel_att = self.get_kernel_attention(avg_x)
        else:
            kernel_att = 1.0
            
        return channel_att, filter_att, kernel_att

def report_stats(tag, tensor):
    if tensor is None:
        return
    
    # 【修复 1】只检查 tensor.is_meta
    if tensor.is_meta or tensor.device.type == 'meta':
        # 仅在调试初始化时打开，训练时太吵可以注释掉
        # print(f"    -> {tag:20s}: [Meta Device] Tensor is on meta device.")
        return

    # 【修复 2】只在主进程打印 (防止多卡训练刷屏)
    # 如果没有 rank 信息，默认打印。Verl/Ray 环境通常有 local_rank 环境变量
    import os
    if os.environ.get("RANK", "0") != "0":
        return

    # detach()防止追踪梯度，float()确保精度
    t = tensor.detach().cpu().float()
    
    # 【新增】检查是否有梯度（仅针对 Parameter 或 retain_grad 的 Tensor）
    grad_info = ""
    if tensor.requires_grad:
        if tensor.grad is not None:
            g = tensor.grad.detach().cpu().float()
            grad_info = f" | Grad Norm={g.norm().item():.6e}"
        else:
            grad_info = " | Grad=None"
            
    print(f"    -> {tag:20s}: Mean={t.mean():.6e} | Std={t.std():.6e} | Min={t.min():.6e} | Max={t.max():.6e}{grad_info}")

class FDLinear(nn.Linear):
    def __init__(self, 
                 in_features, 
                 out_features, 
                 bias=True,
                 reduction=0.0625, 
                 kernel_num=4,
                 kernel_temp=1.0,
                 temp=None,
                 att_multi=2.0,
                 param_ratio=1,
                 param_reduction=1.0,
                 ksm_global_act='sigmoid',
                 convert_param=True,
                 use_ksm_local=False, # 这里的 Local KSM 对 Linear 意义较小，设为可选
                 **kwargs,
                 ):
        # 调用父类初始化，注册 self.weight 和 self.bias
        super().__init__(in_features, out_features, bias=bias)
        
        self.kernel_num = kernel_num
        self.param_ratio = param_ratio
        self.param_reduction = param_reduction
        self.att_multi = att_multi
        self.ksm_global_act = ksm_global_act
        self.use_ksm_local = use_ksm_local

        if self.kernel_num is None:
            self.kernel_num = max(4, self.out_features // 16)
        
        if temp is None:
            temp = kernel_temp
        if kernel_temp is None:
            kernel_temp = math.sqrt(self.kernel_num * self.param_ratio)

        # 用于缩放 DFT 权重的 alpha
        self.alpha = min(self.out_features, self.in_features) // 2 * self.param_ratio / param_reduction
        
        # 初始化注意力模块
        self.KSM_Global = KernelSpectralModulation_Linear(
            in_features, out_features, 
            reduction=reduction, 
            kernel_num=self.kernel_num * self.param_ratio,
            temp=temp, kernel_temp=kernel_temp, 
            att_multi=att_multi, 
            act_type=ksm_global_act
        )
        
        if self.use_ksm_local:
            # 简单的局部通道交互，类似 ECA
            _k = round((math.log2(self.in_features) / 2) + 0.5) // 2 * 2 + 1
            self.local_conv = nn.Conv1d(1, self.out_features, kernel_size=_k, padding=_k // 2, bias=False)
            nn.init.constant_(self.local_conv.weight, 1e-6)

        # 将标准 Linear 权重转换为频域动态参数
        self.convert2dftweight(convert_param)

    def convert2dftweight(self, convert_param):
        # 针对 Linear，权重形状为 (out_features, in_features)
        # 我们将其视为 2D 图像进行 FFT
        d1, d2 = self.out_features, self.in_features
        
        # 获取频域索引 (out, in_rfft)
        freq_indices, _ = get_fft2freq(d1, d2, use_rfft=True) 
        # print(d1, d2)
        # print(freq_indices.shape)
        
        weight = self.weight # (out, in)
        # rfft2 默认在最后两个维度进行
        weight_rfft = torch.fft.rfft2(weight.float()) # (out, in//2 + 1)
        
        if self.param_reduction < 1:
            num_to_keep = int(freq_indices.size(1) * self.param_reduction)
            freq_indices = freq_indices[:, :num_to_keep] # 保留低频分量
            
            # 提取对应的频域系数
            weight_rfft = torch.stack([weight_rfft.real, weight_rfft.imag], dim=-1)
            weight_rfft = weight_rfft[freq_indices[0, :], freq_indices[1, :]]
            
            # 重复以适应 param_ratio
            weight_rfft = weight_rfft.reshape(-1, 2)[None, ].repeat(self.param_ratio, 1, 1) / (min(self.out_features, self.in_features) // 2)
        else:
            weight_rfft = torch.stack([weight_rfft.real, weight_rfft.imag], dim=-1)[None, ].repeat(self.param_ratio, 1, 1, 1) / (min(self.out_features, self.in_features) // 2)

        if convert_param:
            self.dft_weight = nn.Parameter(weight_rfft, requires_grad=True)
            # 删除原始权重，释放内存并防止优化器更新
            del self.weight
        else:
            # 如果不转换，保留原始权重作为 Parameter
            pass
            
        # 注册索引 buffer
        indices = []
        for i in range(self.param_ratio):
            indices.append(freq_indices.reshape(2, self.kernel_num, -1)) 
        self.register_buffer('indices', torch.stack(indices, dim=0), persistent=False)

    def get_FDW(self):
        # 兼容性函数：如果未转换参数，动态计算

        d1, d2 = self.out_features, self.in_features
        weight = self.weight
        weight_rfft = torch.fft.rfft2(weight.float())
        weight_rfft = torch.stack([weight_rfft.real, weight_rfft.imag], dim=-1)[None, ].repeat(self.param_ratio, 1, 1, 1) / (min(d1, d2) // 2)
        # print('weight_rfft mean std:', weight_rfft.mean(), weight_rfft.std())
        # print('self.weight mean std:', self.weight.mean(), self.weight.std())
        return weight_rfft

    def old_forward(self, x):
        print('x:', x.shape)
        # x shape: (B, N, C_in) 或 (B, C_in)
        # 为了 KSM，我们需要一个全局上下文向量 (B, C_in)
        if x.dim() == 3:
            # (B, N, C) -> (B, C)
            global_x = x.mean(dim=1)
        elif x.dim() == 2:
            x = x[None, ]
            global_x = x.mean(dim=1)
        else:
            global_x = x
            
        # 生成注意力
        channel_att, filter_att, kernel_att = self.KSM_Global(global_x)
        # print(channel_att.shape, filter_att.shape, kernel_att.shape)
        # channel_att: (B, C_in)
        # filter_att: (B, C_out)
        # kernel_att: (B, param_ratio * kernel_num) -> reshape later
        
        if self.use_ksm_local:
            # 简单的 local attention 补充
            # (B, C) -> (B, 1, C) -> Conv1d -> (B, Cout, C)
            local_att = self.local_conv(global_x.unsqueeze(1)).sigmoid() * self.att_multi

        b = x.size(0)
        
        # 准备频域图
        # 大小为 (B, C_out, C_in_rfft, 2)
        # rfft 后最后一维大小为 in_features // 2 + 1
        dft_shape = (b, self.out_features, self.in_features // 2 + 1, 2)
        DFT_map = torch.zeros(dft_shape, device=x.device)
        
        kernel_att = kernel_att.reshape(b, self.param_ratio, self.kernel_num, -1)
        
        if hasattr(self, 'dft_weight'):
            dft_weight = self.dft_weight
        else:
            dft_weight = self.get_FDW()

        # 动态重构权重
        for i in range(self.param_ratio):
            indices = self.indices[i] # (2, kernel_num, num_freqs)
            
            # 获取当前 ratio 的 kernel attention: (B, kernel_num, 1) (假设广播)
            # 注意: kernel_att 是 (B, param_ratio, kernel_num, 1) (如果 reshape 正确)
            # 上面 reshape 导致最后一维可能是 1 或其他，取决于 total output size
            # 简化逻辑：kernel_att [:, i] 是 (B, kernel_num, chunk_size)
            
            k_att = kernel_att[:, i] # (B, kernel_num, -1)

            if self.param_reduction < 1:
                # 稀疏模式
                w = dft_weight[i].reshape(self.kernel_num, -1, 2)[None] # (1, kn, freq, 2)
                # k_att 广播乘 w
                # 结果加到 DFT_map 的特定索引位置
                # 索引操作比较 tricky，因为有 batch 维。
                # 我们先计算加权的频域值 (B, kn, freq, 2) -> sum over kn -> (B, freq, 2)
                weighted_freqs = torch.sum(w * k_att.unsqueeze(-1), dim=1) # (B, freq, 2)
                
                # 赋值 (需要处理 batch 索引)
                # DFT_map[:, r_idx, c_idx] = weighted_freqs
                # indices[0] 是 row (out_feature) 索引, indices[1] 是 col (in_feature freq) 索引
                # flatten indices for indexing
                flat_indices_r = indices[0].flatten()
                flat_indices_c = indices[1].flatten()
                DFT_map[:, flat_indices_r, flat_indices_c] += weighted_freqs
                
            else:
                # 稠密模式 (还原完整矩阵)
                # w: (out, in_freq, 2)
                # w 是被切分成 kernel_num 块来组合的
                # 这里简化：假设 dft_weight[i] 直接存储了完整频域图的基
                # 但原始代码逻辑是将权重拆解。
                # 让我们遵循原始逻辑：dft_weight[i] 是压缩后的存储
                
                # 提取当前 indices 对应的权重基
                w_basis = dft_weight[i][indices[0], indices[1]] # (kernel_num, num_points_per_kernel, 2)
                w_basis = w_basis.unsqueeze(0) # (1, kn, pts, 2)
                
                print(indices.shape)
                print(w_basis.shape)
                print(dft_weight.shape)
                print(DFT_map.shape)
                # 加权: (B, kn, 1, 1) * (1, kn, pts, 2) -> sum(dim=1) -> (B, pts, 2)
                # k_att 需要匹配形状
                k_att_expanded = k_att.view(b, self.kernel_num, -1).unsqueeze(-1)
                # weighted_freqs = torch.sum(w_basis * k_att_expanded, dim=1) * self.alpha
                print(k_att_expanded.shape)
                
                DFT_map[:, indices[0], indices[1]] += w_basis * k_att_expanded * self.alpha

        # iRFFT 恢复到空间域 (权重矩阵域)
        # input: (B, out, in_rfft) complex
        adaptive_weights = torch.fft.irfft2(torch.view_as_complex(DFT_map), s=(self.out_features, self.in_features))
        # adaptive_weights shape: (B, out_features, in_features)

        # 应用 Channel 和 Filter Attention
        # channel_att: (B, in) -> (B, 1, in)
        # filter_att: (B, out) -> (B, out, 1)
        print(local_att.shape)
        print(adaptive_weights.shape)
        aggregate_weight = adaptive_weights * channel_att.unsqueeze(1) * filter_att.unsqueeze(2) * local_att
        
        print(self.weight)
        # 执行 Batch Linear 变换
        # x: (B, N, in) 或 (B, in)
        # weight: (B, out, in)
        # 目标: x @ weight.T -> (B, N, out)
        
        if x.dim() == 2:
            x = x.unsqueeze(1) # (B, 1, in)
            output = torch.bmm(x, aggregate_weight.transpose(1, 2)) # (B, 1, out)
            output = output.squeeze(1)
        else:
            # (B, N, in) @ (B, in, out)
            output = torch.bmm(x, aggregate_weight.transpose(1, 2))

        if self.bias is not None:
            output = output + self.bias

        return output

    def _generate_dynamic_weights(self, global_x):
        """
        生成动态权重。
        Optimization: Explicitly ensure dtype matches to avoid float32 bloat.
        """
        b = global_x.size(0)
        target_dtype = global_x.dtype # e.g., bfloat16

        # 1. Attention
        channel_att, filter_att, kernel_att = self.KSM_Global(global_x)

        # =========== 【新增调试代码 Start】 ===========
        # 我们不看 kernel_att (中间变量)，我们看生成它的权重 (叶子节点)
        # 只有 Rank 0 打印，避免刷屏
        # import os
        # if self.training and os.environ.get("RANK", "0") == "0":
        #     # 只有当 KSM_Global 里的权重有梯度时才打印
        #     if hasattr(self.KSM_Global, 'kernel_fc'):
        #         weight_param = self.KSM_Global.kernel_fc.weight
        #         if weight_param.grad is not None:
        #             g_norm = weight_param.grad.detach().norm().item()
        #             w_norm = weight_param.detach().norm().item()
        #             # 打印权重的梯度范数，这才是判断是否学习的金标准
        #             print(f"DEBUG [FDLinear]: kernel_fc weight | Val Norm: {w_norm:.6e} | Grad Norm: {g_norm:.6e}")
        #         else:
        #             print(f"DEBUG [FDLinear]: kernel_fc weight no grad yet")
        #             # 刚开始几个 step 可能是 None，或者 accum step 还没结束
        #             pass
        # =========== 【新增调试代码 End】 =============
        
        if self.use_ksm_local:
            local_att = self.local_conv(global_x.unsqueeze(1)).sigmoid() * self.att_multi
        else:
            local_att = 1.0

        # 2. Reconstruct DFT Map
        # Important: Use target_dtype for zeros to save memory
        dft_shape = (b, self.out_features, self.in_features // 2 + 1, 2)
        DFT_map = torch.zeros(dft_shape, device=global_x.device, dtype=torch.float32) # FFT usually needs float32
        
        kernel_att = kernel_att.reshape(b, self.param_ratio, self.kernel_num, -1)
        
        if hasattr(self, 'dft_weight'):
            dft_weight = self.dft_weight
        else:
            dft_weight = self.get_FDW()

        # Optimize loop to reduce peak memory if possible, but mainly rely on checkpointing outside
        for i in range(self.param_ratio):
            indices = self.indices[i]
            k_att = kernel_att[:, i]

            if self.param_reduction < 1:
                w = dft_weight[i].reshape(self.kernel_num, -1, 2)[None] 
                weighted_freqs = torch.sum(w * k_att.unsqueeze(-1), dim=1)
                flat_indices_r = indices[0].flatten()
                flat_indices_c = indices[1].flatten()
                DFT_map[:, flat_indices_r, flat_indices_c] += weighted_freqs
            else:
                w_basis = dft_weight[i][indices[0], indices[1]]
                w_basis = w_basis.unsqueeze(0)
                k_att_expanded = k_att.view(b, self.kernel_num, -1).unsqueeze(-1)
                
                # Careful accumulation to avoid intermediate explosions
                DFT_map[:, indices[0], indices[1]] += (w_basis * k_att_expanded * self.alpha).to(DFT_map.dtype)

        # 3. iRFFT
        # Convert to complex
        DFT_map_complex = torch.view_as_complex(DFT_map)
        adaptive_weights = torch.fft.irfft2(DFT_map_complex, s=(self.out_features, self.in_features))
        
        # Cast back to low precision (bf16/fp16) immediately
        adaptive_weights = adaptive_weights.to(dtype=target_dtype)
        
        # Free memory
        del DFT_map, DFT_map_complex
        # print('channel_att, filter_att, kernel_att, mean:', channel_att.mean(), filter_att.mean(), kernel_att.mean())
        # print('channel_att, filter_att, kernel_att, std:', channel_att.std(), filter_att.std(), kernel_att.std())

        # 4. Apply Modulation
        aggregate_weight = adaptive_weights * channel_att.unsqueeze(1) * filter_att.unsqueeze(2) * local_att
        
        # """
        # for test
        # aggregate_weight = adaptive_weights 
        # print('global_x, mean, std:', global_x.mean(), global_x.std())
        # print('channel_att, mean, std:', channel_att.mean(), channel_att.std())
        # print('filter_att, mean, std:', filter_att.mean(), filter_att.std())
        report_stats("kernel_att", k_att)
        # print('kernel_att, mean, std:', k_att.mean(), k_att.std())
        # print('weight.shape', aggregate_weight.shape, self.weight.shape)
        # print('weight diff std:', (aggregate_weight[0] - self.weight).std())
        # print('weight std:', (aggregate_weight).std())
        # print('weight mean:', aggregate_weight.mean(), self.weight.mean())
        # print('weight ratio:', aggregate_weight.mean() / self.weight.mean())
        # print('alpha:', self.alpha)
        # """
        
        return aggregate_weight

    def _compute_segment_forward(self, seg_x, global_x_i):
        """
        Helper function to be checkpointed.
        Input: 
            seg_x: (L, In)
            global_x_i: (1, In)
        Output:
            out: (L, Out)
        """
        # 1. Generate FULL dense weight matrix for this segment
        # This allocates [Out, In] memory.
        seg_w_wrapper = self._generate_dynamic_weights(global_x_i) 
        seg_w = seg_w_wrapper[0] # (Out, In)

        # 2. Compute Linear
        out = F.linear(seg_x, seg_w, bias=None)
        
        # By returning 'out' and exiting this function, 'seg_w' goes out of scope.
        # When using checkpoint(), 'seg_w' is NOT saved for backward.
        # Instead, 'seg_x' and 'global_x_i' are saved, and this function is re-run during backward.
        return out

    def forward(self, x, cu_seqlens=None):
        """
        x: (Total_Tokens, C) [Packed]
        cu_seqlens: Optional[Tensor]
        """
        # === Packed Sequence Processing (Critical for Qwen2-VL) ===
        if cu_seqlens is not None and x.dim() == 2:
            num_segments = len(cu_seqlens) - 1
            outputs = []
            bias = self.bias
            
            for i in range(num_segments):
                start, end = cu_seqlens[i], cu_seqlens[i+1]
                if start == end: continue
                
                seg_x = x[start:end]         # (L_seg, In)
                
                if seg_x.shape[0] > 0:
                    # 【核心修复】强制转换为 float32 进行均值计算，防止 BF16 溢出导致 NaN
                    # 计算完后再转回 x.dtype (即 bfloat16)
                    # global_x_i = seg_x.to(torch.float32).mean(dim=0, keepdim=True).to(x.dtype)
                    global_x_i = seg_x.mean(dim=0, keepdim=True)
                else:
                    global_x_i = torch.zeros(1, self.in_features, device=x.device, dtype=x.dtype)

                # CRITICAL FIX: Use checkpointing for the heavy weight generation + matmul
                # This prevents storing the Dense Weight Matrix (500MB+) per segment for backward pass
                if self.training and seg_x.requires_grad:
                    out = checkpoint(self._compute_segment_forward, seg_x, global_x_i, use_reentrant=False)
                else:
                    out = self._compute_segment_forward(seg_x, global_x_i)
                
                outputs.append(out)
            
            if len(outputs) > 0:
                final_output = torch.cat(outputs, dim=0)
            else:
                final_output = torch.empty(0, self.out_features, device=x.device, dtype=x.dtype)
            
            if bias is not None:
                final_output = final_output + bias
                
            return final_output

        # === Standard Batch Processing (Legacy/Testing) ===
        else:
            if x.dim() == 3:
                # 【核心修复】同样适用于 Batch 模式
                # global_x = x.to(torch.float32).mean(dim=1).to(x.dtype)
                global_x = x.mean(dim=1)
            else:
                global_x = x
            
            # For standard batch, we also checkpoint to save memory
            def _batch_forward(batch_x, g_x):
                w = self._generate_dynamic_weights(g_x)
                if batch_x.dim() == 2:
                    return torch.bmm(batch_x.unsqueeze(1), w.transpose(1, 2)).squeeze(1)
                else:
                    return torch.bmm(batch_x, w.transpose(1, 2))

            if self.training and x.requires_grad:
                output = checkpoint(_batch_forward, x, global_x, use_reentrant=False)
            else:
                output = _batch_forward(x, global_x)

            if self.bias is not None:
                output = output + self.bias

            return output

    def extra_repr(self):
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}, kernel_num={self.kernel_num}'