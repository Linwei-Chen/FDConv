_base_ = [
'../_base_/models/faster_rcnn_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=8,
)

KERNEL_NUM = 64
PARAM_RATIO = 1
model = dict(
    backbone=dict(
        # type='ResNet',
        type='ResNet_FDConv',
        # use_checkpoint=True,
        need_grad_list=['KSM', 'FBM'],
        with_cp=True,
        conv_cfg={
            'type':'FDConv',
            'kernel_num': KERNEL_NUM,
            'use_fdconv_if_c_gt': 16, 
            'use_fdconv_if_k_in': [1, 3],
            'use_fbm_if_k_in': [3],
            'temp':1,
            # 'use_rfft':True,
            # 'use_convert':False,
            'att_multi':2.0,
            'param_ratio': PARAM_RATIO,
            'param_reduction': 1.0,
            'kernel_temp':1.0,
            'temp': None,
            'att_multi':2.0,
            'ksm_only_kernel_att': False,
            'use_ksm_local': True,
            'ksm_local_act': 'sigmoid',
            'ksm_global_act': 'sigmoid',
            'spatial_freq_decompose': False,
            'convert_param': True,
            'linear_mode': False,
            'fbm_cfg':{
                'k_list':[2, 4, 8],
                'lowfreq_att':False,
                'fs_feat':'feat',
                'act':'sigmoid',
                'spatial':'conv',
                'spatial_group':1,
                'spatial_kernel':3,
                'init':'zero',
                },
        },
    ),
)
# fp16 = dict(loss_scale='dynamic') 