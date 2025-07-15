_base_ = [
    '../_base_/models/mask_rcnn_r50_fpn.py',
    '../_base_/datasets/coco_instance.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

import math
KERNEL_NUM = 64
PARAM_RATIO = 1
model = dict(
    backbone=dict(
        type='ResNet_FDConv',
        need_grad_list=['KSM', 'FBM'],
        # with_cp=True,
        conv_cfg={
            'type':'FDConv',
            'kernel_num': KERNEL_NUM,
            'use_fdconv_if_c_gt': 16, 
            'use_fdconv_if_k_in': [1, 3],
            'use_fbm_if_k_in': [3],
            'use_fbm_for_stride':False, 
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
    neck=dict(
            # type='FPN',
            # upsample_cfg={'type':'nearest',},
        )
)

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

# augmentation strategy originates from DETR / Sparse RCNN
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='AutoAugment',
         policies=[
             [
                 dict(type='Resize',
                      img_scale=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                                 (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                                 (736, 1333), (768, 1333), (800, 1333)],
                      multiscale_mode='value',
                      keep_ratio=True)
             ],
             [
                 dict(type='Resize',
                      img_scale=[(400, 1333), (500, 1333), (600, 1333)],
                      multiscale_mode='value',
                      keep_ratio=True),
                 dict(type='RandomCrop',
                      crop_type='absolute_range',
                      crop_size=(384, 600),
                      allow_negative_crop=True),
                 dict(type='Resize',
                      img_scale=[(480, 1333), (512, 1333), (544, 1333),
                                 (576, 1333), (608, 1333), (640, 1333),
                                 (672, 1333), (704, 1333), (736, 1333),
                                 (768, 1333), (800, 1333)],
                      multiscale_mode='value',
                      override=True,
                      keep_ratio=True)
             ]
         ]),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=64),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
]

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=8,
    train=dict(pipeline=train_pipeline),
)

optimizer = dict(
    _delete_=True, 
    type='AdamW', lr=0.0001, betas=(0.9, 0.999), weight_decay=0.05,
        paramwise_cfg=dict(custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            # 'bias': dict(decay_mult=0.),
            # 'attention': dict(lr_mult=2.),
            # 'dft': dict(lr_mult=KERNEL_NUM),
            }))

# optim_wrapper = dict(
#     optimizer = dict(
#                 _delete_=True, 
#                 type='AdamW', lr=0.0001, betas=(0.9, 0.999), weight_decay=0.05,
#                     paramwise_cfg=dict(custom_keys={
#                         'absolute_pos_embed': dict(decay_mult=0.),
#                         'relative_position_bias_table': dict(decay_mult=0.),
#                         'norm': dict(decay_mult=0.),

#                         'attention': dict(lr_mult=2.),
#                         # 'dft': dict(lr_mult=KERNEL_NUM),
#                         })),
#     accumulative_counts=2,
# )
# optimizer_config=dict(
#     _delete_=True, 
#     grad_clip=dict(max_norm=35, norm_type=2)
# )
# optimizer_config = dict(_delete_=True, grad_clip=dict(max_norm=0.1, norm_type=2))

lr_config = dict(
    # warmup_ratio=0.0001,
    step=[10, 11]
    )
runner = dict(type='EpochBasedRunner', max_epochs=12)
# find_unused_parameters = True

log_config = dict(
    # interval=1,
    )


fp16 = dict(loss_scale='dynamic') 