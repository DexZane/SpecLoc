"""RSOD 流程基线：CenterNet-Update + ResNet-50 + FPN。"""

default_scope = 'mmdet'
custom_imports = dict(
    imports=['specloc.registry'],
    allow_failed_imports=False,
)

data_root = 'data/RSOD/'
data_version = 'RSOD-SpecLoc-70-15-15'
data_manifest = 'data/RSOD/manifest.json'
data_validator = 'python scripts/validateRsod.py --root data/RSOD'
require_verified_data = True
require_clean_git = True

classes = ('aircraft', 'oiltank', 'overpass', 'playground')
palette = [
    (220, 20, 60),
    (255, 140, 0),
    (0, 128, 255),
    (0, 170, 90),
]

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(800, 800), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='PhotoMetricDistortion',
        brightness_delta=32,
        contrast_range=(0.5, 1.5),
        saturation_range=(0.5, 1.5),
        hue_delta=18,
    ),
    dict(type='PackDetInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(800, 800), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor'),
    ),
]


def _dataset(ann_file, test_mode=False):
    return dict(
        type='CocoDataset',
        data_root=data_root,
        metainfo=dict(classes=classes, palette=palette),
        ann_file=ann_file,
        data_prefix=dict(img='images/'),
        test_mode=test_mode,
        # RSOD 中 40 张无 XML 的 playground 图像是合法负样本，不能过滤。
        filter_cfg=dict(filter_empty_gt=False, min_size=1),
        pipeline=test_pipeline if test_mode else train_pipeline,
    )


train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=_dataset('annotations/instances_train.json'),
)

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=_dataset('annotations/instances_val.json', test_mode=True),
)

test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=_dataset('annotations/instances_test.json', test_mode=True),
)

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/instances_val.json',
    metric='bbox',
    classwise=True,
    proposal_nums=(100, 300, 1000),
)
test_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/instances_test.json',
    metric='bbox',
    classwise=True,
    proposal_nums=(100, 300, 1000),
)

model = dict(
    type='CenterNet',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32,
    ),
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50'),
    ),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_output',
        num_outs=5,
        init_cfg=dict(type='Caffe2Xavier', layer='Conv2d'),
        relu_before_extra_convs=True,
    ),
    bbox_head=dict(
        type='CenterNetUpdateHead',
        num_classes=4,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        strides=[8, 16, 32, 64, 128],
        loss_cls=dict(
            type='GaussianFocalLoss',
            pos_weight=0.25,
            neg_weight=0.75,
            loss_weight=1.0,
        ),
        loss_bbox=dict(type='GIoULoss', loss_weight=2.0),
    ),
    train_cfg=None,
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.6),
        max_per_img=1000,
    ),
)

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=1e-4,
        betas=(0.9, 0.999),
        weight_decay=0.05,
    ),
    clip_grad=dict(max_norm=35, norm_type=2),
)
param_scheduler = [
    dict(type='LinearLR', start_factor=1e-3, by_epoch=False, begin=0, end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=12,
        by_epoch=True,
        milestones=[8, 11],
        gamma=0.1,
    ),
]
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=12, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
auto_scale_lr = dict(enable=False, base_batch_size=32)

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=20),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,
        max_keep_ckpts=3,
        save_best='coco/bbox_mAP',
        rule='greater',
    ),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='DetVisualizationHook'),
)
custom_hooks = [dict(type='ExperimentMetadataHook')]

randomness = dict(seed=42, deterministic=False)
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='DetLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer',
)
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)
log_level = 'INFO'
load_from = None
resume = False
work_dir = './work_dirs/rsod_centernet_r50'
