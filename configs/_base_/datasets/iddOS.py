# dataset settings

#from base_dirs import BASE_DATA_FOLDER

dataset_type = 'XMLDataset'
data_root = '/home/root/'

classes = ('car', 'person','rider','truck','motorcycle','bus','bicycle')

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1000, 600), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1000, 600),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type='RepeatDataset',
        times=3,
        dataset=dict(
            type=dataset_type,
            classes=classes,
            ann_file=data_root + 'IDD_DetectionCS/train.txt',
            img_prefix=data_root + 'IDD_DetectionCS/',
            pipeline=train_pipeline)
        ),
    trainCS=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'IDD_DetectionCS/train.txt',
        img_prefix=data_root + 'IDD_DetectionCS/',
        pipeline=test_pipeline),
    val=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'IDD_DetectionCS/val.txt',
        img_prefix=data_root + 'IDD_DetectionCS/',
        pipeline=test_pipeline),
    testCS=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'IDD_DetectionCS/val.txt',
        img_prefix=data_root + 'IDD_DetectionCS/',
        pipeline=test_pipeline),
    testOS=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'IDD_Detection/val.txt',
        img_prefix=data_root + 'IDD_Detection/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'IDD_DetectionCS/val.txt',
        img_prefix=data_root + 'IDD_DetectionCS/',
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='mAP')
