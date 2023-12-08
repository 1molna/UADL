_base_ = [
    '../_base_/models/retinanet_r50_fpn.py',
    '../_base_/datasets/iddOS.py',
    '../_base_/default_runtime.py'
]
model = dict(bbox_head=dict(num_classes=7))
# optimizer
optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
# actual epoch = 3 * 3 = 9
lr_config = dict(policy='step', step=[3])
# runtime settings
runner = dict(
    type='EpochBasedRunner', max_epochs=7)  # actual epoch = 4 * 3 = 12
