_base_ = '../../yolov6/yolov6_s_syncbn_fast_8xb32-400e_coco.py'

deepen_factor = 0.6
widen_factor = 0.75
affine_scale = 0.9
max_epochs = 100
data_root = './Iono4311/'

work_dir = './work_dirs/yolov6_m_100e'
load_from = './work_dirs/yolov6_m_syncbn_fast_8xb32-300e_coco_20221109_182658-85bda3f4.pth'  # noqa

train_batch_size_per_gpu = 32
train_num_workers = 4

save_epoch_intervals = 2  

#  base_lr_default * (your_bs 32 / default_bs (8 x 32))
base_lr = _base_.base_lr / 8

class_name = ('E', 'Es-l', 'Es-c', 'F1', 'F2', 'Spread-F')
num_classes = len(class_name)
metainfo = dict(
    classes = class_name,
    palette = [(250, 165, 30), (120, 69, 125), (53, 125, 34),
               (0, 11, 123), (130, 20, 12), (120, 121, 80)]
)
image_scale = (400, 360)

train_cfg = dict(
    max_epochs=max_epochs,
    val_begin=30,  
    val_interval=save_epoch_intervals,  
    # dynamic_intervals=[(max_epochs - _base_.num_last_epochs, 1)]
)

model = dict(
    backbone=dict(
        type='YOLOv6CSPBep',
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        hidden_ratio=2. / 3,
        block_cfg=dict(type='RepVGGBlock'),
        act_cfg=dict(type='ReLU', inplace=True)),
    neck=dict(
        type='YOLOv6CSPRepPAFPN',
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        block_cfg=dict(type='RepVGGBlock'),
        hidden_ratio=2. / 3,
        block_act_cfg=dict(type='ReLU', inplace=True)),
    bbox_head=dict(
        type='YOLOv6Head',
        head_module=dict(widen_factor=widen_factor,
            num_classes=num_classes)),
    train_cfg=dict(
        initial_assigner=dict(num_classes=num_classes),
        assigner=dict(num_classes=num_classes)))

mosaic_affine_pipeline = [
    dict(
        type='Mosaic',
        img_scale=_base_.img_scale,
        pad_val=114.0,
        pre_transform=_base_.pre_transform),
    dict(
        type='YOLOv5RandomAffine',
        max_rotate_degree=0.0,
        max_shear_degree=0.0,
        scaling_ratio_range=(1 - affine_scale, 1 + affine_scale),
        border=(-_base_.img_scale[0] // 2, -_base_.img_scale[1] // 2),
        border_val=(114, 114, 114))
]

train_pipeline = [
    *_base_.pre_transform, *mosaic_affine_pipeline,
    dict(
        type='YOLOv5MixUp',
        prob=0.1,
        pre_transform=[*_base_.pre_transform, *mosaic_affine_pipeline]),
    dict(type='YOLOv5HSVRandomAug'),
    dict(type='mmdet.RandomFlip', prob=0.5),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'flip',
                   'flip_direction'))
]

train_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    num_workers=train_num_workers,
    dataset=dict(
        _delete_=True,
        type='RepeatDataset',
        times=1,
        dataset=dict(
            type=_base_.dataset_type,
            data_root=data_root,
            metainfo=metainfo,
            ann_file='annotations/train.json',
            data_prefix=dict(img='train_images/'),
            filter_cfg=dict(filter_empty_gt=False, min_size=32),
            pipeline=train_pipeline)))

val_dataloader = dict(
    dataset=dict(
        metainfo=metainfo,
        data_root=data_root,
        ann_file='annotations/val.json',
        data_prefix=dict(img='val_images/')))

test_dataloader = dict(
    dataset=dict(
        metainfo=metainfo,
        data_root=data_root,
        ann_file='annotations/test.json',
        data_prefix=dict(img='test_images/')))

val_evaluator = dict(ann_file=data_root + 'annotations/val.json')
test_evaluator = dict(ann_file=data_root + 'annotations/test.json')

optim_wrapper = dict(optimizer=dict(lr=base_lr))

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=save_epoch_intervals,
        max_keep_ckpts=3,
        save_best='auto'),
    param_scheduler=dict(max_epochs=max_epochs),
    logger=dict(type='LoggerHook', interval=50))

custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0001,
        update_buffers=True,
        strict_load=False,
        priority=49),
    dict(
        type='mmdet.PipelineSwitchHook',
        switch_epoch=max_epochs - _base_.num_last_epochs,
        switch_pipeline=_base_.train_pipeline_stage2)
]
visualizer = dict(vis_backends=[dict(type='LocalVisBackend'), dict(type='WandbVisBackend')])
