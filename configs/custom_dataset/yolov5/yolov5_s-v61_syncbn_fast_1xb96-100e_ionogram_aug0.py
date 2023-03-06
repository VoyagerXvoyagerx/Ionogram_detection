_base_ = './yolov5_s-v61_syncbn_fast_1xb96-100e_ionogram.py'

work_dir = './work_dirs/yolov5_s_100e_aug0'

train_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=dict(backend='disk')),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='YOLOv5KeepRatioResize', scale=(640, 640)),
    dict(
        type='LetterResize',
        scale=(640, 640),
        allow_scale_up=False,
        pad_val=dict(img=114)),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'pad_param'))
]

train_dataloader = dict(
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='RepeatDataset',
        times=1,
        dataset=dict(
            type='YOLOv5CocoDataset',
            data_root='./Iono4311/',
            metainfo=dict(
                classes=('E', 'Es-l', 'Es-c', 'F1', 'F2', 'Spread-F'),
                palette=[(250, 165, 30), (120, 69, 125), (53, 125, 34),
                         (0, 11, 123), (130, 20, 12), (120, 121, 80)]),
            ann_file='annotations/train.json',
            data_prefix=dict(img='train_images/'),
            filter_cfg=dict(filter_empty_gt=False, min_size=32),
            pipeline=train_pipeline)),
    collate_fn=dict(type='yolov5_collate'))
