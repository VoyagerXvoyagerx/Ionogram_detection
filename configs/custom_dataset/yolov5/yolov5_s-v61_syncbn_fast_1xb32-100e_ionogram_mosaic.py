_base_ = './yolov5_s-v61_syncbn_fast_1xb96-100e_ionogram.py'

work_dir = './work_dirs/yolov5_s_abl/mosaic_1280'

train_batch_size_per_gpu = 32
base_lr = _base_.base_lr * train_batch_size_per_gpu / _base_.train_batch_size_per_gpu / 2

optim_wrapper = dict(optimizer=dict(lr=base_lr))

train_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=dict(backend='disk')),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Mosaic',
        img_scale=(640, 640),
        pad_val=114.0,
        pre_transform=[
            dict(
                type='LoadImageFromFile',
                file_client_args=dict(backend='disk')),
            dict(type='LoadAnnotations', with_bbox=True)
        ]),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape'))
]

train_dataloader = dict(
    batch_size = train_batch_size_per_gpu,
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

val_dataloader = dict(
    batch_size = train_batch_size_per_gpu
)

test_dataloader = dict(
    batch_size = train_batch_size_per_gpu
)