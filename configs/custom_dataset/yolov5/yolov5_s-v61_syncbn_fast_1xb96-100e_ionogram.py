_base_ = '../../yolov5/yolov5_s-v61_syncbn_fast_8xb16-300e_coco.py'

max_epochs = 100
data_root = './Iono4311/'
work_dir = './work_dirs/yolov5_s_100e'
load_from = './work_dirs/yolov5_s-v61_syncbn_fast_8xb16-300e_coco_20220918_084700-86e02187.pth'  # noqa

# dataloader, base_lr, YOLOv5-s 8 x 16bs by default
train_batch_size_per_gpu = 96
train_num_workers = 8
# train_batch_size_per_gpu = 64 # for amp = False

# default_hooks
save_epoch_intervals = 10
val_interval = 2
logger_interval = 20
max_keep_ckpts = 1

# base_lr_default * (your_bs 32 / default_bs (8x16))    100/(8*16)
base_lr = _base_.base_lr * train_batch_size_per_gpu / (8 * 16)
checkpoint_interval = 10

# Optimized anchor
anchors = [
    [[8, 6], [24, 4], [19, 9]],
    [[22, 19], [17, 49], [29, 45]],
    [[44, 66], [96, 76], [126, 59]]
]

class_name = ('E', 'Es-l', 'Es-c', 'F1', 'F2', 'Spread-F')
num_classes = len(class_name)

# dataloader
metainfo = dict(
    classes = class_name,
    palette = [(250, 165, 30), (120, 69, 125), (53, 125, 34),
               (0, 11, 123), (130, 20, 12), (120, 121, 80)]
)

train_cfg = dict(
    max_epochs=max_epochs,
    val_begin=20,  
    val_interval=val_interval
)

tta_model = None

model = dict(
    bbox_head=dict(
        head_module=dict(num_classes=num_classes),
        prior_generator=dict(base_sizes=anchors),
        loss_cls=dict(loss_weight=0.5 *
                      (num_classes / 80 * 3 / _base_.num_det_layers))))

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
            pipeline=_base_.train_pipeline)))

val_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    num_workers=train_num_workers,
    dataset=dict(
        metainfo=metainfo,
        data_root=data_root,
        ann_file='annotations/val.json',
        data_prefix=dict(img='val_images/')))

test_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    num_workers=train_num_workers,
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
        save_param_scheduler=None,  # for yolov5
        interval=save_epoch_intervals,
        max_keep_ckpts=max_keep_ckpts,
        save_best='auto'),
    param_scheduler=dict(max_epochs=max_epochs),
    logger=dict(type='LoggerHook', interval=logger_interval))

visualizer = dict(
    vis_backends=[dict(type='LocalVisBackend'),
                  dict(type='WandbVisBackend'), dict(type='TensorboardVisBackend')])