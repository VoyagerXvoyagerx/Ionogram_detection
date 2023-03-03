_base_ = '../yolov5/yolov5_m-v61_syncbn_fast_8xb16-300e_coco.py'

'''
modify
max_epochs, work_dirs, val_begin, load_from, loss, test_dataloader, test_evaluator
flip
'''

max_epochs = 100  # 训练的最大 epoch
data_root = './Iono4311/'  # 数据集目录的绝对路径
work_dir = './work_dirs/yolov5_m_100e'

# 因为本教程是在 cat 数据集上微调，故这里需要使用 `load_from` 来加载 MMYOLO 中的预训练模型，这样可以在加快收敛速度的同时保证精度
load_from = './work_dirs/yolov5_m-v61_syncbn_fast_8xb16-300e_coco_20220917_204944-516a710f.pth'  # noqa

# 根据自己的 GPU 情况，修改 batch size，YOLOv5-s 默认为 8卡 x 16bs
train_batch_size_per_gpu = 32
train_num_workers = 4  # 推荐使用 train_num_workers = nGPU x 4

save_epoch_intervals = 2  # 每 interval 轮迭代进行一次保存一次权重

# 根据自己的 GPU 情况，修改 base_lr，修改的比例是 base_lr_default * (your_bs 32 / default_bs (8x16))
base_lr = _base_.base_lr / 4

# anchors = [
#     [[8, 6], [24, 4], [19, 9]],
#     [[22, 19], [17, 49], [29, 45]],
#     [[44, 66], [96, 76], [126, 59]]
# ]

# anchors = [
#     [[8, 6], [24, 4], [19, 9]],
#     [[22, 19], [17, 49], [29, 45]],
#     [[44, 66], [96, 76], [126, 59]]
# ]
anchors = [[(28, 16), (41, 84), (157, 119)]]

class_name = ('E', 'Es-l', 'Es-c', 'F1', 'F2', 'Spread-F')  # 根据 class_with_id.txt 类别信息，设置 class_name
num_classes = len(class_name)

metainfo = dict(
    classes = class_name,
    palette = [(250, 165, 30), (120, 69, 125), (53, 125, 34), (0, 11, 123), (130, 20, 12), (120, 121, 80)]  # 画图时候的颜色，随便设置即可
)

train_cfg = dict(
    max_epochs=max_epochs,
    val_begin=20,  # 第几个 epoch 后验证，这里设置 20 是因为前 20 个 epoch 精度不高，测试意义不大，故跳过
    val_interval=save_epoch_intervals  # 每 val_interval 轮迭代进行一次测试评估
)

model = dict(
    bbox_head=dict(
        head_module=dict(num_classes=num_classes),
        prior_generator=dict(base_sizes=anchors),

        # loss_cls 会根据 num_classes 动态调整，但是 num_classes = 1 的时候，loss_cls 恒为 0
        loss_cls=dict(loss_weight=0.3 *
                      (num_classes / 80 * 3 / _base_.num_det_layers))))

train_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    num_workers=train_num_workers,
    dataset=dict(
        _delete_=True,
        type='RepeatDataset',
        # 数据量太少的话，可以使用 RepeatDataset ，在每个 epoch 内重复当前数据集 n 次，这里设置 5 是重复 5 次
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
    # 设置间隔多少个 epoch 保存模型，以及保存模型最多几个，`save_best` 是另外保存最佳模型（推荐）
    checkpoint=dict(
        type='CheckpointHook',
        interval=save_epoch_intervals,
        max_keep_ckpts=1,
        save_best='auto'),
    param_scheduler=dict(max_epochs=max_epochs),
    # logger 输出的间隔 (每个batch)
    logger=dict(type='LoggerHook', interval=50))

visualizer = dict(vis_backends=[dict(type='LocalVisBackend'), dict(type='WandbVisBackend')])
# visualizer = dict(vis_backends=[dict(type='LocalVisBackend'),dict(type='TensorboardVisBackend')])
