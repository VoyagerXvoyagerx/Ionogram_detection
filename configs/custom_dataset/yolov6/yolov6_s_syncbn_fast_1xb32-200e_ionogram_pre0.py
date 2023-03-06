_base_ = './yolov6_s_syncbn_fast_1xb32-100e_ionogram.py'

work_dir = './work_dirs/yolov6_s_200e_pre0'

base_lr = _base_.base_lr * 4

optim_wrapper = dict(optimizer=dict(lr=base_lr))
max_epochs=200

train_cfg = dict(
    max_epochs=max_epochs,
    val_begin=20,
)

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=2,
        max_keep_ckpts=1,
        save_best='auto'),
    param_scheduler=dict(max_epochs=max_epochs),
    logger=dict(type='LoggerHook', interval=50))


load_from = None