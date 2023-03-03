_base_ = './yolov6_s_syncbn_fast_1xb32-100e_ionogram.py'

work_dir = './work_dirs/yolov6_s_200e_pre0'

base_lr = _base_.base_lr * 4

optim_wrapper = dict(optimizer=dict(lr=base_lr))
max_epochs=200

train_cfg = dict(
    max_epochs=max_epochs,
    val_begin=20,  # 第几个 epoch 后验证，这里设置 20 是因为前 20 个 epoch 精度不高，测试意义不大，故跳过
)

default_hooks = dict(
    # 设置间隔多少个 epoch 保存模型，以及保存模型最多几个，`save_best` 是另外保存最佳模型（推荐）
    checkpoint=dict(
        type='CheckpointHook',
        interval=2,
        max_keep_ckpts=1,
        save_best='auto'),
    param_scheduler=dict(max_epochs=max_epochs),
    # logger 输出的间隔 (每个batch)
    logger=dict(type='LoggerHook', interval=50))


load_from = None