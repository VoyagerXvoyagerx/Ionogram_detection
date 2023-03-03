_base_ = './rtmdet_l_syncbn_fast_1xb32-100e_ionogram.py'

# work_dir and pre-train
load_from = './work_dirs/rtmdet_x_syncbn_fast_8xb32-300e_coco_20221231_100345-b85cd476.pth'
work_dir = './work_dirs/rtmdet_x_100e'
deepen_factor = 1.33
widen_factor = 1.25

model = dict(
    backbone=dict(deepen_factor=deepen_factor, widen_factor=widen_factor),
    neck=dict(deepen_factor=deepen_factor, widen_factor=widen_factor),
    bbox_head=dict(head_module=dict(widen_factor=widen_factor)))