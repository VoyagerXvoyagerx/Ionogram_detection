_base_ = './yolov7_l_syncbn_fast_1xb32-100e_ionogram.py'

# work_dir and pre-train
load_from = './work_dirs/yolov7_x_syncbn_fast_8x16b-300e_coco_20221124_215331-ef949a68.pth'  # noqa
work_dir = './work_dirs/yolov7_x_100e'
anchors = [
    [(14, 11), (44, 7), (32, 20)],
    [(24, 67), (40, 83), (64, 108)],
    [(117, 118), (190, 92), (185, 142)]]
strides = _base_.strides

model = dict(
    backbone=dict(arch='X'),
    neck=dict(
        in_channels=[640, 1280, 1280],
        out_channels=[160, 320, 640],
        block_cfg=dict(
            type='ELANBlock',
            middle_ratio=0.4,
            block_ratio=0.4,
            num_blocks=3,
            num_convs_in_block=2),
        use_repconv_outs=False),
    bbox_head=dict(
        head_module=dict(in_channels=[320, 640, 1280]),
        prior_generator=dict(
            type='mmdet.YOLOAnchorGenerator',
            base_sizes=anchors,
            strides=strides)))
