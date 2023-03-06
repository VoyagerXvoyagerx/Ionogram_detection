# 自定义数据集配置文件修改指南

## 必须修改的项目

- \_base\_
- work_dir

## 模型尺寸不变，修改策略时

继承自修改过的config
根据需要修改

## 修改模型尺寸时

继承自修改过的config

- num_classes related (e.g. loss_cls)
- load_from
- 官方config中的内容

## 使用新的模型训练自定义数据集

继承自官方config

- visualizer
- dataset settings
  - data_root
  - class_name
  - num_classes
  - metainfo
  - img_scale
- train, val, test
  - batch_size, num_workers
  - train_cfg
    - max_epochs, save_epoch_intervals, val_begin
  - default_hooks
    - max_keep_ckpts
    - save_best
  - lr
  - val_dataloder, test_dataloader
    - metainfo
    - root
  - val_evaluator, test_evaluator

```python
data_root = './Iono4311/'
class_name = ('E', 'Es-l', 'Es-c', 'F1', 'F2', 'Spread-F')
num_classes = len(class_name)
metainfo = dict(
    classes = class_name,
    palette = [(250, 165, 30), (120, 69, 125), (53, 125, 34), (0, 11, 123), (130, 20, 12), (120, 121, 80)])
img_scale = (640, 640)
'''
