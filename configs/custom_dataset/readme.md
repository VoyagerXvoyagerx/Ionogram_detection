# 基于YOLO系列算法的频高图度量benchmark

## 数据集

电离层频高图是获取电离层实时信息最重要的途径。电离层不规则结构变化特征研究对检测电离层不规则结构，精准提取和度量电离层各层轨迹和关键参数，具有非常重要的研究意义。

利用中国科学院在海南、武汉、怀来获取的不同季节的4311张频高图建立数据集，人工标注出E层、Es-c层、Es-l层、F1层、F2层、Spread F层共6种结构。

数据预览

![标注好的图像示例](./c "fig1")

数据集中各类别实例数量

```python
annotations_all 4311 images
      E  Esl  Esc    F1    F2  Fspread
0  2040  753  893  2059  4177      133

train 3019 images
      E  Esl  Esc    F1    F2  Fspread
0  1436  529  629  1459  2928       91

val 646 images
     E  Esl  Esc   F1   F2  Fspread
0  311  101  137  303  626       20

test 646 images
     E  Esl  Esc   F1   F2  Fspread
0  293  123  127  297  623       22
```

## 实验结果（未完待续）

| Model | epoch(best) | FLOPs(G) | Params(M) | pretrain | val mAP | test mAP | config |
| --- | --- | --- | --- | --- | --- | --- | --- |
| YOLOv5-s | 50(50) | 7.95 | 7.04 | Coco | 0.579 |  | yolov5_s-v61_syncbn_fast_1xb32-50e_ionogram |
| YOLOv5-s | 100(75) | 7.95 | 7.04 | Coco | 0.577  |  | yolov5_s-v61_syncbn_fast_1xb32-100e_ionogram |
| YOLOv5-s | 200(145) | 7.95 | 7.04 | None | 0.565 |  | yolov5_s-v61_syncbn_fast_1xb32-100e_ionogram_pre0 |
| YOLOv5-m | 100(70) | 24.05 | 20.89 | Coco | 0.587  | 0.586 | yolov5_m-v61_syncbn_fast_1xb32-100e_ionogram |
| YOLOv6-s | 100(54) | 24.2 | 18.84 | Coco | 0.584 |  | yolov6_s_syncbn_fast_1xb32-100e_ionogram |
| YOLOv6-s | 200(188) | 24.2 | 18.84 | None | 0.557 |  | yolov6_s_syncbn_fast_1xb32-100e_ionogram_pre0 |
| YOLOv6-m | 100(76) | 37.08 | 44.42 | Coco | 0.590 |  | yolov6_m_syncbn_fast_1xb32-100e_ionogram |
| YOLOv6-l | 100(76) | 71.33 | 58.47 | Coco | 0.605 | 0.597 | yolov6_l_syncbn_fast_1xb32-100e_ionogram |
| YOLOv7-l | 100(88) | 52.42 | 37.22 | Coco | 0.590 |  | yolov7_l_syncbn_fast_1xb32-100e_ionogram |
| YOLOv7-x | 100(58) | 94.27 | 70.85 | Coco | 0.602 |  | yolov7_x_syncbn_fast_1xb32-100e_ionogram |
| rtmdet-l | 100(80) | 79.96 | 52.26 | Coco | 0.601 |  | rtmdet_l_syncbn_fast_1xb32-100e_ionogram |
| rtmdet-x | 100(94) | 141 | 94.79 | Coco | 0.603 |  | rtmdet_x_syncbn_fast_1xb32-100e_ionogram |


[训练过程可视化](https://wandb.ai/19211416/mmyolo-tools/reports/Object-Detection-for-Ionogram-Automatic-Scaling--VmlldzozNTI4NTk5) 

现有的实验结果中，YOLOv6-l的验证集mAP最高。

对比loss下降的过程可以发现，使用预训练权重时，loss下降得更快。可见即使是自然图像数据集上预训练的模型，在雷达图像数据集上微调，也可以加快收敛。
