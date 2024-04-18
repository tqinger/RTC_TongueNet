## 论文:RTC_TongueNet：A tongue image segmentation model based on DeepLabV3 improvement  的代码

## 该项目主要是来自pytorch官方torchvision模块中的源码基础上改进
* https://github.com/pytorch/vision/tree/main/torchvision/models/segmentation

## 环境配置：
* Python3.6/3.7/3.8
* Pytorch1.10
* Ubuntu或Centos(Windows暂不支持多GPU训练)
* 最好使用GPU训练
* 详细环境配置见```requirements.txt```

## 文件结构：
```
  ├── src: 模型的backbone以及DeepLabv3的搭建
  ├── train_utils: 训练、验证以及多GPU训练相关模块
  ├── my_dataset.py: 自定义dataset用于读取VOC数据集
  ├── train.py: 以deeplabv3_resnet50为例进行训练
  ├── train_multi_GPU.py: 针对使用多GPU的用户使用
  ├── predict.py: 简易的预测脚本，使用训练好的权重进行预测测试
  ├── validation.py: 利用训练好的权重验证/测试数据的mIoU等指标，并生成record_mAP.txt文件
  └── pascal_voc_classes.json: pascal_voc标签文件
```

## 预训练权重下载地址：
* 注意：官方提供的预训练权重是在COCO上预训练得到的，训练时只针对和PASCAL VOC相同的类别进行了训练，所以类别数是21(包括背景)
* deeplabv3_resnet50: https://download.pytorch.org/models/deeplabv3_resnet50_coco-cd0a2569.pth
* deeplabv3_resnet101: https://download.pytorch.org/models/deeplabv3_resnet101_coco-586e9e4e.pth
* deeplabv3_mobilenetv3_large_coco: https://download.pytorch.org/models/deeplabv3_mobilenet_v3_large-fc3c493d.pth
* 注意，下载的预训练权重记得要重命名，比如在train.py中读取的是```deeplabv3_resnet50_coco.pth```文件，
  不是```deeplabv3_resnet50_coco-cd0a2569.pth```
 
 
## 数据集，本例程使用的数据集介绍见论文
* 或者请参考我的博文: 


## 注意事项
* 在使用训练脚本时，注意要将'--data-path'(VOC_root)设置为自己存放'VOCdevkit'文件夹所在的**根目录**
* 在使用预测脚本时，要将'weights_path'设置为你自己生成的权重路径。
* 使用validation文件时，注意确保你的验证集或者测试集中必须包含每个类别的目标，并且使用时只需要修改'--num-classes'、'--aux'、'--data-path'和'--weights'即可，其他代码尽量不要改动


