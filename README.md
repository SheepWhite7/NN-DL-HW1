# 神经网络和深度学习作业个人作业

## 仓库介绍

本项目为课程神经网络和深度学习作业《从零开始构建三层神经网络分类器，实现图像分类》的代码仓库

* 作业：从零开始构建三层神经网络分类器，实现图像分类

* 任务描述：
  手工搭建三层神经网络分类器，在CIFAR数据集上进行训练以实现图像分类

* 基本要求：
  （1）本次作业要求自主实现反向传播，**不允许使用pytorch，tensorflow**等现成的支持自动微分的深度学习框架，可以使用numpy
  （2）最终提交的代码中应至少包含**模型**、**训练**、**测试**和**参数查找**四个部分，鼓励进行模块化设计
  （3）其中模型部分应允许自定义隐藏层大小、激活函数类型，支持通过反向传播计算给定损失的梯度；训练部分应实现SGD优化器、学习率下降、交叉熵损失和L2正则化，并能根据验证集指标自动保存最优的模型权重；参数查找环节要求调节学习率、隐藏层大小、正则化强度等超参数，观察并记录模型在不同超参数下的性能；测试部分需支持导入训练好的模型，输出在测试集上的分类准确率（Accuracy）

## Requirements

```bash
pip install numpy
pip install scikit-learn
pip install os
pip install yaml

# for visualization
pip install seaborn
pip install matplotlib
```

## 文件说明
```bash
- cifar-10-batches-py  # 数据
- model/  # 模型代码
  - checkpoints/  # 保存的模型参数
    - best_model.npy
    - training_metrics.npz
  - cnn.py  # 卷积神经网络
  - nn.py  # 神经网络
  - config.yaml  # 相关配置
- image  # 可视化图像
- train_nn.py  # nn模型训练主程序
- train_cnn.py  # cnn模型训练主程序
- cnn_cifar10.pth  # 保存cnn模型参数
- test.py  # 模型测试主程序
- data_loader.py  # 读取数据集的函数
- para_search.py  # 模型参数探索主程序
- para_search.log  # 模型参数探索的日志记录
- visualize.py  # 可视化主程序
```

## 一、 模型的训练与测试

### 数据下载

从[CIFAR-10官网](https://www.cs.toronto.edu/~kriz/cifar.html)可以下载CIFAR-10数据集，解压`.gz`文件即可

### 模型训练

* 进入仓库根目录，在config.yaml中配置需要的模型参数以及超参数，运行：
```bash
python train.py
```

生成的模型权重会以`npy`的形式自动保存在`model/checkpoints`文件夹中；训练中产生的loss和Accuracy信息会以`npz`文件的形式保存在`model/checkpoints`文件夹中

### 模型测试

* 模型权重地址：[https://pan.baidu.com/s/12TFqWsPeAtSvVF5i1W4uUw?pwd=tyzx](https://pan.baidu.com/s/12TFqWsPeAtSvVF5i1W4uUw?pwd=tyzx)
* 将模型权重文件放至目录`model/checkpoints`中
* 运行：
```bash
python test.py
```


## 二、模型参数搜索与可视化

### 1. 模型参数搜索
* 在para_search.py中设置想要探索的超参数组合，运行：
```bash
python para_search.py
```
超参数组合及其训练结果会保存para_search.log中

### 2. 训练信息以及模型参数可视化

[`visualize.py`](visualize.py)提供了
- 训练过程信息的可视化（包括loss和Accuracy）
- 对模型训练后各层参数的可视化代码（热力图）


## 更多实验结果详见报告
