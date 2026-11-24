# note_image_recognition
my note about image recognition
# 深度学习图像识别：PyTorch 与 CNN 实训笔记

本仓库用于记录《深度学习图像识别》课程的核心原理笔记与代码复盘，涵盖 PyTorch 基础、神经网络底层反向传播推导与实现，以及经典卷积神经网络（CNN / AlexNet）构建。

## 📚 笔记目录与学习进度

### 第一阶段：PyTorch 核心基础
*对应笔记：[01-pytorch-basics.md](./notes/01-pytorch-basics.md)*
- [ ] **01. Pytorch 之 Tensor 初探**：张量创建、索引、维度变换与常用运算
- [ ] **02. Pytorch 之 autograd**：计算图与自动求导机制
- [ ] **03. Pytorch 之 torch.nn 初探**：网络层定义与模块封装
- [ ] **04. Pytorch 之 torch.nn 进阶**：自定义层与复杂网络结构搭建
- [ ] **05. Pytorch 之优化**：SGD、Adam 等常见优化器与学习率调整
- [ ] **06. PyTorch 之线性回归**：端到端训练流程（前向、算损失、反向、更新）
- [ ] **07. Pytorch 之神经网络**：多层感知机（MLP）构建与实战

### 第二阶段：深度学习底层原理与反向传播手写实现
*对应笔记：[02-dl-math-and-backprop.md](./notes/02-dl-math-and-backprop.md)*
- [ ] **08. 常用损失函数的实现**：MSE、交叉熵损失（Cross-Entropy）公式与实现
- [ ] **09. 全连接层和激活函数的反向传播实现**：矩阵求导、ReLU/Sigmoid 梯度传导
- [ ] **10. 卷积层和池化层的反向传播实现**：特征图梯度计算与步长/填充处理
- [ ] **11. 实现神经网络模型的前向传播、反向传播和训练**：手写完整训练闭环

### 第三阶段：图像识别经典网络构建
*对应笔记：[03-cnn-and-alexnet.md](./notes/03-cnn-and-alexnet.md)*
- [ ] **12. 模型构建 --- CNN**：卷积神经网络特征提取与分类器搭建
- [ ] **13. 模型构建 --- AlexNet**：经典深度卷积网络结构解析与 PyTorch 实现
