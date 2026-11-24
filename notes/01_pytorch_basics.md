###Tensor的概念：
**Tensor**（张量）是人工智能、机器学习中非常基础的一个概念，是一种能够存储数字的数据结构，可以理解为单个数字到多维数组的统一形式。

###Tensor维度鱼数学概念的对应：

在 PyTorch 中，不同维度的张量对应不同的数学对象，可以通过 `ndim`（维度数）和 `shape`（各维度大小）来查看：

```python
import torch

# 0维：标量 (Scalar)
scalar = torch.tensor(5)
print("标量:", scalar, "| ndim:", scalar.ndim, "| shape:", scalar.shape)

# 1维：向量 (Vector)
vector = torch.tensor([1, 2, 3])
print("向量:", vector, "| ndim:", vector.ndim, "| shape:", vector.shape)

# 2维：矩阵 (Matrix)
matrix = torch.tensor([[1, 2, 3], [4, 5, 6]])
print("矩阵:", matrix, "| ndim:", matrix.ndim, "| shape:", matrix.shape)

# 3维：高维张量（如单张 C×H×W 图像）
img = torch.zeros((3, 32, 32))
print("3维张量 | ndim:", img.ndim, "| shape:", img.shape)
```

> **小提示**：一维 Tensor 本身没有“行向量”和“列向量”之分，形状为 `torch.Size([n])`；需要显式区分行列时，通常会用二维形状 `(1, n)` 或 `(n, 1)`。
