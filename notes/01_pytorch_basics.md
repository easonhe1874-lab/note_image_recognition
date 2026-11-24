## 1.Tensor的概念
**Tensor**（张量）是人工智能、机器学习中非常基础的一个概念，是一种能够存储数字的数据结构，可以理解为单个数字到多维数组的统一形式。

* **Tensor**维度的数学概念的对应：

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

## NumPy 与 Tensor 互转：`torch.from_numpy()`

在 PyTorch 中，`Tensor` 是一种包含单一数据类型元素的多维矩阵（译作张量），与 NumPy 的 `ndarray` 很相似（一维叫 Vector，二维叫 Matrix，三维及以上称为 Tensor），两者可以相互转换[span_1](start_span)[span_1](end_span)。

* **作用**：将 `numpy.ndarray` 转换为 PyTorch 的 `Tensor`[span_2](start_span)[span_2](end_span)。
* **基本语法**：`torch.from_numpy(ndarray)`[span_3](start_span)[span_3](end_span)

```python
import numpy as np
import torch

# 1. 声明一个 numpy array
np_data = np.array([1, 2, 3])

# 2. 利用 from_numpy 将其转为 tensor
torch_data = torch.from_numpy(np_data)

print("numpy array:", np_data)
print("torch tensor:", torch_data)
```

> **踩坑注意**：通过 `torch.from_numpy()` 转换得到的 Tensor 和原 `ndarray` 在 CPU 上通常**共享内存**——修改其中一个的值，另一个也可能跟着变（如果不想共享，可以用 `.clone()` 拷贝一份）。


# 3.Tensor 创建的多种方法

PyTorch 提供了多种创建张量（Tensor）的方式。下面介绍通过形状、Python 序列和常用函数创建张量的方法。

## 1. 根据形状创建张量

`torch.Tensor` 可以根据给定的维度创建张量。通过这种方式创建的张量是**未初始化的**，其中的值来自内存中的现有数据，不能假设为随机值或固定值。`torch.Tensor` 默认对应 `torch.FloatTensor`。

```python
import torch

t1 = torch.Tensor(2, 2)
print(t1)
print(t1.dtype)
print(t1.shape)
```

也可以使用 `torch.empty()` 创建未初始化的张量：

```python
t2 = torch.empty(2, 2)
```

## 2. 根据 Python 序列创建张量

可以将 Python 列表等序列转换为张量。推荐使用 `torch.tensor()`，它会根据输入数据推断数据类型；也可以显式指定类型。

```python
import torch

t1 = torch.tensor([[1, 2, 3], [4, 5, 6]])
print(t1)
print(t1.dtype)
print(t1.shape)
```

指定浮点类型：

```python
t2 = torch.tensor(
    [[1, 2, 3], [4, 5, 6]],
    dtype=torch.float32
)
```

`t1` 的形状为 `2 × 3`，内容如下：

```text
tensor([[1, 2, 3],
        [4, 5, 6]])
```

## 3. 创建常用的特殊张量

| 方法 | 作用 |
| --- | --- |
| `torch.ones(n, m)` | 创建形状为 `n × m`、元素全为 1 的张量 |
| `torch.zeros(n, m)` | 创建形状为 `n × m`、元素全为 0 的张量 |
| `torch.eye(n, m)` | 创建形状为 `n × m` 的张量，主对角线为 1，其余元素为 0 |
| `torch.linspace(start, end, steps=100)` | 在 `start` 和 `end` 之间生成指定数量的等间距点，默认生成 100 个 |
| `torch.rand(*sizes)` | 从 `[0, 1)` 均匀分布中采样，形状由参数指定 |
| `torch.randn(*sizes)` | 从标准正态分布（均值为 0、方差为 1）中采样，形状由参数指定 |

示例：

```python
import torch

print(torch.ones(2, 3))
print(torch.zeros(2, 3))
print(torch.eye(3))
print(torch.linspace(0, 1, steps=5))
print(torch.rand(2, 2))
print(torch.randn(2, 2))
```

## 4. 查看张量的形状

可以通过 `.size()` 或 `.shape` 查看张量的形状：

```python
a = torch.randn(2, 2)

print(a)
print(a.size())  # torch.Size([2, 2])
print(a.shape)   # torch.Size([2, 2])
```

随机张量每次运行时的具体数值可能不同。
```
