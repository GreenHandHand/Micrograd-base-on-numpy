# Micrograd - 基于 Numpy 的简易深度学习框架

Micrograd 是我为了学习深度学习而实现的一个简单框架，主要使用 Numpy 实现了自动微分、神经网络层、优化器等功能。这个框架的设计目标是帮助我更好地理解深度学习的基础原理，虽然功能不复杂，但涵盖了许多深度学习框架的核心概念。

> [!note]
> 该项目是在学习了 [karpathy/micrograd](https://github.com/karpathy/micrograd) 之后，我对其进行了一些修改和扩展，使其更加完善。(但是这个框架仍然非常简单！)
> 如果你有兴趣，你可以看看[The spelled-out intro to neural networks and backpropagation: building micrograd - Andrej Karpathy](https://youtu.be/VMj-3S1tku0?si=iRJsTtFb_LZUBruY)

## Features

- Auto Grad：框架通过 `Variable` 类实现了自动微分功能，可以追踪张量的计算过程并计算梯度，支持反向传播。
- Torch-Like Module：实现了一个 `Module` 基类，用户可以基于它创建不同的神经网络层。当前实现了一个简单的 `Linear` 层。
- Optimizer：实现了 `SGD`（随机梯度下降）优化器，用于更新神经网络的参数。
- Loss Function：框架提供了 CrossEntropyLoss 损失函数，适用于分类任务。
- Torch-Like Variable Operator：支持常见的张量操作（如加法、乘法、矩阵乘法、指数运算等），并且这些操作会自动计算梯度。
- Pytest：框架包含了完整的测试用例，使用 pytest 进行单元测试，确保代码的正确性。
- 计算图绘制：使用 graphviz 库支持计算图的可视化，帮助更好地理解反向传播的过程。

## Installation

首先，克隆仓库并进入项目目录：
```bash
git clone https://github.com/GreenHandHand/Micrograd-base-on-numpy.git
cd Micrograd-base-on-numpy
```
然后安装项目依赖：
```bash
pip install -r requirements.txt
```

## Usage

你可以使用这个框架来构建简单的神经网络模型，进行训练和优化(效率可能会很低)。`main.py` 是一个简单的使用方法，使用 `main.py` 需要几个常用的机器学习库：
- `scikit-learn`
- `tqdm`

## Structure

- `micrograd/`：包含框架的核心实现，包括 Variable 类、Module 基类、Linear 层和优化器等。
- `micrograd/tests/`：包含测试用例，使用 pytest 进行单元测试。

## Test

框架内含有完整的测试用例，你可以使用 pytest 来运行这些测试，确保各个功能模块正常工作：
```bash
pytest --pyargs micrograd
```

## 计算图绘制

框架支持通过 `graphviz` 库绘制计算图，帮助可视化反向传播过程中的计算图。你可以像这样生成并保存计算图：
```python
# 假设你要绘制的变量是 y_pred
y_pred.draw(save_dir="output", save_name="computation_graph")
```
该功能需要安装 graphviz，可以通过以下命令进行安装：
```bash
pip install graphviz pygraphviz
# 如果 pygraphviz 安装失败你可以试试
conda install pygraphviz
```

## 免责声明

这个项目仅用于学习目的，旨在帮助我理解深度学习框架的基本原理。它并不适用于生产环境，功能较为简化，适合用作学习工具。

## license

MIT
