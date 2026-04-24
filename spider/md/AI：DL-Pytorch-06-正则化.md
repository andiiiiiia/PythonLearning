# 1. dropout 随机失活 torch.nn.Dropout(p)
## 1.1 原理

为了防止模型**过拟合**（overfitting）的，它通过在训练过程中随机“关闭”一部分神经元，从而强制网络学习更加鲁棒和泛化性更强的特征。  

在每次**训练**迭代中，Dropout 会以一定的概率（通常为 0.2 到 0.5）随机将神经元的输出设为 0，相当于临时移除这些神经元。这样做的好处是：  

- 防止神经元之间过度依赖

- 降低模型复杂度

- 提高模型的泛化能力

未被dropout的输出会除以(1-p)进行数值缩放  

在测试或推理阶段，Dropout 通常会被关闭。  

实际应用中，**一般在激活函数之后，增加dropout层**，来对激活值以p的概率进行随机置0失活   

## 1.2 代码说明
```python
def demo():
    # 创建输入层
    inputs = torch.randint(0, 10, size=(1, 4)).float()
    print("输入层 input:\n", inputs)

    # 创建隐藏层1
    hidden1 = torch.nn.Linear(4, 5)
    # 加权求和
    logits1 = hidden1(inputs)
    print("隐藏层1 加权求和 输出结果:\n", logits1)
    # 激活函数
    activation1_out = torch.relu(logits1)
    print("隐藏层1 activate 输出结果:\n", activation1_out)

    # 创建drop，只有训练阶段有
    dropout1 = torch.nn.Dropout(p=0.4)
    dropout1_out = dropout1(activation1_out)  # 随机失活后的数据
    print("隐藏层1 dropout 输出结果:\n", dropout1_out)
```

**执行结果：**  
```text
输入层 input:
 tensor([[7., 5., 9., 4.]])
隐藏层1 加权求和 输出结果:
 tensor([[ 1.3604,  5.5111,  6.0967, -4.7166,  2.3364]],
       grad_fn=<AddmmBackward0>)
隐藏层1 activate 输出结果:
 tensor([[1.3604, 5.5111, 6.0967, 0.0000, 2.3364]], grad_fn=<ReluBackward0>)
隐藏层1 dropout 输出结果:
 tensor([[0.0000, 9.1851, 0.0000, 0.0000, 3.8940]], grad_fn=<MulBackward0>)
```

# 2. BN 批量归一化
## 2.1 原理和公式

在神经网络中，每一层的输入分布会随着前一层参数的更新而变化，这种现象称为**内部协变量偏移**。  

用于加速训练过程并提高模型的稳定性。它通过在每个小批(mini-batch)上对输入进行标准化（均值为0，方差为1），从而减少内部协变量偏移（Internal Covariate Shift），归一化后，区间范围小一些，更稳定一些。   

假设上一层Level0的输出为：x  

本层L1计算过程如下：  

**加权求和**：  

$z = w*x+b$    

**BN批量归一化**：  

标准化：  

$\hat{z_{i}} = \frac{z_{i}-\mu_{B}}{\sqrt{\sigma_{B}^{2} + \epsilon}}$    

再引入学习参数  $\gamma$  和  $\beta$  来恢复网络表达能力  

$y_{i} = \gamma * \hat{z_{i}} + \beta$  

**激活函数**  

上述BN操作之后，数据基本在0-1区间，激活函数有更好的效果，收敛会更快。  

level1_out = activate(y)  

**注意**  

$\mu_{B}$  : 当前mini-batch的均值  

$\sigma_{B}^{2}$  ： 当前minibatch的方差    

$\epsilon$  : 防止分母为0的小常数  

## 2.2 代码说明
```python
def demo():
    # 本层加权求和后结果,含义：两通道，三行四列
    logist1 = torch.randn(size=(1, 2, 3, 4)) * 50
    print("logist1=\n", logist1)
    # 做bn归一化,
    # num_features: 输入特征图的通道数
    # momentum : 控制在训练过程中对均值和方差的移动平均更新速度。momentum 越小，更新越慢，保留的历史信息越多。
    # affine：是否开启缩放和平移。
    #   若为True，标准化之后，还会进行缩放和平移，学习两个参数：gamma和beta。
    #   若为False，只做标准化，标准化之后，不会再缩放和平移，相当于强制输出均值为 0，方差为 1
    bn2d = torch.nn.BatchNorm2d(num_features=2, eps=1e-5, momentum=0.1, affine=True)
    bn2d_out = bn2d(logist1)
    print("bn2d_out=\n", bn2d_out)
```

**执行结果**  

```text
logist1=
 tensor([[[[-64.4567,  31.1585, -76.6693,   7.0935],
          [-29.0604, -58.9067,  -7.0265,  37.5548],
          [ 78.8859, -46.1850,  -0.7877, -52.9365]],

         [[-23.0126, -53.1067,   1.3608, -15.7884],
          [  2.9588,  41.9785, -27.5607,  28.6389],
          [-30.7444, -37.7078,  38.8973, -53.5600]]]])
bn2d_out=
 tensor([[[[-1.0758,  1.0088, -1.3421,  0.4841],
          [-0.3041, -0.9548,  0.1763,  1.1482],
          [ 2.0493, -0.6775,  0.3123, -0.8246]],

         [[-0.3859, -1.3242,  0.3741, -0.1606],
          [ 0.4239,  1.6406, -0.5277,  1.2247],
          [-0.6270, -0.8441,  1.5445, -1.3384]]]],
       grad_fn=<NativeBatchNormBackward0>)
```






