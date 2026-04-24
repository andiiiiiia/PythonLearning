用来衡量模型参数质量的函数，比较神经网络输出值和真实值之间的差异  

也称为目标函数，代价函数，误差函数等  

# 1. 分类模型的损失函数
## 1.1 多分类交叉熵损失
### 1.1.1 公式
**对一个样本计算损失如下：**  

$Loss = -\sum_{i=1}^{C}y_{i}ln(\hat{y_{i}})$  


$y_{i}$  :  是真实标签的第i个元素(one-hot编码)  

$C$  :  是类别总数  

$\hat{y_{i}}$  :  模型预测的第i个类别的概率，经过softmax激活函数得到的。

**所有样本的损失计算如下：**  

$Loss = -\frac{1}{N}\sum_{i=1}^{N}ln(\hat{y_{k_{n}}})$    

$N$  :  是样本总数  

$k_{n}$  :  第n个样本在真实类别索引。  


$\hat{y_{k_{n}}}$  :  第n个样本在真实类别上的预测概率，经过softmax激活函数得到的。  

ps:为什么省略  $y_i$  ，因为真实类别时，为1。  

**注意**  
在 torch.nn.CrossEntropyLoss中，会对输入的真实值做一次softmax。  

### 1.1.2 举例说明：

假设一个样本，标签真实值为（0,1,0），模型输出层的三个神经元加权求和后输出值为(0.1,2,1)，经过**激活函数**softmax后（0.1,0.7,0.2）  

则该样本的多分类交叉熵损失为：  $-0 * ln(0.1)-1 * ln(0.7)-0 * ln(0.2) = -ln(0.7)$  

### 1.1.3 代码 nn.CrossEntropyLoss()

```python
def multi_class_cross_loss():
    print("=" * 40)
    print("\n多分类交叉熵损失:\n")
    # 一共有两个样本
    # 假设真实值如下(one-hot)
    y_true = torch.tensor([[0, 1, 0], [1, 0, 0]], dtype=torch.float32)
    # y_true = torch.tensor([1,0], dtype=torch.int64)  # 等价于这个写法，0代表第一个分类，1代表第2个分类，
    # 假设模型输出值如下：
    y_pred = torch.tensor([[0.1, 0.8, 0.1], [0.7, 0.2, 0.1]], dtype=torch.float32, requires_grad=True)

    # 实例化交叉熵损失对象
    loss = nn.CrossEntropyLoss()
    # 计算损失
    my_loss = loss(y_pred, y_true).detach().numpy()
    print("\n损失值:\n", my_loss)
    pass
```

**执行结果：**  

```text
========================================

多分类交叉熵损失:


损失值:
 0.72883815
```

## 1.2 二分类交叉熵损失
### 1.2.1 公式

**对于一个样本：**

$Loss = -yln(\hat{y}) - (1-y)ln(1-\hat{y})$  

y:真实标签值(0或1)

$\hat{y}$  : 模型预测概率(经过sigmoid激活函数后的输出值)

**对于多个样本：**  

$Loss = -\frac{1}{N}\sum_{i=1}^{N}[y_{i}ln(\hat{y_{i}}) + (1-y_{i})ln(1-\hat{y_{i}})]$  

**注意：**   
为什么用ln而不是log？  在pytorch中，默认使用自然对数ln，主要是求导方便，便于计算。

### 1.2.2 举例说明

略

### 1.2.3 代码 nn.BCELoss()

```python
def binary_class_cross_loss():
    print("=" * 40)
    print("\n二分类交叉熵损失:\n")
    # 假设三个样本
    y_true = torch.tensor([0, 1, 0], dtype=torch.float32)
    # 假设预测值是经过sigmoid激活之后得到的值
    y_pred = torch.tensor([0.6901, 0.5459, 0.2469], requires_grad=True)

    # 实例化交叉熵损失函数
    loss = nn.BCELoss()
    # 计算损失
    my_loss = loss(y_pred, y_true).detach().numpy()
    print("\n损失值:\n", my_loss)
```

**执行结果：**  

```text
========================================

二分类交叉熵损失:


损失值:
 0.6867941
```

# 2. 回归模型的损失函数
## 2.1 MAE损失函数（L1 Loss）
### 2.1.1 公式

**多个样本：**  

$Loss = \frac{1}{N}\sum_{i=1}^{N}|y_{i}-\hat{y_{i}}|$  

### 2.1.2 说明

L1 Loss 具有稀疏性，为了惩罚较大的值，通常将其加入到其他Loss损失函数中，作为约束。  

L1 Loss 梯度在零点处不平滑，会导致跳过极小值  

### 2.1.3 代码
```python
def mae_loss():
    print("=" * 40)
    print("\nmae损失函数(L1 Loss):\n")
    y_true = torch.tensor([2., 2., 2.], dtype=torch.float32)
    y_pred = torch.tensor([1.2, 1.3, 1.9], requires_grad=True)
    loss = nn.L1Loss()
    my_loss = loss(y_pred, y_true).detach().numpy()
    print("\n损失值:\n", my_loss)

```

**运行结果**  

```text
========================================

mae损失函数(L1 Loss):


损失值:
 0.0
```

## 2.2 MSE损失函数（欧式距离）
### 2.2.1 公式
**多个样本：**  

$Loss = \frac{1}{N}\sum_{i=1}^{N}(y_{i}-\hat{y_{i}})^{2}$  

### 2.2.2 说明

通常作为正则项  

当预测值和目标值相差很大，容易导致梯度爆炸  

### 2.2.3 代码
```python
def mse_loss():
    print("=" * 40)
    print("\nmse损失函数:\n")
    y_true = torch.tensor([2., 2., 2.], dtype=torch.float32)
    y_pred = torch.tensor([1.2, 1.3, 1.9], requires_grad=True)
    loss = nn.MSELoss()
    my_loss = loss(y_pred, y_true).detach().numpy()
    print("\n损失值:\n", my_loss)
```

**执行结果**  

```text
========================================

mse损失函数:


损失值:
 0.38
```

## 2.3 Smooth L1损失函数
### 2.3.1 公式

$$
Loss = \begin{cases}
0.5x^{2}, & |x|<1\\
|x|-0.5, & otherwise
\end{cases}
$$   

$x = y_{i} - \hat{y_{i}}$  真实值-预测值  

### 2.3.2 说明

综合mae和mse，采用分段函数，解决零点不平滑和梯度爆炸问题  

图例略  

### 2.3.3 代码
```python
# smoothL1损失函数
def smoothL1_loss():
    print("=" * 40)
    print("\nsmoothL1损失函数:\n")
    y_true = torch.tensor([0,3], dtype=torch.float32)
    y_pred = torch.tensor([0.6,0.4], requires_grad=True)
    loss = nn.SmoothL1Loss()
    my_loss = loss(y_pred, y_true).detach().numpy()
    print("\n损失值:\n", my_loss)
```

**运行结果**  

```text
========================================

smoothL1损失函数:


损失值:
 47.84
```

