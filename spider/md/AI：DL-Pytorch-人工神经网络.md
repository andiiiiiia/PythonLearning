# 1. 人工神经网络（ANN Artificial Neural Network）介绍
ANN 简称 NN  
## 1.1 **针对一个样本的一个神经元如下示例：**  
图缺失

激活函数：  $f()$  

## 1.2 **针对一个样本的神经网络**  
图缺失


### 1.2.1 **特点：**  
同一层的神经元之间没有连接  
第 N 层的每个神经元和第 N-1 层的所有神经元相连（这就是full connected的含义），这就是全连接神经网络  
全连接神经网络接收的样本数据是二维的，数据在每一层之间需要以二维的形式传递  
第N-1层神经元的输出就是第N层神经元的输入  
每个连接都有一个权重值（w系数和b系数）  

### 1.2.2 **反向传播**  
反向传播是通过**链式法则**（Chain Rule）来计算损失函数对每一层参数的梯度。它的核心思想是：  
从输出层开始，逐层向前传播误差（梯度），每一层的梯度由下一层的梯度和当前层的输出共同决定。  
 

## 1.3 神经网络中神经元的内部状态和激活值
图缺失


一个神经元工作时，前向传播会产生两个值，内部状态值（加权求和值）和激活值；  
反向传播时会产生激活值梯度和内部状态值梯度。  

**内部状态值：**  
神经元或隐藏单元的内部存储值，它反映了当前神经元接收到的输入、历史信息以及网络内部的权重计算结果。  
z = W * x + b   
W：权重矩阵  
x：输入值  
b：偏置  

**激活值：**  
通过激活函数（如 ReLU、Sigmoid、Tanh）  
a = f(z)  
f：激活函数  
z：内部状态值  

## 1.4 举例：  
### 1.4.1 数据与模型准备
PS：为了便于计算和理解，不考虑偏置量b  

输入层为两个样本，每个样本有5个特征，均为(1*5)：  
$
X_1 = \begin{bmatrix}
{x_{11}}&{x_{12}}&{x_{13}}&{x_{14}}&{x_{15}}\\
\end{bmatrix}
$  
  
$
X_2 = \begin{bmatrix}
{x_{21}}&{x_{22}}&{x_{23}}&{x_{24}}&{x_{25}}\\
\end{bmatrix}
$  

真实标签值为：  
$
Y_1 = \begin{bmatrix}
{y_{11}}&{y_{12}}\\
\end{bmatrix}
$  

$
Y_2 = \begin{bmatrix}
{y_{21}}&{y_{22}}\\
\end{bmatrix}
$  

隐藏层(1)有4个神经元，神经元分别为:  $Neuron_{11},Neuron_{12},Neuron_{13},Neuron_{14}$  
输出层(2)为2个神经元，神经元分别为:  $Neuron_{21},Neuron_{22}$  
  
以第一个样本为例，计算过程如下：  
### 1.4.2 隐藏层计算过程
**Neuron_11计算过程:**  
权重矩阵(5*1)为:  
$
W_{11}=\begin{bmatrix}
{w_{111}}\\
{w_{112}}\\
{w_{113}}\\
{w_{114}}\\
{w_{115}}\\
\end{bmatrix}
$  

PS：第一个下标为层数，第二个下标为第几个神经元，第三个下标为第几个权重值。  

$Neuron_{11}$  的输入值为：  
$in_{11} = x_{11}*w_{111} + x_{12}*w_{112}+x_{13}*w_{113}+x_{14}*w_{114}+x_{15}*w_{115}$  

经激活函数之后(假设sigmoid)值(输出值)为：  
$out_{11} = sigmoid(in_{11})$  

**Neuron_12计算过程:**  
权重矩阵(5*1)为:  
$
W_{12} = \begin{bmatrix}
{w_{121}}\\
{w_{122}}\\
{w_{123}}\\
{w_{124}}\\
{w_{125}}\\
\end{bmatrix}
$  
ps：第一个下标为层数，第二个下标为第几个神经元，第三个下标为第几个权重值。  

$Neuron_{12}$  的输入值为：  
$in_{12} = x_{11}*w_{121} + x_{12}*w_{122}+x_{13}*w_{123}+x_{14}*w_{124}+x_{15}*w_{125}$  

经激活函数之后(假设sigmoid)值(输出值)为：  
$out_{12} = sigmoid(in_{12})$  

**Neuron_13计算过程:**  
权重矩阵(5*1)为:  
$
W_{13} = \begin{bmatrix}
{w_{131}}\\
{w_{132}}\\
{w_{133}}\\
{w_{134}}\\
{w_{135}}\\
\end{bmatrix}
$  
ps：第一个下标为层数，第二个下标为第几个神经元，第三个下标为第几个权重值。  

$Neuron_{13}$  的输入值为：  
$in_{13} = x_{11}*w_{131} + x_{12}*w_{132}+x_{13}*w_{133}+x_{14}*w_{134}+x_{15}*w_{135}$  

经激活函数之后(假设sigmoid)值(输出值)为：  
$out_{13} = sigmoid(in_{13})$  

**Neuron_14计算过程:**  
权重矩阵(5*1)为:  
$
W_{14} = \begin{bmatrix}
{w_{141}}\\
{w_{142}}\\
{w_{143}}\\
{w_{144}}\\
{w_{145}}\\
\end{bmatrix}
$  
ps：第一个下标为层数，第二个下标为第几个神经元，第三个下标为第几个权重值。  

$Neuron_{14}$  的输入值为：  
$in_{14} = x_{11}*w_{141} + x_{12}*w_{142}+x_{13}*w_{143}+x_{14}*w_{144}+x_{15}*w_{145}$  

经激活函数之后(假设sigmoid)值(输出值)为：  
$out_{14} = sigmoid(in_{14})$  

**隐藏层1总结：**  
**权重矩阵(5*4)为：**  
$
W_{1}=\begin{bmatrix}
{w_{111}}&{w_{121}}&{w_{131}}&{w_{141}}\\
{w_{112}}&{w_{122}}&{w_{132}}&{w_{142}}\\
{w_{113}}&{w_{123}}&{w_{133}}&{w_{143}}\\
{w_{114}}&{w_{124}}&{w_{134}}&{w_{144}}\\
{w_{115}}&{w_{125}}&{w_{135}}&{w_{145}}\\
\end{bmatrix}
$  
**输出矩阵(1*4)为：**  
$
out_{1} = \begin{bmatrix}
{out_{11}}&{out_{12}}&{out_{13}}&{out_{14}}\\
\end{bmatrix}
$  

PS: 这里为了方便理解，不进行转置，权重矩阵一列代表一个神经元，采用5 * 4来表示。  实际在pytorch中时，一行代表一个神经元的权重矩阵 ，总体权重矩阵应该是(4 * 5)的，  $y = x * w^T$    下面输出层同理，不赘述。

### 1.4.3 输出层计算过程
**Neuron_21计算过程:**  
权重矩阵(4*1)为:  
$
W_{21}=\begin{bmatrix}
{w_{211}}\\
{w_{212}}\\
{w_{213}}\\
{w_{214}}\\
\end{bmatrix}
$  
ps：第一个下标为层数，第二个下标为第几个神经元，第三个下标为第几个权重值。  

$Neuron_{21}$  的输入值为：  
$in_{21} = out_{11}*w_{211} + out_{12}*w_{212} + out_{13}*w_{213} + out_{14}*w_{214}$  

经激活函数之后(假设sigmoid)值(输出值)为：  
$out_{21} = sigmoid(in_{21})$  

**Neuron_22计算过程:**  
权重矩阵(4*1)为:  
$
W_{22}=\begin{bmatrix}
{w_{221}}\\
{w_{222}}\\
{w_{223}}\\
{w_{224}}\\
\end{bmatrix}
$  
ps：第一个下标为层数，第二个下标为第几个神经元，第三个下标为第几个权重值。  

$Neuron_{22}$  的输入值为：  
$in_{22} = out_{11}*w_{221} + out_{12}*w_{222} + out_{13}*w_{223} + out_{14}*w_{224}$  

经激活函数之后(假设sigmoid)值(输出值)为：  
$out_{22} = sigmoid(in_{22})$  

**输出层2总结：**  
**权重矩阵(4*2)为：**  
$
W_{2}=\begin{bmatrix}
{w_{211}}&{w_{221}}\\
{w_{212}}&{w_{222}}\\
{w_{213}}&{w_{223}}\\
{w_{214}}&{w_{224}}\\
\end{bmatrix}
$  
**输出矩阵(1*2)为：**  
$
out_{2} = \begin{bmatrix}
{out_{21}}&{out_{22}}\\
\end{bmatrix}
$  

### 1.4.4 损失函数
假设损失函数采用均方误差MSE:  
$Loss = \frac{1}{2}[(y_{11}-out_{21})^2 + (y_{12}-out_{22})^2]$  


### 1.4.5 计算输出层梯度并更新
**计算输出层权重梯度：**  
以输出层第一个神经元的第一个权重值  $w_{211}$  为例：  
$\frac{\partial Loss }{\partial w_{211}} = \frac{\partial Loss }{\partial out_{21}} * \frac{\partial out_{21} }{\partial w_{211}} = (out_{21} - y_{11})*out_{11}*out_{21}*(1-out_{21})$   
**同理可得，**  
$\frac{\partial Loss }{\partial w_{212}} , \frac{\partial Loss }{\partial w_{213}} , \frac{\partial Loss }{\partial w_{214}} , \frac{\partial Loss }{\partial w_{221}} , \frac{\partial Loss }{\partial w_{222}} , \frac{\partial Loss }{\partial w_{223}} , \frac{\partial Loss }{\partial w_{224}}$  

**更新输出层权重：**  
$w_{2ij} = w_{2ij} - \eta*\frac{\partial Loss }{\partial w_{2ij}}$  

### 1.4.6 计算隐藏层梯度并更新
以隐藏层1的第一个神经元的第一个权重值  $w_{111}$  为例:  
$\frac{\partial Loss }{\partial w_{111}} = (\frac{\partial Loss }{\partial out_{21}}*\frac{\partial out_{21} }{\partial out_{11}} + \frac{\partial Loss }{\partial out_{22}}*\frac{\partial out_{22} }{\partial out_{11}})*\frac{\partial out_{11} }{\partial w_{111}} $  

$\frac{\partial Loss }{\partial out_{21}} = out_{21} - y_{11}$   
  
$\frac{\partial Loss }{\partial out_{22}} = out_{22} - y_{12}$
  
$\frac{\partial out_{21} }{\partial out_{11}} = w_{211}*out_{21}*(1-out_{21})$  
  
$\frac{\partial out_{22} }{\partial out_{11}} = w_{221}*out_{22}*(1-out_{22})$  
  
注意：这里的$w_{211} 和 w_{221} 都是输出层更新后的权重$  

同理可得，其他权重下降梯度  

**更新隐藏层权重：**   
同上  

# 2. 常见激活函数
## 2.1 sigmoid
### 2.1.1 **函数与导数：**  
**函数：**  
$f(x) = \frac{1}{1+e^{-x}}$  
f(x)取值范围：(0,1)    
在负无穷到正无穷上，单调递增。  
f(0) = 0.5  
  
**导数：**  
导数梯度：  $\dot{f(x)} = f(x)*(1-f(x))$  
当f(x) = 0.5时，导数取得最大值0.25 

### 2.1.2 绘制代码
```python
import os

import matplotlib.pyplot as plt
import torch

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# 1.准备数据
x = torch.linspace(-20, 20, 1000)
y = torch.sigmoid(x)
_, axes = plt.subplots(1, 2)

# 2.在第一个图上绘制sigmoid激活函数图像
axes[0].plot(x, y)
axes[0].grid()
axes[0].set_title('sigmoid')

# 3.在第二个图上绘制sigmoid激活函数的导数图像
x = torch.linspace(-20, 20, 1000, requires_grad=True)
torch.sigmoid(x).sum().backward()
axes[1].plot(x.detach(), x.grad)
axes[1].grid()
axes[1].set_title('sigmoid grad')

# 4.绘制
plt.show()

```
  
**绘制结果：**  
图缺失


### 2.1.3 问题：
W新 = W旧 - 学习率*导数梯度
因为梯度最大为0.25，如果隐藏层全部使用sigmod当激活函数，最多5层，就会很容易出现梯度消失的问题。

### 2.1.4 **应用场景：**  
一般应用于二分类的最后输出层   

## 2.2 tanh

### 2.2.1 函数与导数
**函数：**  
$f(x) = \frac{1-e^{-2x}}{1+e^{-2x}}$  
取值范围(-1,1)  
f(0) = 0  
在负无穷到正无穷上单调递增  
在 -3<x<3 范围上，函数值变化最明显。 

**导数：**  
$\dot{f(x)} = 1-{f}^2(x)$  

导数取值范围(0,1]  
f(x) = 0时，导数最大，为1  

### 2.2.2 绘制代码
```python
import os

import matplotlib.pyplot as plt
import torch

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# 1.准备数据
x1 = torch.linspace(-20, 20, 1000)
y1 = torch.tanh(x1)
_, axes = plt.subplots(1, 2)

# 2.在第一个图上绘制tanh激活函数图像
axes[0].plot(x1, y1)
axes[0].grid()
axes[0].set_title('tanh')

# 3.在第二个图上绘制tanh激活函数的导数图像
x2 = torch.linspace(-20, 20, 1000, requires_grad=True)
torch.tanh(x2).sum().backward()
axes[1].plot(x2.detach(), x2.grad)
axes[1].grid()
axes[1].set_title('tanh grad')

# 4.绘制
plt.show()

```

**绘制结果：**  

图缺失


### 2.2.3 问题
梯度值比sigmod大，收敛速度比sigmod快。  
在x很小和x很大时，导数趋近于0，也会存在梯度消失的情况。  

### 2.2.4 适用场景
主要应用于浅层神经网络的隐藏层  

## 2.3 Relu
### 2.3.1 函数与导数
**函数：**  
f(x) = max(0,x)  

图缺失


**导数:**  
$\dot{f(x)} = {0,1}$   
因为f(x)>=0,所以导数一直=1  

图缺失


### 2.3.2 绘制代码
```python
import os

import matplotlib.pyplot as plt
import torch

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# 1.准备数据
x = torch.linspace(-20, 20, 1000)
y = torch.relu(x)
_, axes = plt.subplots(1, 2)

# 2.在第一个图上绘制relu激活函数图像
axes[0].plot(x, y)
axes[0].grid()
axes[0].set_title('relu')

# 3.在第二个图上绘制relu激活函数的导数图像
x = torch.linspace(-20, 20, 1000, requires_grad=True)
torch.relu(x).sum().backward()
axes[1].plot(x.detach(), x.grad)
axes[1].grid()
axes[1].set_title('relu grad')

# 4.绘制
plt.show()

```

**绘制结果：**  
图缺失


### 2.3.3 问题
ReLU 激活函数将小于 0 的值映射为 0，而大于 0 的值则保持不变，它更加重视正信号，而忽略负信号，这种激活函数运算更为简单，能够提高模型的训练效率。  

当x<0时，ReLU导数为0，而当x>0时，则不存在饱和问题。  

所以，ReLU能够在x>0时保持梯度不衰减，从而缓解梯度消失问题。  

然而，随着训练的推进，导致更新后的权重值落入小于0区域，导致对应权重无法更新。这种现象被称为“**神经元死亡**”。 

### 2.3.4 适用场景
ReLU是目前最常用的激活函数。与sigmoid相比，RELU的优势是：  

采用sigmoid函数，计算量大（指数运算），反向传播求误差梯度时，计算量相对大，而采用Relu激活函数，整个过程的计算量节省很多。   

sigmoid函数反向传播时，很容易就会出现梯度消失的情况，从而无法完成深层网络的训练。Relu会使一部分神经元的输出为0，这样就造成了网络的稀疏性，并且减少了参数的相互依存关系，缓解了过拟合问题的发生。  
## 2.4 softmax
### 函数与导数
**函数：**   
对于输入向量  $z = [z_{1},z_{2}...z_{n}]$  
$softmax(z_{i}) = \frac{e^{z_{i}}}{\sum_{j=1}^{n}e^{z_{j}}}$    
### 绘制代码
### 问题
### 适用场景
多分类  

# 3. 常见参数初始化方法
**为什么要做初始化？**  
避免梯度爆炸或消失  
打破梯度对称性，避免学不到特征  
加速模型的收敛速度，


## 3.1 均匀分布初始化：nn.init.uniform_()
默认区间：(0,1)。  
某个神经元的输入数量为d（上一层数量），可以设置为  $(\frac{-1}{\sqrt{d}},\frac{1}{\sqrt{d}})$  

**优点：**   
能有效打破对称性  

**缺点：**  
随机选择范围不当可能导致梯度问题  

**适用场景：**  
浅层网络或低复杂度模型。隐藏层1-3层，总层数不超过5层。  

## 3.2 正态分布初始化：nn.init.normal_()
默认：均值为0，标准差为1的高斯分布，使用一些很小的值对参数进行初始化  

## 3.3 全0初始化：nn.init.zeros_()
**优点：**   
简单  

**缺点：**  
可能有对称性问题，导致所有神经元更新方向相同，无法有效训练  

**适用场景：**  
几乎无，偏置初始化可能会用  

## 3.4 全1初始化：nn.init.ones_()
**优点：**   
简单  

**缺点：**  
可能有对称性问题，导致所有神经元更新方向相同，无法有效训练  
激活值在神经网络中呈指数增长，出现梯度爆炸  

**适用场景：**  
测试或调试：比如验证神经网络是否能正常前向传播和反向传播  
特殊模型结构：某些稀疏网络或特定的自定义网络中可能需要手动设置部分参数为1  
偏置初始化：偶尔可以将偏置初始化为小的正值（如0.1），但很少用1作为偏置的初始值  

## 3.5 固定值初始化：nn.init.constant_()
所有权重参数初始化为某个固定值  

**优点：**   
简单  

**缺点：**  
可能有对称性问题，导致所有神经元更新方向相同，无法有效训练  
出现梯度爆炸或消失  

**适用场景：**  
测试或调试  

## 3.6 kaiming初始化：nn.init.kaiming_normal_()/nn.init.kaiming_uniform_()

**优点**  
适合和relu激活函数一起使用  

**缺点：**  
对非relu一般  

**适用场景**  
深层网络，10层及以上   

### 3.6.1 HE正态：nn.init.kaiming_normal_()
假设上一层的神经元个数（特征个数）为fan_in ,  $std = \sqrt{\frac{2}{fan_in}}$   
则，从均值为0,标准差为std中，随机抽取样本   
std值越大，w权重值离均值0分布相对较广，计算得到的内部状态值有较大的正值或负值   

### 3.6.2 HE均匀：nn.init.kaiming_uniform_()
假设上一层的神经元个数（特征个数）为fan_in ,  $limit = \sqrt{\frac{6}{fan_in}}$   
则，从[-limit,limit]中均匀抽取样本  

## 3.7 Xavier初始化：nn.init.xavier_normal_()/nn.init.xavier_uniform_()

**优点：**  
适用于Sigmoid、Tanh等激活函数，解决梯度消失问题  

**缺点：**  
对ReLU等激活函数表现欠佳  

**适用场景：**  
深度网络(10层及以上)，使用Sigmoid或Tanh激活函数  

### 3.7.1 正态nn.init.xavier_normal_()
假设上一层的神经元个数为fan_in, 本层的神经元个数为fan_out,  那么  $std = \sqrt{\frac{2}{fan\_in+fan\_out}}$  
取值范围为：均值为：0,标准差为：std  
std值越小，w权重值距离0分布越集中，计算可得内部装填值有较小的正直或负值   

### 3.7.2 均匀nn.init.xavier_uniform_()
假设上一层输出的（本层输入的）神经元个数为fan_in, 本层的神经元个数为fan_out,  那么  $std = \sqrt{\frac{6}{fan\_in+fan\_out}}$  
取值范围为：[-limit,limit]  

## 3.8 代码案例


# 4. 神经网络模型的搭建(正向传播)
## 4.1 搭建方法
定义继承自nn.Moudle的模型类
在_init_方法中定义网络中的层结构
在forward方法中定义数据传输方式
## 4.2 举例

输入层:(5*3)
隐藏层1:3个神经元，(5*3)，参数12个：weight(3*3),bias(1*3)
隐藏层2:2个神经元，(5*2),参数8个：weight(3*2),bias(1*2)
输出层:2个神经元，(5*2),参数6个：weight(2*2),bias(1*2)

**代码**  

```python
import torch.nn
from torch import nn
from torchsummary import summary


class MyModel(nn.Module):
    def __init__(self):
        # 调用父类的初始化方法，初始化属性值
        super(MyModel, self).__init__()

        # 第1个隐藏层，
        self.linear1 = nn.Linear(3, 3)  # 输入特征3，输出特征3
        nn.init.xavier_normal_(self.linear1.weight)  # 初始化权重
        nn.init.zeros_(self.linear1.bias)  # 初始化偏置

        # 第2个隐藏层
        self.linear2 = nn.Linear(3, 2)  # 输入特征3，输出特征2
        nn.init.kaiming_normal_(self.linear2.weight)  # 初始化权重
        nn.init.zeros_(self.linear2.bias)  # 初始化偏置

        # 输出层
        self.output = nn.Linear(2, 2)  # 输入特征2，输出特征2

    # 前向传播，从输入层 - 隐藏层 - 输出层
    # 底层会自动调用该函数
    def forward(self, x):
        # 第1个隐藏层计算：加权求和+激活函数
        # x = self.linear1(x)  # 加权求和
        # x = torch.sigmoid(x) # 激活函数
        x = torch.sigmoid(self.linear1(x))

        # 第2个隐藏层计算：加权求和+激活函数
        x = torch.relu(self.linear2(x))

        # 输出层计算：加权求和+激活函数
        # dim=-1:按行计算
        x = torch.softmax(self.output(x), dim=-1)  # 激活函数

        # 返回预测值
        return x


def train():
    # 创建模型
    my_model = MyModel()
    # 创建随机样本
    data = torch.randn(5, 3)
    print("\ndata:\n", data)
    # 模型预测
    result = my_model(data)
    print("\nresult:\n", result)
    print("\nresult.shape:\n", result.shape)
    print("\nresult.requires_grad\n", result.requires_grad)

    # 计算和查看整个模型的参数
    print("=" * 40)
    print("\n计算整个模型的参数\n")
    summary(my_model, input_size=(5, 3))

    print("=" * 40)
    print("\n查看整个模型的参数\n")
    for name, param in my_model.named_parameters():
        print(name, param,"\n")


if __name__ == '__main__':
    train()

```

**运行结果：**  
```text
data:
 tensor([[ 0.6133, -0.6439,  1.4298],
        [-0.8388,  1.7895, -0.4667],
        [ 0.1753, -0.2807, -0.4236],
        [-2.3510, -1.0320, -0.9208],
        [-0.4933,  0.8716, -0.4945]])

result:
 tensor([[0.6799, 0.3201],
        [0.6590, 0.3410],
        [0.6604, 0.3396],
        [0.6276, 0.3724],
        [0.6577, 0.3423]], grad_fn=<SoftmaxBackward0>)

result.shape:
 torch.Size([5, 2])

result.requires_grad
 True
========================================

计算整个模型的参数

----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Linear-1                 [-1, 5, 3]              12
            Linear-2                 [-1, 5, 2]               8
            Linear-3                 [-1, 5, 2]               6
================================================================
Total params: 26
Trainable params: 26
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 0.00
Params size (MB): 0.00
Estimated Total Size (MB): 0.00
----------------------------------------------------------------
========================================

查看整个模型的参数

linear1.weight Parameter containing:
tensor([[-0.3775, -0.2500,  0.5781],
        [ 0.6796,  0.4124, -0.1862],
        [ 0.2898, -0.2008,  0.4139]], requires_grad=True) 

linear1.bias Parameter containing:
tensor([0., 0., 0.], requires_grad=True) 

linear2.weight Parameter containing:
tensor([[ 1.0501, -0.1004,  0.0976],
        [ 0.1577,  2.2687,  0.1924]], requires_grad=True) 

linear2.bias Parameter containing:
tensor([0., 0.], requires_grad=True) 

output.weight Parameter containing:
tensor([[ 0.2651,  0.5694],
        [-0.0785,  0.3119]], requires_grad=True) 

output.bias Parameter containing:
tensor([0.3204, 0.1583], requires_grad=True) 
```
