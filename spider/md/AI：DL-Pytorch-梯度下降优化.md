# 1. 梯度下降算法

用来使损失函数最小化的方法  

# 2. 补充：神经网络中的epoch，batch，iter的区别和联系

epoch:一个训练周期，整个训练数据集被完整的训练一次。通常会训练多个 epoch，直到模型性能不再提升。  

batch：一个批次，每次输入模型进行训练的数据量。多batch 可以提高训练效率和模型泛化能力。  

itr：一次迭代，每使用一个 batch 数据进行一次前向传播和反向传播，就完成一次 iteration。 iteration 的数量 = 总样本数 / batch_size。   

注意：一个batch的数据通常只做一次前向传播和反向传播，然后更新模型的参数。再由下一batch继续训练。

# 3. 理解反向传播算法
## 3.1 举例说明
### 3.1.1 初始化：
**input_features：**  

$$
input = \begin{bmatrix}
{input_{1} = 0.05}&{input_{2} = 0.1}\\
\end{bmatrix}
$$  

**input_labels：**  

$$
label = \begin{bmatrix}
{y_{1} = 0.1}&{y_{2} = 0.99}\\
\end{bmatrix}
$$  

**hide1：**

$$
hide1\\_weight = \begin{bmatrix}
{w_{111}=0.15}&{w_{121}=0.25}\\
{w_{112}=0.2}&{w_{122}=0.3}\\
\end{bmatrix}
$$  

$$
hide1\\_bias = \begin{bmatrix}
{b_{11}=0.35}&{b_{12}=0.35}\\
\end{bmatrix}
$$  

**output：**

$$
output\\_weight = \begin{bmatrix}
{w_{211}=0.4}&{w_{221}=0.5}\\
{w_{212}=0.45}&{w_{222}=0.55}\\
\end{bmatrix}
$$  

$$
output\\_bias = \begin{bmatrix}
{b_{21}=0.6}&{b_{22}=0.6}\\
\end{bmatrix}
$$  

**备注**  

权重w的第1,2,3个下标分别表示，第几层，第几个神经元，以及神经元内第几个参数。因为输入层无超参，所以从hide1开始计数。    

### 3.1.2 正向传播

**hide1层第1个神经元：**  

加权求和：  

$net_{11} = input_{1} * w_{111} + input_{2} * w_{112} + b_{11} = 0.05 * 0.15 + 0.1 * 0.2 + 0.35 = 0.3775$  

激活函数：  

$out_{11} = sigmoid(net_{11}) = \frac{1}{1 + e^{-net_{11}}} = \frac{1}{1 + e^{-0.3775}} = 0.593269992$  

**hide1层第2个神经元：**  

加权求和：  

$net_{12} = input_{1} * w_{121} + input_{2} * w_{122} + b_{12} = 0.05 * 0.25 + 0.1 * 0.3 + 0.35 = 0.3925$  

激活函数：  

$out_{12} = sigmoid(net_{12}) = \frac{1}{1+e^{-net_{12}}} = \frac{1}{1 + e^{-0.3925}} = 0.596884378$  

**output层第1个神经元：**  

加权求和：  

$net_{21} = out_{11} * w_{211} + out_{12} * w_{212} + b_{21} = 0.593269992 * 0.4+0.596884378 * 0.45+0.6 = 1.105905967$  

激活函数：  

$out_{21} = sigmoid(net_{21}) = \frac{1}{1 + e^{-net_{21}}} = \frac{1}{1 + e^{-1.105905967}} = 0.75136507$  

**output层第2个神经元：**  

加权求和：  

$net_{22} = out_{11} * w_{221} + out_{12} * w_{222} + b_{22} = 0.593269992 * 0.5 + 0.596884378 * 0.55+0.6 = 1.2249214039$  

激活函数：  

$out_{22} = sigmoid(net_{22}) = \frac{1}{1 + e^{-net_{22}}} = \frac{1}{1 + e^{-1.2249214039}} = 0.772928465$  

### 3.1.3 计算损失MSE

$Loss = \frac{1}{2}(y_{1} - out_{21})^{2} + \frac{1}{2}(y_{2} - out_{22})^{2}$  


### 3.1.4 反向传播

**求  $w_{211}$  的更新后的权重？**  

**梯度计算：**  

$\frac{\partial{Loss}}{\partial{w_{211}}} = \frac{\partial{Loss}}{\partial{out_{21}}} * \frac{\partial{out_{21}}}{\partial{net_{21}}} * \frac{\partial{net_{21}}}{\partial{w_{211}}}$  
 
$\frac{\partial{Loss}}{\partial{out_{21}}} = out_{21} - y_{1} = 0.75136507 - 0.1 = 0.65136507 $  
 
$\frac{\partial{out_{21}}}{\partial{net_{21}}} = out_{21} * (1-out_{21}) = 0.75136507 * (1-0.75136507) = 0.186815602 $  

$\frac{\partial{net_{21}}}{\partial{w_{211}}} = out_{11} = 0.593269992 $  
 
**权重更新：**  

$w_{211} = w_{211} - \eta * \frac{\partial{Loss}}{\partial{w_{211}}}$   


**求  $w_{111}$  的更新后的权重？**  

同理可得：  

$\frac{\partial{Loss}}{\partial{w_{111}}} = (\frac{\partial{Loss}}{\partial{out_{21}}} * \frac{\partial{out_{21}}}{\partial{net_{21}}} * \frac{\partial{net_{21}}}{\partial{out_{11}}}  + \frac{\partial{Loss}}{\partial{out_{22}}} * \frac{\partial{out_{22}}}{\partial{net_{22}}} * \frac{\partial{net_{22}}}{\partial{out_{11}}} )* \frac{\partial{out_{11}}}{\partial{net_{11}}} * \frac{\partial{net_{11}}}{\partial{w_{111}}}$


# 4. 梯度下降优化方法

## 4.1 直接求偏导，然后更新权重不就行了，为什么还要梯度下降优化方法？  
- 直接梯度下降容易陷入局部最小值。优化方法（如 Momentum、Adam）可以帮助模型跳出局部最小值  
- 直接梯度下降更新方向不稳定。优化方法（如 Momentum、RMSProp、Adam）可以平滑梯度方向，使训练更稳定  
- 学习率难设置。优化方法（如 Adam）可以自动调整学习率，不需要手动调参  
- 不同参数的梯度变化幅度不同。优化方法（如 RMSProp、Adam）可以对不同参数使用不同的学习率
- 平缓区域，梯度值小，参数变化缓慢
- 碰到鞍点，梯度为0，参数无法更新

图例略  

## 4.2 基础：指数加权移动平均 EWMA
主要特点是对历史数据赋予不同的权重，具体是离当前时刻越近的数据权重越大，而权重按照指数方式递减。  

### 4.2.1 基本原理与公式：  
假设有一组数据 $x_{1},x_{2}...x_{t}$  ,则这组数据的EWMA计算公式如下： 

$S_{t} = \alpha * x_{t} + (1-\alpha)*S_{t-1}$  

$S_{t}$  : 时间点t的指数移动加权平均值  

$x_{t}$  : 事件单t的实际观测值   

$\alpha$ : 平滑系数  $(0< \alpha < 1)$  ,决定了当前数据点的权重。如果接近 1，说明当前数据点对平均值的影响较大，平均值变化较敏感。如果接近 0，说明历史数据对平均值的影响较大，平均值变化较平缓。  

$S_{t-1}$ : 前一时刻的EWMA  

### 4.2.2 示例：  
数据序列： $x = [10,12,15,13,14,16,18]$  ,平滑系数为：  $\alpha = 0.3$    
$S_{0} = x_{0} = 10$  
$S_{1} = \alpha * x_{1} + (1-\alpha) * S_{0}= 10.6$  
$S_{2} = \alpha * x_{2} + (1-\alpha) * S_{1}= 11.92$  

### 4.2.3 优缺点：  
**优点：**  

- 能够快速适应数据的最新变化。

- 计算简单，效率高。

**缺点：**  

- 对于数据中的周期性波动可能会产生延迟反应。  

- 如果数据存在趋势性或季节性，需要结合其他方法进行分析。  

**应用场景：**  

## 4.3 动量算法Momentum SGD
### 4.3.1 基本原理与公式

它通过引入一个“动量”项来利用梯度的历史信息，使参数更新具有一定的“惯性”。  

在标准的梯度下降方法中，参数更新完全依赖于当前梯度，这可能导致：  

- 收敛速度慢：尤其是当目标函数的曲率变化较大时。  

- 震荡：当目标函数的曲率不一致时，梯度下降可能会在某个方向上反复震荡。  

MomentM通过引入一个“动量”项，使参数更新不仅依赖于当前梯度，还依赖于之前梯度的累积信息。这可以：  

- 加速收敛：通过“惯性”帮助模型更快地向最小值方向移动。  

- 减少震荡：通过累积历史梯度信息，避免在某个方向上反复震荡。   


**计算原始梯度**  

$梯度grad_{t} = \frac{\partial{Loss}}{\partial{w}}$   

**计算动量项**    

$动量项v_{t} = \alpha * v_{t-1} + g_{t}$   

**权重更新**  

$w_{t} = w_{t-1} - \eta * v_{t}$  

**动量系数**   

动量系数越大，对历史数据依赖越强，更新越平滑   

动量系数越小，对当前数据依赖越强，更新越灵活     

若动量系数=0，则退化为标准的梯度下降  

### 4.3.2 举例说明

假设动量系数为0.9，初始化动量项为v0 = 0，batch1时，计算梯度为grad1    

batch1:  假设计算梯度：grad1  动量项：v1 = 0.9 * v0 + grad1 = grad1   

batch2:  假设计算梯度：grad2  动量项：v2 = 0.9 * v1 + grad2 = grad1 + grad2   

batch3:  假设计算梯度：grad3  动量项：v2 = 0.9 * v2 + grad3 = 0.9 * ( grad1 + grad2) + grad3   

### 4.3.3 代码说明
```python
import torch


def momentum():
    #
    print("="*50)
    print("\nmomentum示例:\n")
    print("\nbatch1:\n")
    # 初始化权重值
    weight = torch.tensor([1.0], requires_grad=True, dtype=torch.float32)
    print("\nbatch 1 ：更新前weight为:\n", weight.detach().numpy())
    # 2. 定义损失函数
    loss = ((weight ** 2) / 2.0).sum()
    # 3. 创建优化器对象
    # sgd+参数 -> momentum, sgd指定动量参数alpha = 0.9, lr:学习率,默认初始化动量v0 = 0
    optimizer = torch.optim.SGD([weight], lr=0.01, momentum=0.9)
    print("\n momentum优化器为:\n", optimizer)
    # 4. 计算梯度值：梯度清零，反向传播，参数更新
    optimizer.zero_grad()
    loss.backward()  # 计算梯度
    optimizer.step()  # 更新weight
    print("\nbatch 1 ：梯度weight.grad为:\n", weight.grad.numpy())
    print("\nbatch 1 ：更新后weight为:\n", weight.detach().numpy())


    # 第二次更新计算梯度，并更新weight
    print("\nbatch2:\n")
    print("\nbatch 2 ：更新前weight为:\n", weight.detach().numpy())
    # 5. 定义损失函数
    loss = ((weight ** 2) / 2.0).sum()
    # 6. 计算梯度值：梯度清零，反向传播，参数更新
    optimizer.zero_grad()
    loss.backward()  # 计算梯度
    optimizer.step()  # 更新weight
    print("\nbatch 2 ：梯度weight.grad为:\n", weight.grad.numpy())
    print("\nbatch 2 ：更新后weight为:\n", weight.detach().numpy())

if __name__ == '__main__':
    momentum()
```

**执行结果：**  
```text
==================================================

momentum示例:


batch1:


batch 1 ：更新前weight为:
 [1.]

 momentum优化器为:
 SGD (
Parameter Group 0
    dampening: 0
    differentiable: False
    foreach: None
    fused: None
    lr: 0.01
    maximize: False
    momentum: 0.9
    nesterov: False
    weight_decay: 0
)

batch 1 ：梯度weight.grad为:
 [1.]

batch 1 ：更新后weight为:
 [0.99]

batch2:


batch 2 ：更新前weight为:
 [0.99]

batch 2 ：梯度weight.grad为:
 [0.99]

batch 2 ：更新后weight为:
 [0.97110003]
```

### 4.3.4 优缺点

**优点：**  

- 惯性机制，加速收敛  

- 平滑梯度，减少震荡  

- 非凸场景，冲过鞍点或局部最小值  

- 动量缓冲，对学习率鲁棒性强  

**缺点：**  

- 动量过大，冲过全局最小值，设置合理的动量系数，一般在0.9左右  

- 无法自适应调整学习率  

### 4.3.5 应用场景

深度学习，cnn，rnn  

非凸问题优化  

梯度噪声较大时  

## 4.4 自适应学习率优化 AdaGrad (Adaptive Gradient Estimation)
### 4.4.1 基本原理与公式

为每个参数维护一个不同的学习率，从而在训练过程中自动调整学习率的大小。  

**初始化：**  

给定学习率  $\eta > 0$  ,累计平方梯度 $G_{0} = 0$  ,防止除以0的小常数  $\epsilon = 10^{-8}$  。

**计算累计平方梯度：**  

$G_{t} = G_{t-1} + grad_{t} \bigodot grad_{t}$  

ps:   $\bigodot$  代表逐个元素相乘，点乘  

**更新权重：**  

$w_{new} = w_{old} - \frac{\eta}{\sqrt{G_{t}+\epsilon}} \bigodot grad_{t}$  


### 4.4.2 举例说明

**初始化**：学习率为  $\eta_{0}$  ,累计平方梯度  $G_{0} = 零向量$      

**batch1**:  假设计算梯度：grad1。  累计平方梯度：  $G_{1} = G_{0} + grad_{1} \bigodot grad_{1}$  。学习率：  $\eta_{1}  = \frac{\eta_{0}}{\sqrt{G_{1}} + \epsilon}$   

**batch2**:  假设计算梯度：grad2。  累计平方梯度：  $G_{2} = G_{1} + grad_{2} \bigodot grad_{2}$  。学习率：  $\eta_{2}  = \frac{\eta_{0}}{\sqrt{G_{2}} + \epsilon}$   

**batch3**:  假设计算梯度：grad3。  累计平方梯度：  $G_{3} = G_{2} + grad_{3} \bigodot grad_{3}$  。学习率：  $\eta_{3}  = \frac{\eta_{0}}{\sqrt{G_{3}} + \epsilon}$   

### 4.4.3 代码说明
```python
def ada_grad():
    #
    print("=" * 50)
    print("\nada_grad示例:\n")
    print("\nbatch1:\n")
    # 初始化权重值
    weight = torch.tensor([1.0], requires_grad=True, dtype=torch.float32)
    print("\nbatch 1 ：更新前weight为:\n", weight.detach().numpy())
    # 2. 定义损失函数
    loss = ((weight ** 2) / 2.0).sum()
    # 3. 创建优化器对象
    # Adagrad指定学习率0.01
    optimizer = torch.optim.Adagrad([weight], lr=0.01)
    print("\n Adagrad优化器为:\n", optimizer)
    # 4. 计算梯度值：梯度清零，反向传播，参数更新
    optimizer.zero_grad()
    loss.backward()  # 计算梯度
    optimizer.step()  # 更新weight
    print("\nbatch 1 ：梯度weight.grad为:\n", weight.grad.numpy())
    print("\nbatch 1 ：更新后weight为:\n", weight.detach().numpy())

    # 第二次更新计算梯度，并更新weight
    print("\nbatch2:\n")
    print("\nbatch 2 ：更新前weight为:\n", weight.detach().numpy())
    # 5. 定义损失函数
    loss = ((weight ** 2) / 2.0).sum()
    # 6. 计算梯度值：梯度清零，反向传播，参数更新
    optimizer.zero_grad()
    loss.backward()  # 计算梯度
    optimizer.step()  # 更新weight
    print("\nbatch 2 ：梯度weight.grad为:\n", weight.grad.numpy())
    print("\nbatch 2 ：更新后weight为:\n", weight.detach().numpy())
```

**执行结果**  
1 -> 0.9829646  

```text
==================================================

ada_grad示例:


batch1:


batch 1 ：更新前weight为:
 [1.]

 Adagrad优化器为:
 Adagrad (
Parameter Group 0
    differentiable: False
    eps: 1e-10
    foreach: None
    fused: None
    initial_accumulator_value: 0
    lr: 0.01
    lr_decay: 0
    maximize: False
    weight_decay: 0
)

batch 1 ：梯度weight.grad为:
 [1.]

batch 1 ：更新后weight为:
 [0.99]

batch2:


batch 2 ：更新前weight为:
 [0.99]

batch 2 ：梯度weight.grad为:
 [0.99]

batch 2 ：更新后weight为:
 [0.9829646]
```

### 4.4.4 优缺点

### 4.4.5 应用场景

## 4.5 RMSProp (Root Mean Square Propagation)
### 4.5.1 基本原理和代码

在adaGrad基础上，在计算学习率时，增加**滑动平均**来缓解AdaGrad的学习率过早衰减的问题。  

**初始化：**  

给定学习率  $\eta > 0$  ,滑动平均累计平方梯度  $G_{0} = 0$  ,防止除以0的小常数  $\epsilon = 10^{-8}$  。滑动系数(衰减率)：  $\alpha$  。  

**计算滑动平均累计平方梯度：**  

$G_{t} = \alpha * G_{t-1} + (1-\alpha) * grad_{t} \bigodot grad_{t}$  

ps:   $\bigodot$  代表逐个元素相乘，点乘  

**更新权重：**   

$w_{new} = w_{old} - \frac{\eta}{\sqrt{G_{t}+\epsilon}} \bigodot grad_{t}$  


### 4.5.2 举例说明

初始化：学习率为  $\eta_{0}$  ,滑动平均累计平方梯度  $G_{0} = 零向量$  ,滑动系数(衰减率)为：  $\alpha$  。  

batch1:  假设计算梯度：grad1。  滑动平均累计平方梯度：  $G_{1} = \alpha * G_{0} + (1-\alpha) * grad_{1} \bigodot grad_{1}$  。学习率：  $\eta_{1}  = \frac{\eta_{0}}{\sqrt{G_{1}} + \epsilon}$     

batch2:  假设计算梯度：grad2。  滑动平均累计平方梯度：  $G_{2} = \alpha * G_{1} + (1-\alpha) * grad_{2} \bigodot grad_{2}$  。学习率： $\eta_{2}  = \frac{\eta_{0}}{\sqrt{G_{2}} + \epsilon}$  

batch3:  假设计算梯度：grad3。  滑动平均累计平方梯度：  $G_{3} = \alpha * G_{2} + (1-\alpha) * grad_{3} \bigodot grad_{3}$  。学习率：  $\eta_{3}  = \frac{\eta_{0}}{\sqrt{G_{3}} + \epsilon}$   

### 4.5.3 代码说明
```python
    #
    print("=" * 50)
    print("\nRMSprop示例:\n")
    print("\nbatch1:\n")
    # 初始化权重值
    weight = torch.tensor([1.0], requires_grad=True, dtype=torch.float32)
    print("\nbatch 1 ：更新前weight为:\n", weight.detach().numpy())
    # 2. 定义损失函数
    loss = ((weight ** 2) / 2.0).sum()
    # 3. 创建优化器对象
    # Adagrad指定学习率0.01
    optimizer = torch.optim.RMSprop([weight], lr=0.01,alpha=0.99)
    print("\n RMSprop优化器为:\n", optimizer)
    # 4. 计算梯度值：梯度清零，反向传播，参数更新
    optimizer.zero_grad()
    loss.backward()  # 计算梯度
    optimizer.step()  # 更新weight
    print("\nbatch 1 ：梯度weight.grad为:\n", weight.grad.numpy())
    print("\nbatch 1 ：更新后weight为:\n", weight.detach().numpy())

    # 第二次更新计算梯度，并更新weight
    print("\nbatch2:\n")
    print("\nbatch 2 ：更新前weight为:\n", weight.detach().numpy())
    # 5. 定义损失函数
    loss = ((weight ** 2) / 2.0).sum()
    # 6. 计算梯度值：梯度清零，反向传播，参数更新
    optimizer.zero_grad()
    loss.backward()  # 计算梯度
    optimizer.step()  # 更新weight
    print("\nbatch 2 ：梯度weight.grad为:\n", weight.grad.numpy())
    print("\nbatch 2 ：更新后weight为:\n", weight.detach().numpy())
```

**执行结果**   
1 -> 0.832918  

```text
==================================================

RMSprop示例:


batch1:


batch 1 ：更新前weight为:
 [1.]

 RMSprop优化器为:
 RMSprop (
Parameter Group 0
    alpha: 0.99
    capturable: False
    centered: False
    differentiable: False
    eps: 1e-08
    foreach: None
    lr: 0.01
    maximize: False
    momentum: 0
    weight_decay: 0
)

batch 1 ：梯度weight.grad为:
 [1.]

batch 1 ：更新后weight为:
 [0.90000004]

batch2:


batch 2 ：更新前weight为:
 [0.90000004]

batch 2 ：梯度weight.grad为:
 [0.90000004]

batch 2 ：更新后weight为:
 [0.832918]
```

## 4.6 自适应矩估计  Adam (Adaptive Momentum Estimation)
### 4.6.1 基本原理和公式
RmsProp(修正学习率) 和 Momentum(修正梯度) 相结合。 

**初始化：**  

给定学习率  $\eta > 0$  ,滑动平均累计平方梯度 $G_{0} = 0$  ,防止除以0的小常数  $\epsilon = 10^{-8}$  , 衰减率：  $\alpha$  。  

给定滑动系数  $\beta$  , 动量项初始化为0  $v_{0} = 0$  。

**计算累计平方梯度(为了修正学习率)：梯度的二阶矩估计**  

$G_{t} = \alpha * G_{t-1} + (1-\alpha) * grad_{t} \bigodot grad_{t}$  

偏差校正：  

$\hat{G_{t}} = \frac{G_{t}}{1-\alpha^t} $  

ps:   $\bigodot$  代表逐个元素相乘，点乘  

**计算动量项(为了修正梯度)：梯度的一阶矩估计**  

$v_{t} = \beta * v_{t-1} + (1-\beta) * grad_{t}$  

偏差校正：  

$\hat{v_{t}} = \frac{v_{t}}{1-\beta^t} $

**更新权重：**  

$w_{new} = w_{old} - \frac{\eta}{\sqrt{\hat{G_{t}}+\epsilon}} · \hat{v_{t}}$  

### 4.6.2 举例说明
### 4.6.3 代码说明
```python
def adam():
    #
    print("=" * 50)
    print("\nadam示例:\n")
    print("\nbatch1:\n")
    # 初始化权重值
    weight = torch.tensor([1.0], requires_grad=True, dtype=torch.float32)
    print("\nbatch 1 ：更新前weight为:\n", weight.detach().numpy())
    # 2. 定义损失函数
    loss = ((weight ** 2) / 2.0).sum()
    # 3. 创建优化器对象
    # adam指定beta1和beta2。beta1是梯度一阶矩估计参数，用来修正梯度。beta2是梯度二阶矩估计参数，用来修正学习率。
    optimizer = torch.optim.Adam([weight], lr=0.01, betas=(0.9, 0.99))
    print("\n adam优化器为:\n", optimizer)
    # 4. 计算梯度值：梯度清零，反向传播，参数更新
    optimizer.zero_grad()
    loss.backward()  # 计算梯度
    optimizer.step()  # 更新weight
    print("\nbatch 1 ：梯度weight.grad为:\n", weight.grad.numpy())
    print("\nbatch 1 ：更新后weight为:\n", weight.detach().numpy())
    # 第二次更新计算梯度，并更新weight
    print("\nbatch2:\n")
    print("\nbatch 2 ：更新前weight为:\n", weight.detach().numpy())
    # 5. 定义损失函数
    loss = ((weight ** 2) / 2.0).sum()
    # 6. 计算梯度值：清除上次计算的梯度，否则会累加上次计算的梯度，反向传播，参数更新
    optimizer.zero_grad()
    loss.backward()  # 计算梯度
    optimizer.step()  # 更新weight
    print("\nbatch 2 ：梯度weight.grad为:\n", weight.grad.numpy())
    print("\nbatch 2 ：更新后weight为:\n", weight.detach().numpy())
```
**执行结果**  
1 -> 0.9800025  

```text
==================================================

adam示例:


batch1:


batch 1 ：更新前weight为:
 [1.]

 adam优化器为:
 Adam (
Parameter Group 0
    amsgrad: False
    betas: (0.9, 0.99)
    capturable: False
    decoupled_weight_decay: False
    differentiable: False
    eps: 1e-08
    foreach: None
    fused: None
    lr: 0.01
    maximize: False
    weight_decay: 0
)

batch 1 ：梯度weight.grad为:
 [1.]

batch 1 ：更新后weight为:
 [0.99]

batch2:


batch 2 ：更新前weight为:
 [0.99]

batch 2 ：梯度weight.grad为:
 [0.99]

batch 2 ：更新后weight为:
 [0.9800025]
```
## 4.7 梯度下降优化方法选择
|优化算法|优点|缺点|适用场景|
|--|--|--|--|
|SGD|简单、容易实现。|收敛速度较慢，容易震荡，特别是在复杂问题中。|用于简单任务，或者当数据特征分布相对稳定时。|
|Momentum|可以加速收敛，减少震荡，特别是在高曲率区域。|需要手动调整动量超参数，可能会在小步长训练中过度更新。|用于非平稳优化问题，尤其是深度学习中的应用。|
|AdaGrad|自适应调整学习率，适用于稀疏数据。|学习率会在训练过程中逐渐衰减，可能导致早期停滞。|适合稀疏数据，如 NLP 或推荐系统中的特征。|
|RMSProp|解决了 AdaGrad 学习率过早衰减的问题，适应性强。|需要选择合适的超参数，更新可能会过于激进。|适用于动态问题、非平稳目标函数，如深度学习训练。|
|Adam|结合了Momentum和RMSProp的特点，适应性强且稳定。|需要调节更多的超参数，训练过程中可能会产生较大波动。|广泛适用于各种深度学习任务，特别是非平稳和复杂问题。|
