# 1. ANN案例-手机价格区间预测
## 1.1 创建Tensor数据集
```python
# 1. 获取并创建tensor数据集
def create_data():
    # 1. 加载数据
    data = pd.read_csv("./data/train.csv")
    # 2. 获取x特征和y标签
    x, y = data.iloc[:, :-1], data.iloc[:, -1]
    # 3. 类型转换浮点
    x = x.astype(np.float32)
    # 4. 分割训练集和测试集
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1, stratify=y)

    # 5. 将上述数据集封装成tensor数据集
    train_data = TensorDataset(torch.tensor(x_train.values), torch.tensor(y_train.values))
    test_data = TensorDataset(torch.tensor(x_test.values), torch.tensor(y_test.values))
    print("\n训练集:", train_data, "测试集:", test_data)
  
    # 6. 返回训练集和测试集，输入层特征数，标签数
    return train_data, test_data, x_train.shape[1], len(np.unique(y))
```
## 1.2 定义神经网络模型类

定义模型,三个连接层:  

hide1:    input_dim:20,   output_dim:128,     线性层     relu激活函数  

hide2:    input_dim:128,  output_dim:256,     线性层     relu激活函数  

output:   input_dim:256,  output_dim:4,       线性层     softmax激活函数  

```python
# 2. 定义模型
# 三个连接层
# hide1:    input_dim:20,   output_dim:128,     线性层     relu激活函数
# hide2:    input_dim:128,  output_dim:256,     线性层     relu激活函数
# output:   input_dim:256,  output_dim:4,       线性层     softmax激活函数
class PhonePricePredictModel(nn.Module):
    # 初始化：定义神经网络结构和
    def __init__(self, input_dim, output_dim):
        super(PhonePricePredictModel, self).__init__()

        # hide1:
        # 定义线性层
        self.linear1 = nn.Linear(input_dim, 128)
        # 初始化权重
        nn.init.xavier_normal_(self.linear1.weight)
        # 初始化偏置
        nn.init.zeros_(self.linear1.bias)

        # hide2:
        # 定义线性层
        self.linear2 = nn.Linear(128, 256)
        # 初始化权重
        nn.init.xavier_normal_(self.linear2.weight)
        # 初始化偏置
        nn.init.zeros_(self.linear2.bias)

        # linear3 输出层:
        # 定义线性层
        self.linear3 = nn.Linear(256, output_dim)
        # 初始化权重
        nn.init.xavier_normal_(self.linear3.weight)
        # 初始化偏置
        nn.init.zeros_(self.linear3.bias)

    def forward(self, x):
        # hide1
        x = torch.relu(self.linear1(x))

        # hide2
        x = torch.relu(self.linear2(x))

        # output
        x = self.linear3(x)
        # 后续使用CrossEntropyLoss,默认会先进行softmax计算,所以输出层不做softmax
        # x = torch.softmax(x, dim=1)
        return x
```
## 1.3 模型训练

**训练模型过程：**  

**准备工作：**     

- 创建数据加载器  
- 创建神经网络模型对象  
- 创建损失函数  
- 创建梯度下降优化器(Adam)  
- 定义epoch数  
- 执行每轮训练(见下述)  
- 保存模型参数  

**每轮训练：**     

- 定义总损失  
- 定义批次号(记录)  
- 执行每批次训练(见下述)  
- 打印信息(轮数，总损失)  

**每批次训练：**    

- model.train()切换模型状态  
- y_pred=model(x)前向传播(模型预测)  
- loss=criterion(y_true,y_pred)计算损失  
- optimizer.zero_grad()梯度清零  
- loss.backward()计算梯度  
- optimizer.step()更新权重  
- 累加总损失(可选)  
- 累加批次号(可选)  

```python
# 3. 训练模型：
# 准备工作：     创建数据加载器 - 创建神经网络模型对象 - 创建损失函数 - 创建梯度下降优化器(Adam) - 定义epoch数 - 执行每轮训练(见下述) - 保存模型参数
# 每轮训练：     定义总损失 - 定义批次号(记录) - 执行每批次训练(见下述) - 打印信息(轮数，总损失)
# 每批次训练：    model.train()切换模型状态 - y_pred=model(x)前向传播(模型预测) - loss=criterion(y_true,y_pred)计算损失 - optimizer.zero_grad()梯度清零 - loss.backward()计算梯度 - optimizer.step()更新权重 - 累加总损失(可选) - 累加批次号(可选)
def train(train_data, input_dim, output_dim):
    # 创建数据加载器
    train_loader = DataLoader(dataset=train_data, batch_size=16, shuffle=True)
    # 创建神经网络模型对象
    model = PhonePricePredictModel(input_dim, output_dim)
    # 创建损失函数
    criterion = torch.nn.CrossEntropyLoss()
    # 创建梯度下降优化器(Adam)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
    # 定义epoch数
    epochs = 100
    for epoch in range(epochs):
        # 定义总损失
        total_loss = 0.0
        # 定义批次号(记录)
        batch_num = 0
        # 执行每批次训练(见下述)
        for x, y_true in train_loader:
            # 切换模型状态
            model.train()
            # 前向传播(模型预测)
            y_pred = model(x)
            # 计算损失
            loss = criterion(y_pred, y_true)
            # 梯度清零
            optimizer.zero_grad()
            # backward计算梯度
            loss.backward()
            # step更新权重
            optimizer.step()
            # 累加总损失
            total_loss += loss.item()
            # 累加批次号(可选)
            batch_num += 1
        print("\n第", epoch + 1, "轮(epoch): \t总损失为: ", total_loss, "\t平均损失为: ", total_loss / batch_num)
    # 保存模型参数
    print("\n模型参数model.state_dict():\n", model.state_dict())
    torch.save(model.state_dict(), "./model/PhonePricePredictModel.pth")
    pass
```
## 1.4 模型验证

**准备工作:**     

- 创建数据加载器  
- 创建神经网络模型对象  
- 加载模型参数  
- 切换模型状态  
- 定义正确数  
- 执行每轮测试(见下述)  
- 计算预测精度    
- 打印信息(正确率)  

**每轮测试:**       
- y_pred=model(x)前向传播(模型预测)  
- argmax()调整预测值(因为模型没做softmax，输出的是每条数据的四个类别的概率)  
- 累加正确数(判断y_true是否等于y_pred)  

```python
# 4. 模型验证
# 准备工作:      创建数据加载器 - 创建神经网络模型对象 - 加载模型参数 - 切换模型状态 - 定义正确数 - 执行每轮测试(见下述) - 计算预测精度 - 打印信息(正确率)
# 每轮测试:       y_pred=model(x)前向传播(模型预测) - argmax()调整预测值(因为模型没做softmax，输出的是每条数据的四个类别的概率) - 累加正确数(判断y_true是否等于y_pred)
def model_valid(valid_data, input_dim, output_dim):
    # 加载测试数据集
    dataloader = DataLoader(dataset=valid_data, batch_size=8, shuffle=False)
    # 创建神经网络模型对象
    model = PhonePricePredictModel(input_dim, output_dim)
    # 加载模型参数
    model.load_state_dict(torch.load("./model/PhonePricePredictModel.pth"))
    # 切换模型状态
    model.eval()
    # 创建评估结果
    correct = 0
    # 执行每轮测试(见下述)
    for x, y_true in dataloader:
        # 前向传播(模型预测)
        y_pred = model(x)
        # argmax,dim=1表示按行处理
        y_pred = torch.argmax(y_pred, dim=1)
        # 判断y_true是否等于y_pred
        correct += (y_pred == y_true).sum()
    # 计算预测精度
    acc = correct.item() / len(valid_data)
    # 打印信息(正确率)
    print("\n验证集准确率(acc):", acc)
```
## 1.5 测试运行
```python
if __name__ == '__main__':
    # 1. 准备数据集
    train_data, test_data, input_dim, output_dim = create_data()

    # 2. 构建神经网络
    phonePricePredictModel = PhonePricePredictModel(input_dim=input_dim, output_dim=output_dim)

    # 3. 计算模型参数
    # input_size函数: 计算批次数，
    # input_size=(16, input_dim)：即每个批次16条数据，每个数据20个特征
    # 问linear1有多少个参数？128*21 = 2688
    # summary(model=phonePricePredictModel, input_size=(16, input_dim))

    # 4. 模型训练
    train(train_data=train_data, input_dim=input_dim, output_dim=output_dim)

    # 5. 模型验证
    model_valid(valid_data=test_data, input_dim=input_dim, output_dim=output_dim)
```
**执行结果**  
```text
训练集: <torch.utils.data.dataset.TensorDataset object at 0x000002AA83848E60> 测试集: <torch.utils.data.dataset.TensorDataset object at 0x000002AAD27F0050>

第 1 轮(epoch): 	总损失为:  7174.983462691307 	平均损失为:  71.74983462691307

第 2 轮(epoch): 	总损失为:  138.66666531562805 	平均损失为:  1.3866666531562806

第 3 轮(epoch): 	总损失为:  138.66299450397491 	平均损失为:  1.3866299450397492

...

第 48 轮(epoch): 	总损失为:  138.66011607646942 	平均损失为:  1.3866011607646942

第 49 轮(epoch): 	总损失为:  138.64828312397003 	平均损失为:  1.3864828312397004

第 50 轮(epoch): 	总损失为:  138.64909064769745 	平均损失为:  1.3864909064769746

模型参数model.state_dict():
 OrderedDict({
 'linear1.weight': tensor([[-8.7637e-01,  1.3258e-03, -2.0371e-02,  ..., -5.4536e-02,
         -9.7315e-02, -5.3226e-02],
        [-3.6902e-01, -1.6995e-01,  1.0170e-01,  ...,  4.2495e-02,
          1.2214e-01,  9.5966e-02],
        [-1.6585e+00, -3.1248e-03, -3.5093e-02,  ..., -1.3673e-01,
         -3.5358e-02, -6.9825e-02],
        ...,
        [-1.3711e+00, -5.7987e-02,  1.1366e-02,  ...,  2.7649e-02,
         -1.3705e-01, -3.5838e-01],
        [-8.1465e-01,  4.4320e-02,  1.5587e-01,  ..., -2.4507e-01,
         -1.2332e-01, -4.8619e-02],
        [-2.4415e-02, -1.7366e-01, -1.2068e-01,  ...,  1.0814e-01,
          2.4328e-02,  1.4767e-01]]), 
  'linear1.bias': tensor([-9.1995e-04, -4.0020e-04, -1.5899e-03, -3.9210e-05, -2.4661e-03,
        -1.0050e-03, -6.6346e-04, -2.8653e-03, -1.9221e-03,  0.0000e+00,
         0.0000e+00,  0.0000e+00, -4.3972e-04, -6.3126e-05, -2.8198e-04,
        -3.4304e-03,  0.0000e+00, -5.1683e-04,  0.0000e+00, -1.0841e-03,
        -5.4588e-04, -2.9458e-03, -9.3961e-04, -1.1171e-04, -2.1574e-03,
        ...
         0.0000e+00, -7.9789e-04, -4.1432e-04, -1.7039e-03, -6.5103e-04,
        -4.3647e-05, -3.6526e-04, -2.0250e-03,  0.0000e+00, -3.5447e-04,
        -9.1757e-04, -6.8670e-04, -5.3979e-05]), 
'linear2.weight': tensor([[-0.2283, -0.0879,  0.0208,  ..., -0.1347,  0.0064, -0.0041],
        [ 0.0200,  0.0067, -0.0714,  ...,  0.0554,  0.0881, -0.0212],
        [-0.1065, -0.0477,  0.0914,  ...,  0.0745, -0.0450,  0.0324],
        ...,
        [-0.0367,  0.1075, -0.0682,  ...,  0.0539, -0.0024, -0.0077],
        [-0.2018, -0.1370, -0.1228,  ..., -0.0194,  0.0004,  0.0409],
        [ 0.0792, -0.0113, -0.1349,  ...,  0.0440, -0.0885,  0.0287]]), 
'linear2.bias': tensor([-2.9976e-03, -3.0983e-04, -1.7076e-03, -1.3549e-03, -1.0215e-03,
        -3.4981e-04, -5.9998e-03, -7.2409e-03, -6.7056e-04, -1.7604e-03,
        -1.5282e-03, -1.3486e-02, -1.0553e-03, -1.6571e-02, -1.9678e-02,
        -2.1707e-03, -6.7315e-04, -6.4319e-04, -4.7562e-03, -2.3026e-03,
        ...
        -2.1740e-03, -2.7091e-03, -7.0188e-04, -9.6513e-04, -3.1840e-03,
        -1.0516e-03, -3.3320e-03, -1.2392e-03, -1.0930e-02, -8.7682e-04,
        -6.8748e-05, -1.5160e-03, -4.4367e-03, -1.1176e-02, -2.7478e-03,
        -8.5050e-05]), 
'linear3.weight': tensor([[-0.0552,  0.0043, -0.1789,  ...,  0.5734, -0.1349,  0.4168],
        [-0.4958, -0.0669, -0.3975,  ...,  0.5737, -0.6370,  0.2777],
        [ 0.0839, -0.2493, -0.1267,  ...,  0.0436,  0.2369, -1.0201],
        [ 0.6085,  0.0355,  0.8351,  ..., -1.4549,  0.5348,  0.1467]]), 
'linear3.bias': tensor([ 0.0003,  0.0009, -0.0018,  0.0006])})

验证集准确率(acc): 0.25
```
## 1.6 调优方案
### 1.6.1 方法
- 数据预处理阶段对特征做标准化  
- 梯度下降优化算法  
- 学习率调整  
- 引入正则化层(drop或bn)  
- 增加网络深度  
- 调整训练轮数epoch
- 其他
### 1.6.2 调优后的代码和执行结果
```pythoon
import torch  # PyTorch框架，封装了张量的各种操作
from torch.utils.data import TensorDataset  # 数据集对象.  数据 ⇒ Tensor ⇒ 数据集 ⇒ 数据集加载
from torch.utils.data import DataLoader  # 数据加载器.
import torch.nn as nn  # neural network，封装了神经网络的各种操作
import torch.optim as optim  # 优化器
from sklearn.model_selection import train_test_split  # 训练集和测试集的划分
import matplotlib.pyplot as plt  # 绘图
import numpy as np  # 数组(矩阵)操作
import pandas as pd  # 数据处理
import time  # 时间模块
from torchsummary import summary


# 1. 获取并创建tensor数据集
def create_data():
    # 1. 加载数据
    data = pd.read_csv("./data/train.csv")
    # 2. 获取x特征和y标签
    x, y = data.iloc[:, :-1], data.iloc[:, -1]
    # 3. 类型转换浮点
    x = x.astype(np.float32)
    # 4. 分割训练集和测试集
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1, stratify=y)

    # 5. 将上述数据集封装成tensor数据集
    train_data = TensorDataset(torch.tensor(x_train.values), torch.tensor(y_train.values))
    test_data = TensorDataset(torch.tensor(x_test.values), torch.tensor(y_test.values))
    print("\n训练集:", train_data, "测试集:", test_data)
    # 训练集和测试集，输入层特征数，标签数
    return train_data, test_data, x_train.shape[1], len(np.unique(y))


# 2. 定义模型
# 三个连接层
# hide1:    input_dim:20,   output_dim:128,     线性层     relu激活函数
# hide2:    input_dim:128,  output_dim:256,     线性层     relu激活函数
# output:   input_dim:256,  output_dim:4,       线性层     softmax激活函数
class PhonePricePredictModel(nn.Module):
    # 初始化：定义神经网络结构和
    def __init__(self, input_dim, output_dim):
        super(PhonePricePredictModel, self).__init__()

        # hide1:
        # 定义线性层
        self.linear1 = nn.Linear(input_dim, 128)
        # 模型优化1 - bn归一化
        self.bn1 = nn.BatchNorm1d(128)  # 添加BN层
        # 初始化权重
        nn.init.xavier_normal_(self.linear1.weight)
        # 初始化偏置
        nn.init.zeros_(self.linear1.bias)

        # hide2:
        # 定义线性层
        self.linear2 = nn.Linear(128, 256)
        # 模型优化1 - bn归一化
        self.bn2 = nn.BatchNorm1d(256)  # 添加BN层
        # 初始化权重
        nn.init.xavier_normal_(self.linear2.weight)
        # 初始化偏置
        nn.init.zeros_(self.linear2.bias)

        # 模型优化2： 增加网络深度
        # hide3:
        # 定义线性层
        self.linear3 = nn.Linear(256, 512)
        # 模型优化1 - bn归一化
        self.bn3 = nn.BatchNorm1d(512)  # 添加BN层
        # 初始化权重
        nn.init.xavier_normal_(self.linear3.weight)
        # 初始化偏置
        nn.init.zeros_(self.linear3.bias)

        # hide4:
        # 定义线性层
        self.linear4 = nn.Linear(512, 128)
        # 模型优化1 - bn归一化
        self.bn4 = nn.BatchNorm1d(128)  # 添加BN层
        # 初始化权重
        nn.init.xavier_normal_(self.linear4.weight)
        # 初始化偏置
        nn.init.zeros_(self.linear4.bias)

        # linear3 输出层:
        # 定义线性层
        self.output = nn.Linear(128, output_dim)
        # 初始化权重
        nn.init.xavier_normal_(self.output.weight)
        # 初始化偏置
        nn.init.zeros_(self.output.bias)

    def forward(self, x):
        # hide1
        x = self.linear1(x)
        # 模型优化1 - bn归一化
        x = self.bn1(x)
        x = torch.relu(x)

        # hide2
        x = self.linear2(x)
        # 模型优化1 - bn归一化
        x = self.bn2(x)
        x = torch.relu(x)

        # 模型优化2： 增加网络深度
        # hide3
        x = self.linear3(x)
        # 模型优化1 - bn归一化
        x = self.bn3(x)
        x = torch.relu(x)

        # hide4
        x = self.linear4(x)
        # 模型优化1 - bn归一化
        x = self.bn4(x)
        x = torch.relu(x)

        # output
        x = self.output(x)
        # 后续使用CrossEntropyLoss,默认会先进行softmax计算,所以输出层不做softmax
        # x = torch.softmax(x, dim=1)
        return x


# 3. 训练模型：
# 准备工作：     创建数据加载器 - 创建神经网络模型对象 - 创建损失函数 - 创建梯度下降优化器(Adam) - 定义epoch数 - 执行每轮训练(见下述) - 保存模型参数
# 每轮训练：     定义总损失 - 定义批次号(记录) - 执行每批次训练(见下述) - 打印信息(轮数，总损失)
# 每批次训练：    model.train()切换模型状态 - y_pred=model(x)前向传播(模型预测) - loss=criterion(y_true,y_pred)计算损失 - optimizer.zero_grad()梯度清零 - loss.backward()计算梯度 - optimizer.step()更新权重 - 累加总损失(可选) - 累加批次号(可选)
def train(train_data, input_dim, output_dim):
    # 创建数据加载器
    train_loader = DataLoader(dataset=train_data, batch_size=16, shuffle=True)
    # 创建神经网络模型对象
    model = PhonePricePredictModel(input_dim, output_dim)
    # 创建损失函数
    criterion = torch.nn.CrossEntropyLoss()
    # 创建梯度下降优化器(Adam)
    # 模型优化3：学习率下降，
    # 模型优化4：采用adam梯度下降优化算法
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999))
    # 定义epoch数
    # 模型优化5：增加epochs轮数
    epochs = 100
    for epoch in range(epochs):
        # 定义总损失
        total_loss = 0.0
        # 定义批次号(记录)
        batch_num = 0
        # 执行每批次训练(见下述)
        for x, y_true in train_loader:
            # 切换模型状态
            model.train()
            # 前向传播(模型预测)
            y_pred = model(x)
            # 计算损失
            loss = criterion(y_pred, y_true)
            # 梯度清零
            optimizer.zero_grad()
            # backward计算梯度
            loss.backward()
            # step更新权重
            optimizer.step()
            # 累加总损失
            total_loss += loss.item()
            # 累加批次号(可选)
            batch_num += 1
        print("\n第", epoch + 1, "轮(epoch): \t总损失为: ", total_loss, "\t平均损失为: ", total_loss / batch_num)
    # 保存模型参数
    print("\n模型参数model.state_dict():\n", model.state_dict())
    torch.save(model.state_dict(), "./model/PhonePricePredictModel.pth")
    pass


# 4. 模型验证
# 准备工作:      创建数据加载器 - 创建神经网络模型对象 - 加载模型参数 - 切换模型状态 - 定义正确数 - 执行每轮测试(见下述) - 计算预测精度 - 打印信息(正确率)
# 每轮测试:       y_pred=model(x)前向传播(模型预测) - argmax()调整预测值(因为模型没做softmax，输出的是每条数据的四个类别的概率) - 累加正确数(判断y_true是否等于y_pred)
def model_valid(valid_data, input_dim, output_dim):
    # 加载测试数据集
    dataloader = DataLoader(dataset=valid_data, batch_size=8, shuffle=False)
    # 创建神经网络模型对象
    model = PhonePricePredictModel(input_dim, output_dim)
    # 加载模型参数
    model.load_state_dict(torch.load("./model/PhonePricePredictModel.pth"))
    # 切换模型状态
    model.eval()
    # 创建评估结果
    correct = 0
    # 执行每轮测试(见下述)
    for x, y_true in dataloader:
        # 前向传播(模型预测)
        y_pred = model(x)
        # argmax,dim=1表示按行处理
        y_pred = torch.argmax(y_pred, dim=1)
        # 判断y_true是否等于y_pred
        correct += (y_pred == y_true).sum()
    # 计算预测精度
    acc = correct.item() / len(valid_data)
    # 打印信息(正确率)
    print("\n验证集准确率(acc):", acc)


if __name__ == '__main__':
    # 1. 准备数据集
    train_data, test_data, input_dim, output_dim = create_data()
    # 2. 构建神经网络
    phonePricePredictModel = PhonePricePredictModel(input_dim=input_dim, output_dim=output_dim)
    # 3. 计算模型参数
    # input_size函数: 计算批次数，
    # input_size=(16, input_dim)：即每个批次16条数据，每个数据20个特征
    # 问linear1有多少个参数？128*21 = 2688
    # summary(model=phonePricePredictModel, input_size=(16, input_dim))
    # 4. 模型训练
    train(train_data=train_data, input_dim=input_dim, output_dim=output_dim)
    # 5. 模型验证
    model_valid(valid_data=test_data, input_dim=input_dim, output_dim=output_dim)

```
**执行结果**  
```text
模型参数：略
验证集准确率(acc): 0.925
```
