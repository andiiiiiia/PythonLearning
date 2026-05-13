# 1. 图像知识

**二值图像**  
0黑，1白，每个像素点只取0或1。  

**灰度图**  
单通道，0纯黑，255纯白，中间为过渡  

**索引图像**  

**真彩色RGB图像**  
RGB三通道，  

# 2. CNN概述(Convolutional Neural Network)

适用于处理图像数据。它通过卷积层、池化层和全连接层的组合，能够有效地提取图像中的特征并进行分类、检测或分割等任务。

**CNN神经网络主要包含:**  

卷积层：通过卷积核来提取图像局部特征，并生成特征图。  

激活函数层：引入非线性，使网络可以拟合复杂函数。  

池化层：降维。  

全连接层: 将卷积层和池化层提取的特征进行整合，用于最终的分类或回归任务。通常位于网络的最后几层。  

drop层：防止过拟合。  

归一化层：加速收敛，稳定网络训练。  

输出层:分类任务一般选择softmax，sigmoid等激活函数。回归任务一般选择linear，sigmoid等激活函数。  

# 3. 卷积层：  

## 3.1 计算原理和公式

假设输入图像大小为: (InputH,inputW,inputC)  

卷积核大小为: (FilterH,FilterW,FilterC)  

步长为：**stride**  

边缘填充(0填充)为：**padding**   

则计算得到的特征图大小为：

$OutputH = (InputH + 2 * padding - FilterH) / stride + 1$  

$OutputW = (InputW + 2 * padding - FilterW) / stride + 1$  


**注意：**  

1、每个卷积核的通道数要等于输入图像的通道数  $FilterC = InputC$  。

2、一个输入图像可能会有多个卷积核进行计算，每个卷积核的大小形状和通道数要相同，为了便于后续计算。   

3、由于FilterC = InputC ，因此，每个卷积核计算得到的对应的特征图的通道基本为1。  

4、多个卷积核一起生成的特征图的通道数为 **卷积核数量**，每1个通道对应一个卷积核提取的特征，所有通道组合在一起，表示输入图像的多种特征。  

5、在CNN网络中，一个卷积核一般对应一个神经元。   

## 3.2 padding

**为什么要设置padding?**   

- 为了能让输出特征图的尺寸和输入图一样  
- 保留图像边缘信息，因为边缘点只会被卷积核计算一次。
- 如果不使用 Padding，每一层的输出尺寸都会变小，最终可能导致特征图尺寸过小，无法继续处理。使用 Padding 可以保持尺寸一致，便于网络结构的设计和优化。  

**padding类型**  

|Padding 类型	|说明| 适用|
|--|--|--|
|Zero Padding（零填充）	|在图像边缘填充 0，是最常用的方式||
|Same Padding	|自动计算填充大小，使输出尺寸与输入一致|广泛应用于各种CNN架构中，因为它可以保持特征图的尺寸，方便网络设计和计算。|
|Valid Padding	|不进行填充，只进行有效卷积，输出尺寸会变小|适用于不需要保持输出尺寸的场景，或者输入图像足够大，边缘信息丢失不重要的情况。|
|full padding| 尽可能多地添加填充，使得卷积核的每个元素都至少在输入图像上滑动一次。输出尺寸会增大。|较少使用，因为它会增加计算量，并且可能会在边缘引入一些伪影。|

## 3.3 stride

stride（步长）指的是卷积核在图像上滑动时的步伐大小，即每次卷积时卷积核在图像中向右（或向下）移动的像素数。步长直接影响卷积操作后输出特征图的尺寸，以及计算量和模型的特征提取能力。   

**stride作用：**  

- 控制输出特征图的大小：stride 越大，输出特征图的尺寸越小。这可以减少计算量和参数数量，同时有助于实现下采样（downsampling）。  

- 控制感受野：stride 越大，卷积核在输入上移动得越快，感受野（receptive field）会更大，从而可以捕捉更大范围的特征。  

- 减少计算量：较大的 stride 会减少输出特征图的尺寸，从而减少后续层的计算量。

**stride选择**  

- stride=1：希望保留输入特征图的空间分辨率,常用与网络的前几层卷积层。避免图像细节敏感信息丢失。高精度的特征图（如目标检测、语义分割等任务）。

- stride>1: 下采样,降低特征图的分辨率,常用于网络的中后段，以减少计算量并扩大感受野。


## 3.4 单通道单卷积核卷积计算案例

假设 **input**为5*5*1的一维通道的灰度图像：

$$
\begin{bmatrix}
1&2&3&4&5\\
6&7&8&9&10\\
11&12&13&14&15\\
16&17&18&19&20\\
21&22&23&24&25\\
\end{bmatrix}
$$  

假设采用**垂直边缘检测的卷积核filter**为3*3*1:  

$$
\begin{bmatrix}
1&0&-1\\
1&0&-1\\
1&0&-1\\
\end{bmatrix}
$$  

**假设步长为1，padding为0**  

**计算局部区域1-1：**  
input局部区域1为3*3*1的区域：  

$$
\begin{bmatrix}
1&2&3\\
6&7&8\\
11&12&13\\
\end{bmatrix}
$$  

和卷积核中的逐个元素和相乘并求和，可得：  

$1 * 1 + 0 * 2 + (-1) * 3 + 1 * 6 + 0 * 7 + (-1) * 8 + 1 * 11 + 0 * 12 + (-1) * 13  =  -6$

**向左移动，同理，可计算局部区域1-2：**  

$2 - 4 + 7 - 9 + 12 - 14 = -6$  

**同理，可得局部区域1-3，2-1,2-2,2-3,3-1,3-2,3-3**  

**最终可得特征图为：**  

$$
\begin{bmatrix}
-6&-6&-6\\
-6&-6&-6\\
-6&-6&-6\\
\end{bmatrix}
$$  


## 3.5 多通道单卷积核卷积计算案例
input-channel1：  

$$
\begin{bmatrix}
1&2&3&3&3\\
4&5&6&6&6\\
7&8&9&9&9\\
11&12&13&14&15\\
16&17&18&19&20\\
\end{bmatrix}
$$  

input-channel2：  

$$
\begin{bmatrix}
10&11&12&12&12\\
13&14&15&15&15\\
16&17&18&18&18\\
16&17&18&18&18\\
16&17&18&18&18\\
\end{bmatrix}
$$  


input-channel3： 

$$
\begin{bmatrix}
19&20&21&21&21\\
22&23&24&24&24\\
25&26&27&27&27\\
25&26&27&27&27\\
25&26&27&27&27\\
\end{bmatrix}
$$  

Filter-channel1:   

$$
\begin{bmatrix}
1&0&-1\\
1&0&-1\\
1&0&-1\\
\end{bmatrix}
$$  

Filter-channel2：

$$
\begin{bmatrix}
0&1&0\\
0&1&0\\
0&1&0\\
\end{bmatrix}
$$  

Filter-channel3：

$$
\begin{bmatrix}
-1&0&1\\
-1&0&1\\
-1&0&1\\
\end{bmatrix}
$$  

**则Filter计算为3 * 3 * 1：**    

Filter(0,0) = 求和(input-channel1点乘filter-channel1) + 求和(input-channel2点乘filter-channel2) + 求和(input-channel3点乘filter-channel3) = (1 + 0 -3 + 4 + 0 -6 + 7 + 0 -9) + (0 + 11 + 0 + 0 + 14 + 0 + 0 + 17 + 0) + (-19 + 0 + 21 + -22 + 0 + 24 + -27 + 0 + 29) = 42  

同理可得Filter(0,1),Filter(0,2),Filter(1,0),Filter(1,1),Filter(1,2),Filter(2,0),Filter(2,1),Filter(2,2)。

## 3.6 卷积层的作用

**1. 特征提取（Feature Extraction）**   

核心作用：通过卷积核（Filter）在输入图像上滑动，提取图像的局部特征。  

特征类型：边缘、角点、纹理、形状等。  

自动学习：卷积核的参数在训练过程中自动学习，无需人工设计特征。  

**2. 局部感知（Local Receptive Field）**   

卷积核只关注图像的局部区域（如 3x3、5x5 的区域），而不是整个图像。  

这样可以减少参数数量，提高模型的泛化能力。  

**3. 权值共享（Weight Sharing）**    

同一个卷积核在整个图像上滑动，共享参数。  

这样可以大幅减少模型的参数量，提高计算效率。  

**4. 平移不变性（Translation Invariance）**    

卷积操作对图像的平移具有一定的鲁棒性，即图像中的特征无论出现在哪个位置，都能被检测到。  

这是通过池化层（如 Max Pooling）进一步增强的。  

**5. 多层特征提取（Hierarchical Feature Learning）**    

浅层卷积层：提取低级特征（如边缘、角点）。  

深层卷积层：提取高级特征（如物体轮廓、形状、语义信息）。  

通过多层卷积，网络可以逐步抽象出更复杂的特征。  

## 3.7 卷积层API与demo

**class torch.nn.modules.conv.Conv2d：**  

class torch.nn.modules.conv.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)  


```python
import matplotlib.pyplot as plt
import torch


# 1.加载图像-展示原图-卷积-特征图展示
def conv2d():
    # 1. 读取图像
    img = plt.imread("./data/image.jpg")
    # print("\nimg:\n", img)
    print("\nimg.shape:\n", img.shape)  # (168, 300, 3)

    # 2. img转换为张量
    img_tensor = torch.tensor(img, dtype=torch.float32)
    print("\nimg_tensor.shape:\n", img_tensor.shape)  # (168, 300, 3)

    # 3. HWC 转换为 CHW
    img_tensor_CHW = img_tensor.permute(2, 0, 1)
    print("\nimg_tensor_CHW.shape:\n", img_tensor_CHW.shape)

    # 4. unsqueeze 添加batch维度, 因为pytorch中tensor的卷积是针对batch的
    img_tensor_CHW_batch = img_tensor_CHW.unsqueeze(dim=0)
    print("\nimg_tensor_CHW_batch.shape:\n", img_tensor_CHW_batch.shape)

    # 5. 卷积计算,参数1表示输入图像的通道数，参数2表示输出整体特征图的通道数，参数3表示卷积核大小，stride步长，padding填充
    # 注意：在这个对象初始化过程中，卷积核是由pytorch默认创建的，
    # 一共有out_channels个卷积核。
    # 每个卷积核大小为kernel_size * kernel_size * in_channels
    # 每个卷积核的权重也是随机初始化的
    img_conv = torch.nn.Conv2d(in_channels=3, out_channels=4, kernel_size=3, stride=1, padding=1)

    # 6. 卷积计算
    img_conv_result = img_conv(img_tensor_CHW_batch)
    print("\nimg_conv_result.shape:\n", img_conv_result.shape) # torch.Size([1, 4, 168, 300])

    # 取出第一个数据，并转回HWC
    img_conv_result_zero_HWC = img_conv_result[0].permute(1, 2, 0)
    print("\nimg_conv_result_zero_HWC.shape:\n", img_conv_result_zero_HWC.shape) # torch.Size([168, 300,4])

    # 7. 可视化第一个通道的特征图
    plt.imshow(img_conv_result_zero_HWC[:, :, 0].detach().numpy())
    plt.show()
    # 8. 可视化第二个通道的特征图
    plt.imshow(img_conv_result_zero_HWC[:, :, 1].detach().numpy())
    plt.show()
    # 9. 可视化第三个通道的特征图
    plt.imshow(img_conv_result_zero_HWC[:, :, 2].detach().numpy())
    plt.show()
    # 10. 可视化第四个通道的特征图
    plt.imshow(img_conv_result_zero_HWC[:, :, 3].detach().numpy())
    plt.show()

if __name__ == '__main__':
    conv2d()

```

**测试结果：**  
```text
img.shape:
 (168, 300, 3)

img_tensor.shape:
 torch.Size([168, 300, 3])

img_tensor_CHW.shape:
 torch.Size([3, 168, 300])

img_tensor_CHW_batch.shape:
 torch.Size([1, 3, 168, 300])

img_conv_result.shape:
 torch.Size([1, 4, 168, 300])

img_conv_result_zero_HWC.shape:
 torch.Size([168, 300, 4])
```

# 4. 池化层

## 4.1 什么是池化层

池化层是一种非线性操作，它通过在特征图上滑动一个固定大小的窗口（称为池化窗口），对窗口内的数据进行聚合操作（如取最大值、平均值等），从而生成一个更小的特征图。  

## 4.2 池化层有什么作用

**1. 降低特征图的空间尺寸（高度和宽度）**   

- 通过下采样，减少特征图的尺寸，从而减少后续层的计算量。  

- 例如，一个 5x5 的特征图经过 2x2 的最大池化后，可能变成 2x2。  

**2. 减少参数数量和计算量**  

- 池化层没有可学习的参数，只进行固定操作，因此可以显著减少模型的参数数量和计算量。  

**3. 增强模型的平移不变性**  

- 池化操作对图像中的小平移具有一定的鲁棒性，即图像在平移后，池化后的特征图变化较小。  

- 这有助于模型更好地识别图像中的目标，而不受位置变化的影响。  

**4. 防止过拟合**  

- 通过减少特征图的尺寸和参数数量，池化层有助于防止模型过拟合。

## 4.3 池化层常见类型

**最大池化（Max Pooling）：**  
- 取池化窗口内的最大值  
- 保留最显著的特征  

**平均池化（Average Pooling）：**	 
- 取池化窗口内的平均值  
- 保留整体特征，平滑图像  

## 4.4 池化层计算公式

在池化中，和卷积层类似，也有padding和stride，以及池化窗口，只是计算方式不同。

假设输入大小为: (InputH,inputW,inputC)  

池化窗口大小为: (PoolH,PoolW,PoolC)  

步长为：**stride**  

边缘填充(0填充)为：**padding**   

则计算得到的特征图大小为：

$OutputH = (InputH + 2 * padding - PoolH) / stride + 1$  

$OutputW = (InputW + 2 * padding - PoolW) / stride + 1$  

## 4.5 多通道，单池化窗口池化

**注意！**

1、一个池化层中，池化窗口只能是**单通道**的。

2、一个池化层中，只能有一个池化窗口。

3、和卷积层不同，池化操作只会对每个通道单独进行，不会进行跨通道操作，**不会改变输出的通道数**。  

2、 每个通道都会被**同一个**池化窗口进行池化操作。在一个池化层中，不存在多通道，多池化窗口场景。    

## 4.6 池化层API和demo

**MaxPool2d:**  

class torch.nn.modules.pooling.MaxPool2d(kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)  

**AvgPool2d:**  

class torch.nn.modules.pooling.AvgPool2d(kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True, divisor_override=None)  

```python
import torch


def pool_demo():
    # 创建一个三通道的5*5的二维张量，作为上一个卷积层输出的特征图,CHW
    x = torch.tensor(
        [
            [
                [1., 2., 3., 4., 5.],
                [6., 7., 8., 9., 10.],
                [11., 12., 13., 14., 15.],
                [16., 17., 18., 19., 20.],
                [21., 22., 23., 24., 25.]
            ],
            [
                [1., 2., 29., 4., 30.],
                [6., 7., 8., 9., 10.],
                [11., 26., 13., 27., 15.],
                [16., 17., 18., 19., 20.],
                [21., 22., 28., 24., 25.]
            ],
            [
                [1., 52., 3., 4., 51.],
                [40., 7., 8., 9., 10.],
                [11., 41., 13., 14., 15.],
                [16., 17., 42., 19., 20.],
                [50., 22., 23., 43., 25.]
            ]
        ])
    print("\nx.shape:\n", x.shape)  # torch.Size([3, 5, 5])
    # 创建最大池化层
    maxpool_layer = torch.nn.MaxPool2d(kernel_size=3, stride=1, padding=0)
    output = maxpool_layer(x)
    print("\noutput.shape:\n", output.shape)  # torch.Size([3, 3, 3])
    print("\noutput:\n", output)
    pass


if __name__ == '__main__':
    pool_demo()
    pass

```

**执行结果：**  
```text
x.shape:
 torch.Size([3, 5, 5])

output.shape:
 torch.Size([3, 3, 3])

output:
 tensor([[[13., 14., 15.],
         [18., 19., 20.],
         [23., 24., 25.]],

        [[29., 29., 30.],
         [26., 27., 27.],
         [28., 28., 28.]],

        [[13., 14., 15.],
         [18., 19., 20.],
         [23., 24., 25.]]])
```

# 5. CNN案例 - CIFA10案例
```python
import matplotlib.pyplot as plt
import torch.nn
from torchsummary import summary
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor

BATCH_SIZE = 8


# 1. 准备数据集
def create_dataset():
    train_dataset = CIFAR10(root='./data', train=True, transform=ToTensor(), download=True)
    test_dataset = CIFAR10(root='./data', train=False, transform=ToTensor(), download=True)
    return train_dataset, test_dataset
    pass


# 2. 搭建神经网络
# 要求：
# 图片(CHW = 3*32*32) - 卷积层1(6个神经元，6个卷积核，padding=0,stride=1,每个CHW = 3*3*3) - 特征图(CHW = 6*30*30)
# - 池化层1(1个神经元，1个池化窗口，padding=0,stride=2,每个CHW = 1*2*2) - 特征图(CHW = 6*15*15)
# - 卷积层2(16个神经元，16个卷积核，padding=0,stride=1,每个CHW = 6*3*3) - 特征图(CHW = 16*13*13)
# - 池化层2(1个神经元，1个池化窗口，padding=0,stride=2,每个CHW = 1*2*2) - 特征图(CHW = 16*6*6)
# - 全连接层1(16*6*6)
# - 全连接层2(in = 576 , out = 120)
# - 全连接层3(in = 120 , out = 84)
# - 输出层(in = 84 , out = 10个分类)
class CFA_Model(torch.nn.Module):
    def __init__(self):
        super(CFA_Model, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 6, 3, padding=0, stride=1)
        self.pool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = torch.nn.Conv2d(6, 16, 3, padding=0, stride=1)
        self.pool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        # self.torch.nn.Flatten(),
        self.linear1 = torch.nn.Linear(16 * 6 * 6, 120)
        self.linear2 = torch.nn.Linear(120, 84)
        self.out = torch.nn.Linear(84, 10)
        pass

    def forward(self, x):
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = torch.relu(x)
        x = self.pool2(x)

        x = x.reshape(x.size(0), -1)

        x = torch.relu(self.linear1(x))
        x = torch.relu(self.linear2(x))
        x = self.out(x)
        # 这里不加softmax，是因为后续计算损失时，CrossEntropyLoss，会自带softmax
        return x


# 3. 模型训练
def train(train_dataset):
    # 定义模型
    model = CFA_Model()
    # 定义数据加载器
    data_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    # 定义损失函数
    criterion = torch.nn.CrossEntropyLoss()
    # 定义梯度优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    # 定义训练轮数
    epoches = 100
    for epoch_idx in range(epoches):
        # 信息汇总
        num = 0  # 样本数量
        total_loss = 0.0  # 单轮总损失

        for x, y_true in data_loader:
            # 开启训练模型
            model.train()
            # 正向传播
            y_pred = model(x)
            # 计算平均损失
            loss = criterion(y_pred, y_true)
            # 计算梯度
            optimizer.zero_grad()
            loss.backward()
            # 更新参数
            optimizer.step()

            # 信息汇总：累计总损失和总样本数
            total_loss += loss.item() * len(y_true)
            num += len(y_true)
        print("epoch:", epoch_idx, " loss:", total_loss / num)
    # 保存模型
    torch.save(model.state_dict(), "./model/CFA_Model.pth")
    pass


# 4. 模型验证
def valid(test_dataset):
    dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    # 定义模型
    model  = CFA_Model().load_state_dict(torch.load("./model/CFA_Model.pth"))
    # 开启验证模式
    model.eval()
    # 定义统计信息，用来计算精度
    total_correct = 0
    total_samples = 0

    # 遍历每个batch数据，
    for x, y_true in dataloader:
        # 正向传播
        y_pred = model(x) # 10个分类概率
        # 计算每个batch的精度
        y_pred = torch.argmax(y_pred, dim=1) # 某个分类的索引
        # 统计信息
        total_correct += (y_pred == y_true).sum()
        total_samples += len(y_true) # 一般等于batchsize，但是最后一个batch不一定，考虑除不尽的情况
    print("验证集精度：", total_correct / total_samples)
    pass



if __name__ == '__main__':
    # 1. 查看数据集
    train_data, test_data = create_dataset()
    # print(train_data)
    # #  {'airplane': 0, 'automobile': 1, 'bird': 2, 'cat': 3, 'deer': 4, 'dog': 5, 'frog': 6, 'horse': 7, 'ship': 8, 'truck': 9}
    # print("\n数据集类别：\n", train_data.class_to_idx)
    # # 图像展示
    # plt.figure(figsize=([2, 2]))
    # plt.imshow(train_data.data[11])
    # plt.title(train_data.targets[11])
    # plt.show()

    # 2.测试模型
    # model = CFA_Model()
    # # 卷积层参数数量计算： (输入通道数 * 本卷积层的卷积核面积尺寸 + 1个偏置) * 本卷积层的卷积核数量
    # summary(model, input_size=(3, 32, 32), batch_size=BATCH_SIZE)

    # 3. 训练模型
    train(train_data)

    # 4. 验证数据集
    # valid(test_data)
    pass

```

**训练模型 - 执行结果**  

```text
epoch: 0  loss: 1.9691118521118165
epoch: 1  loss: 1.6765101409816743
epoch: 2  loss: 1.5215903290748596
...
epoch: 98  loss: 0.2795184834892
epoch: 99  loss: 0.27494999399202874
```
