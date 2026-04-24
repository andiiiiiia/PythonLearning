https://docs.pytorch.org/docs/2.5/torch.html

# 1. Tensor创建
**代码**  
```python
def type_change():
    print("=" * 64, "\n 张量类型转换")
    # 创建时指定类型
    tensor1 = torch.tensor([[1, 2], [3, 4]],dtype=torch.float)
    print("tensor1:\n", tensor1, "\ntype:\n", type(tensor1), "\ndtype:\n", tensor1.dtype)
    # 创建后，转换
    tensor2 = tensor1.type(torch.int)
    print("\ntensor2:\n", tensor2, "\ntype:\n", type(tensor2), "\ndtype:\n", tensor2.dtype)
    # half()-float16,float()-float32, double()-float64,short()-int16,int()-int32,long()-int64...
    tensor3 = tensor2.double()
    print("\ntensor3:\n", tensor3, "\ntype:\n", type(tensor3), "\ndtype:\n", tensor3.dtype)


# 创建线性和随机张量
def linear_random_create():
    print("=" * 64, "\n 创建线性和随机张量")
    # 创建线性张量
    tensor1 = torch.arange(start=0, end=10, step=2)
    print("tensor1:\n", tensor1, "\ntype:\n", type(tensor1))

    tensor2 = torch.linspace(start=0, end=5, steps=4)  # steps：元素个数
    print("\ntensor2:\n", tensor2, "\ntype:\n", type(tensor2))

    # 创建随机张量
    torch.initial_seed()  # 默认方式，使用房钱系统时间作为随机种子
    torch.manual_seed(3)  # 手动设置随机种子
    tensor3 = torch.rand(2, 3)  # [0,1)均匀分布
    print("\ntensor3:\n", tensor3, "\ntype:\n", type(tensor3))

    # 创建证书随机张量
    tensor4 = torch.randint(low=0, high=10, size=(3, 5))  # size:张量形状
    print("\ntensor4:\n", tensor4, "\ntype:\n", type(tensor4))


# 创建全0或1或指定值的张量
def all_same_create():
    print("=" * 64, "\n 创建全0或1或指定值的张量")
    tensor1 = torch.zeros(2, 3)
    print("\ntensor1:\n", tensor1, "\ntype:\n", type(tensor1))

    tensor2 = torch.tensor([[1., 2.], [4., 5.], [7., 8.]])
    tensor3 = torch.zeros_like(tensor2)
    print("\ntensor3:\n", tensor3, "\ntype:\n", type(tensor3))

    tensor4 = torch.ones(2, 3)
    print("\ntensor4:\n", tensor4, "\ntype:\n", type(tensor4))

    tensor5 = torch.tensor([[1., 2.], [4., 5.], [7., 8.]])
    tensor6 = torch.ones_like(tensor5)
    print("\ntensor6:\n", tensor6, "\ntype:\n", type(tensor6))

    tensor7 = torch.full(size=(2, 3), fill_value=521)
    print("\ntensor7:\n", tensor7, "\ntype:\n", type(tensor7))

    tensor8 = torch.tensor([[1., 2.], [4., 5.], [7., 8.]])
    tensor9 = torch.full_like(input=tensor8, fill_value=520)
    print("\ntensor9:\n", tensor9, "\ntype:\n", type(tensor9))


# 使用torch.tensor创建，根据数据
def tensor_create():
    print("=" * 64, "\n 使用torch.tensor创建，根据数据")
    tensor1 = torch.tensor(1)
    print("tensor1:\n", tensor1, "\n", type(tensor1))
    tensor2 = torch.tensor([1, 2, 3, 4, 5])
    print("tensor2:\n", tensor2, "\n", type(tensor2))
    tensor3 = torch.tensor(np.array([[1, 2, 3], [4, 5, 6]]), dtype=torch.float)
    print("tensor3:\n", tensor3, "\n", type(tensor3))
    tensor4 = torch.tensor(np.random.randint(size=(2, 3), low=0, high=10))
    print("tensor4:\n", tensor4, "\n", type(tensor4))


# 使用Tensor创建，使用数据或size
def Tensor_create():
    print("=" * 64, "\n  使用Tensor创建，使用数据或size")
    tensor21 = torch.Tensor(1)
    print("tensor21:\n", tensor21, "\n", type(tensor21))
    tensor22 = torch.Tensor([1, 2, 3, 4, 5])
    print("tensor22:\n", tensor22, "\n", type(tensor22))
    tensor23 = torch.Tensor(np.array([[1, 2, 3], [4, 5, 6]]))
    print("tensor23:\n", tensor23, "\n", type(tensor23))
    data24 = np.random.randint(size=(2, 3), low=0, high=10)
    tensor24 = torch.Tensor(data24)
    print("tensor24:\n", tensor24, "\n", type(tensor24))
    tensor25 = torch.Tensor(2, 3)
    print("tensor25:\n", tensor25, "\n", type(tensor25))


# IntTensor: int类型Tensor
# 当data类型和Tensor支持的类型不一致时，会尝试自动转换类型，若转换失败则报错
def IntTensor_create():
    print("=" * 64, "\n IntTensor: int类型Tensor")
    tensor31 = torch.IntTensor(1)
    print("tensor31:\n", tensor31, "\n", type(tensor31), "\n", tensor31.dtype)
    tensor32 = torch.IntTensor([1, 2, 3, 4, 5])
    print("tensor32:\n", tensor32, "\n", type(tensor32), "\n", tensor32.dtype)
    tensor33 = torch.IntTensor(np.array([[1, 2, 3], [4, 5, 6]]))
    print("tensor33:\n", tensor33, "\n", type(tensor33), "\n", tensor33.dtype)
    tensor34 = torch.IntTensor(np.random.randint(size=(2, 3), low=0, high=10))
    print("tensor34:\n", tensor34, "\n", type(tensor34), "\n", tensor34.dtype)
    tensor35 = torch.IntTensor(2, 3)
    print("tensor35:\n", tensor35, "\n", type(tensor35), "\n", tensor35)


if __name__ == '__main__':
    # tensor的type默认是float32

    tensor_create()

    Tensor_create()

    IntTensor_create()

    all_same_create()

    linear_random_create()

    type_change()
```

**结果**  
```text
================================================================ 
 使用torch.tensor创建，根据数据
tensor1:
 tensor(1) 
 <class 'torch.Tensor'>
tensor2:
 tensor([1, 2, 3, 4, 5]) 
 <class 'torch.Tensor'>
tensor3:
 tensor([[1., 2., 3.],
        [4., 5., 6.]]) 
 <class 'torch.Tensor'>
tensor4:
 tensor([[0, 7, 5],
        [2, 1, 8]], dtype=torch.int32) 
 <class 'torch.Tensor'>
================================================================ 
  使用Tensor创建，使用数据或size
tensor21:
 tensor([0.]) 
 <class 'torch.Tensor'>
tensor22:
 tensor([1., 2., 3., 4., 5.]) 
 <class 'torch.Tensor'>
tensor23:
 tensor([[1., 2., 3.],
        [4., 5., 6.]]) 
 <class 'torch.Tensor'>
tensor24:
 tensor([[0., 6., 3.],
        [3., 3., 6.]]) 
 <class 'torch.Tensor'>
tensor25:
 tensor([[0., 0., 0.],
        [0., 0., 0.]]) 
 <class 'torch.Tensor'>
================================================================ 
 IntTensor: int类型Tensor
tensor31:
 tensor([16843009], dtype=torch.int32) 
 <class 'torch.Tensor'> 
 torch.int32
tensor32:
 tensor([1, 2, 3, 4, 5], dtype=torch.int32) 
 <class 'torch.Tensor'> 
 torch.int32
tensor33:
 tensor([[1, 2, 3],
        [4, 5, 6]], dtype=torch.int32) 
 <class 'torch.Tensor'> 
 torch.int32
tensor34:
 tensor([[6, 5, 0],
        [5, 3, 6]], dtype=torch.int32) 
 <class 'torch.Tensor'> 
 torch.int32
tensor35:
 tensor([[1036126176,       1361,          0],
        [         0,          0,          0]], dtype=torch.int32) 
 <class 'torch.Tensor'> 
 tensor([[1036126176,       1361,          0],
        [         0,          0,          0]], dtype=torch.int32)
================================================================ 
 创建全0或1或指定值的张量

tensor1:
 tensor([[0., 0., 0.],
        [0., 0., 0.]]) 
type:
 <class 'torch.Tensor'>

tensor3:
 tensor([[0., 0.],
        [0., 0.],
        [0., 0.]]) 
type:
 <class 'torch.Tensor'>

tensor4:
 tensor([[1., 1., 1.],
        [1., 1., 1.]]) 
type:
 <class 'torch.Tensor'>

tensor6:
 tensor([[1., 1.],
        [1., 1.],
        [1., 1.]]) 
type:
 <class 'torch.Tensor'>

tensor7:
 tensor([[521, 521, 521],
        [521, 521, 521]]) 
type:
 <class 'torch.Tensor'>

tensor9:
 tensor([[520., 520.],
        [520., 520.],
        [520., 520.]]) 
type:
 <class 'torch.Tensor'>
================================================================ 
 创建线性和随机张量
tensor1:
 tensor([0, 2, 4, 6, 8]) 
type:
 <class 'torch.Tensor'>

tensor2:
 tensor([0.0000, 1.6667, 3.3333, 5.0000]) 
type:
 <class 'torch.Tensor'>

tensor3:
 tensor([[0.0043, 0.1056, 0.2858],
        [0.0270, 0.4716, 0.0601]]) 
type:
 <class 'torch.Tensor'>

tensor4:
 tensor([[0, 5, 1, 9, 4],
        [1, 5, 9, 6, 9],
        [5, 4, 0, 4, 6]]) 
type:
 <class 'torch.Tensor'>
================================================================ 
 张量类型转换
tensor1:
 tensor([[1., 2.],
        [3., 4.]]) 
type:
 <class 'torch.Tensor'> 
dtype:
 torch.float32

tensor2:
 tensor([[1, 2],
        [3, 4]], dtype=torch.int32) 
type:
 <class 'torch.Tensor'> 
dtype:
 torch.int32

tensor3:
 tensor([[1., 2.],
        [3., 4.]], dtype=torch.float64) 
type:
 <class 'torch.Tensor'> 
dtype:
 torch.float64
```



# 2. Tensor运算
```python
import torch
import numpy as np


# 张量加减乘除负运算
def tensor_add_sub_mul_div_neg():
    print("=" * 40)
    print("张量加减乘除负运算")

    print("\nadd,sub,mul,div,neg,也可以用符号+,-,*,/,-,不改变原数据:")
    tensor1 = torch.randint(0, 10, (2, 3))
    tensor2 = tensor1.add(10)
    tensor3 = tensor1.sub(10)
    tensor4 = tensor1.mul(10)
    tensor5 = tensor1.div(10)
    tensor6 = tensor1.neg()
    print("\ntensor1:\n", tensor1)
    print("\ntensor2 = tensor1.add(10):\n", tensor2)
    print("\ntensor3 = tensor1.sub(10):\n", tensor3)
    print("\ntensor4 = tensor1.nul(10):\n", tensor4)
    print("\ntensor5 = tensor1.div(10):\n", tensor5)
    print("\ntensor6 = tensor1.neg():\n", tensor6)

    print("\n改变原数据:")
    tensor7 = torch.randint(0, 10, (2, 3))
    print("\nbefore : tensor7:\n", tensor7)
    tensor8 = tensor7.add_(10)
    print("\nafter : tensor7:\n", tensor7)
    print("\ntensor8 = tensor7.add_(10):\n", tensor8)


# 张量均值求和平方运算
def   tensor_sum_mean_pow():
    print("=" * 40)
    print("张量均值求和平方运算")

    tensor1 = torch.randint(0, 10, (2, 3),dtype=torch.float64)
    print("\ntensor1:\n", tensor1)

    print("\n计算均值mean，max和min同理")
    mean = tensor1.mean()
    mean_dim_0 = tensor1.mean(dim=0)
    mean_dim_1 = tensor1.mean(dim=1)
    print("\ntensor1.mean():\n", mean)
    print("\ntensor1.mean(dim=0)，按列:\n", mean_dim_0)
    print("\ntensor1.mean(dim=1)，按行:\n", mean_dim_1)

    print("\n计算求和")
    sum = tensor1.sum()
    sum_dim_0 = tensor1.sum(dim=0)
    sum_dim_1 = tensor1.sum(dim=1)
    print("\ntensor1.sum():\n", sum)
    print("\ntensor1.sum(dim=0)，按列:\n", sum_dim_0)
    print("\ntensor1.sum(dim=1)，按行:\n", sum_dim_1)

    print("\n计算平方")
    pow = tensor1.pow(2)
    print("\ntensor1.pow(2):\n", pow)

    print("\n计算平方根")
    sqrt = tensor1.sqrt()
    print("\ntensor1.sqrt():\n", sqrt)

    print("\n计算exp，e的n次方")
    exp = tensor1.exp()
    print("\ntensor1.exp():\n", exp)

    print("\n计算log,e为底")
    log = tensor1.log()
    print("\ntensor1.log():\n", log)

    print("\n计算log,10为底")
    log10 = tensor1.log10()
    print("\ntensor1.log10():\n", log10)

# 张量矩阵点乘运算
def tensor_dot():
    print("=" * 40)
    print("张量点乘")

    tensor1 = torch.tensor([[1, 2, 3], [4, 5, 6]])
    tensor2 = torch.tensor([[6, 5, 4], [3, 2, 1]])
    tensor3 = tensor1.mul(tensor2)
    tensor4 = tensor1 * tensor2
    print("\ntensor1:\n", tensor1)
    print("\ntensor2:\n", tensor2)
    print("\ntensor3 = tensor1.mul(tensor2):\n", tensor3)
    print("\ntensor4 = tensor1*tensor2:\n", tensor4)


# 张量矩阵乘法运算
def tensor_matmul():
    print("=" * 40)
    print("张量矩阵乘法")

    tensor1 = torch.tensor([[1, 2, 3], [4, 5, 6]])
    tensor2 = torch.tensor([[6, 5, ], [4, 3], [2, 1]])
    tensor3 = tensor1.matmul(tensor2)
    tensor4 = tensor1 @ tensor2
    print("\ntensor1:\n", tensor1)
    print("\ntensor2:\n", tensor2)
    print("\ntensor3 = tensor1.matmul(tensor2):\n", tensor3)
    print("\ntensor4 = tensor1@tensor2:\n", tensor4)


if __name__ == '__main__':
    tensor_add_sub_mul_div_neg()
    tensor_sum_mean_pow()
    tensor_dot()
    tensor_matmul()

```
**结果**  
```text
========================================
张量加减乘除负运算

add,sub,mul,div,neg,也可以用符号+,-,*,/,-,不改变原数据:

tensor1:
 tensor([[7, 7, 2],
        [3, 9, 8]])

tensor2 = tensor1.add(10):
 tensor([[17, 17, 12],
        [13, 19, 18]])

tensor3 = tensor1.sub(10):
 tensor([[-3, -3, -8],
        [-7, -1, -2]])

tensor4 = tensor1.nul(10):
 tensor([[70, 70, 20],
        [30, 90, 80]])

tensor5 = tensor1.div(10):
 tensor([[0.7000, 0.7000, 0.2000],
        [0.3000, 0.9000, 0.8000]])

tensor6 = tensor1.neg():
 tensor([[-7, -7, -2],
        [-3, -9, -8]])

改变原数据:

before : tensor7:
 tensor([[3, 7, 9],
        [3, 5, 9]])

after : tensor7:
 tensor([[13, 17, 19],
        [13, 15, 19]])

tensor8 = tensor7.add_(10):
 tensor([[13, 17, 19],
        [13, 15, 19]])
========================================
张量均值求和平方运算

tensor1:
 tensor([[0., 0., 4.],
        [1., 6., 4.]], dtype=torch.float64)

计算均值mean，max和min同理

tensor1.mean():
 tensor(2.5000, dtype=torch.float64)

tensor1.mean(dim=0)，按列:
 tensor([0.5000, 3.0000, 4.0000], dtype=torch.float64)

tensor1.mean(dim=1)，按行:
 tensor([1.3333, 3.6667], dtype=torch.float64)

计算求和

tensor1.sum():
 tensor(15., dtype=torch.float64)

tensor1.sum(dim=0)，按列:
 tensor([1., 6., 8.], dtype=torch.float64)

tensor1.sum(dim=1)，按行:
 tensor([ 4., 11.], dtype=torch.float64)

计算平方

tensor1.pow(2):
 tensor([[ 0.,  0., 16.],
        [ 1., 36., 16.]], dtype=torch.float64)

计算平方根

tensor1.sqrt():
 tensor([[0.0000, 0.0000, 2.0000],
        [1.0000, 2.4495, 2.0000]], dtype=torch.float64)

计算exp，e的n次方

tensor1.exp():
 tensor([[  1.0000,   1.0000,  54.5982],
        [  2.7183, 403.4288,  54.5982]], dtype=torch.float64)

计算log,e为底

tensor1.log():
 tensor([[  -inf,   -inf, 1.3863],
        [0.0000, 1.7918, 1.3863]], dtype=torch.float64)

计算log,10为底

tensor1.log10():
 tensor([[  -inf,   -inf, 0.6021],
        [0.0000, 0.7782, 0.6021]], dtype=torch.float64)
========================================
张量点乘

tensor1:
 tensor([[1, 2, 3],
        [4, 5, 6]])

tensor2:
 tensor([[6, 5, 4],
        [3, 2, 1]])

tensor3 = tensor1.mul(tensor2):
 tensor([[ 6, 10, 12],
        [12, 10,  6]])

tensor4 = tensor1*tensor2:
 tensor([[ 6, 10, 12],
        [12, 10,  6]])
========================================
张量矩阵乘法

tensor1:
 tensor([[1, 2, 3],
        [4, 5, 6]])

tensor2:
 tensor([[6, 5],
        [4, 3],
        [2, 1]])

tensor3 = tensor1.matmul(tensor2):
 tensor([[20, 14],
        [56, 41]])

tensor4 = tensor1@tensor2:
 tensor([[20, 14],
        [56, 41]])
```

# 3. Tensor索引操作
```python
# 张量的单行列的索引操作
import torch


def tensor_indexing_with_single_row_or_column():
    print("=" * 40)
    print("张量的单行列索引")

    tensor1 = torch.randint(0, 10, [3, 4])
    tensor2 = tensor1[0]  # 第0行
    tensor3 = tensor1[:, 0]  # 所有行，第0列
    tensor4 = tensor1[0, :]
    print("\n原始张量：\n", tensor1)
    print("\ntensor1[0]：\n", tensor2)
    print("\ntensor1[:,0]：\n", tensor3)
    print("\ntensor1[0,:]：\n", tensor4)
    pass


# 张量的列表索引
def tensor_indexing_with_lists():
    print("=" * 40)
    print("张量的列表索引")

    tensor1 = torch.randint(0, 10, [3, 4])
    tensor2 = tensor1[[0, 1], [1, 2]]
    tensor3 = tensor1[[[0], [2]], [1, 2]]
    print("\n原始张量：\n", tensor1)
    print("\ntensor1[[0, 1], [1, 2]]：\n", tensor2)
    print("\ntensor1[[[0],[2]],[1,2]]：\n", tensor3)
    pass


# 张量的范围索引
def tensor_indexing_with_ranges():
    print("=" * 40)
    print("张量的范围索引")
    tensor1 = torch.randint(0, 10, [3, 4])
    tensor2 = tensor1[:2, 1:]  # 取左不取右
    tensor3 = tensor1[2:, :2]  # 取左不取右
    print("\n原始张量：\n", tensor1)
    print("\ntensor1[:2,1:]：\n", tensor2)
    print("\ntensor1[2:,:2]：\n", tensor3)
    pass


# 张量的布尔索引
def tensor_indexing_with_booleans():
    print("=" * 40)
    print("张量的布尔索引")
    tensor1 = torch.randint(0, 10, [3, 4])
    tensor2 = tensor1[tensor1[:, 2] > 3]  # 第3列大于3的  行数据
    tensor3 = tensor1[:, tensor1[1] > 3]  # 第2行大于3的  列数据
    print("\n原始张量：\n", tensor1)
    print("\ntensor1[tensor1[:, :2] > 4]：\n", tensor2)
    print("\ntensor1[:, tensor1[1] > 5]：\n", tensor3)
    pass


# 张量的多维索引
def tensor_indexing_with_multiple_dimensions():
    print("=" * 40)
    print("张量的多维索引")

    tensor1 = torch.randint(0, 10, [3, 4, 5])
    tensor2 = tensor1[0, :, :]  # 获取0轴上的第一个数据
    tensor3 = tensor1[:, 0, :]  # 获取1轴上的第一个数据
    tensor4 = tensor1[:, :, 0]  # 获取2轴上的第一个数据
    print("\n原始张量：\n", tensor1)
    print("\ntensor1[0,:,:]：\n", tensor2)
    print("\ntensor1[:,0,:]：\n", tensor3)
    print("\ntensor1[:,:,0]：\n", tensor4)
    pass


if __name__ == '__main__':
    tensor_indexing_with_single_row_or_column()
    tensor_indexing_with_lists()
    tensor_indexing_with_ranges()
    tensor_indexing_with_booleans()
    tensor_indexing_with_multiple_dimensions()

```
**结果**  
```text
========================================
张量的单行列索引

原始张量：
 tensor([[0, 6, 1, 1],
        [8, 3, 5, 5],
        [4, 9, 6, 6]])

tensor1[0]：
 tensor([0, 6, 1, 1])

tensor1[:,0]：
 tensor([0, 8, 4])

tensor1[0,:]：
 tensor([0, 6, 1, 1])
========================================
张量的列表索引

原始张量：
 tensor([[6, 4, 4, 9],
        [8, 6, 7, 1],
        [8, 5, 4, 5]])

tensor1[[0, 1], [1, 2]]：
 tensor([4, 7])

tensor1[[[0],[2]],[1,2]]：
 tensor([[4, 4],
        [5, 4]])
========================================
张量的范围索引

原始张量：
 tensor([[3, 3, 6, 3],
        [6, 9, 7, 0],
        [1, 6, 4, 4]])

tensor1[:2,1:]：
 tensor([[3, 6, 3],
        [9, 7, 0]])

tensor1[2:,:2]：
 tensor([[1, 6]])
========================================
张量的布尔索引

原始张量：
 tensor([[3, 9, 8, 4],
        [9, 6, 2, 1],
        [3, 5, 4, 8]])

tensor1[tensor1[:, :2] > 4]：
 tensor([[3, 9, 8, 4],
        [3, 5, 4, 8]])

tensor1[:, tensor1[1] > 5]：
 tensor([[3, 9],
        [9, 6],
        [3, 5]])
========================================
张量的多维索引

原始张量：
 tensor([[[8, 5, 2, 9, 6],
         [8, 1, 0, 4, 3],
         [5, 5, 4, 9, 2],
         [9, 6, 4, 6, 3]],

        [[3, 0, 2, 6, 7],
         [2, 5, 7, 8, 3],
         [2, 0, 8, 9, 0],
         [6, 7, 6, 2, 1]],

        [[5, 2, 6, 6, 0],
         [7, 1, 5, 2, 2],
         [1, 9, 3, 4, 3],
         [8, 0, 7, 7, 1]]])

tensor1[0,:,:]：
 tensor([[8, 5, 2, 9, 6],
        [8, 1, 0, 4, 3],
        [5, 5, 4, 9, 2],
        [9, 6, 4, 6, 3]])

tensor1[:,0,:]：
 tensor([[8, 5, 2, 9, 6],
        [3, 0, 2, 6, 7],
        [5, 2, 6, 6, 0]])

tensor1[:,:,0]：
 tensor([[8, 8, 5, 9],
        [3, 2, 2, 6],
        [5, 7, 1, 8]])
```

# 4. Tensor形状操作
```python
import torch


# reshape,保证张量数据不变，变换指定的形状
def tensor_reshape():
    print("=" * 40)
    print("reshape,保证张量数据不变，变换指定的形状")
    tensor1 = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
    tensor1_shape = tensor1.shape
    tensor1_shape0 = tensor1.shape[0]
    tensor1_shape1 = tensor1.shape[1]
    tensor1_size = tensor1.size()
    tensor1_size0 = tensor1.size(0)
    tensor1_size1 = tensor1.size(1)
    tensor2 = tensor1.reshape(4, 2)
    print("\ntensor1:\n", tensor1)
    print("\ntensor1.shape:\n", tensor1_shape)
    print("\ntensor1.shape[0]:\n", tensor1_shape0)
    print("\ntensor1.shape[1]:\n", tensor1_shape1)
    print("\ntensor1.size:\n", tensor1_size)
    print("\ntensor1.size[0]:\n", tensor1_size0)
    print("\ntensor1.size[1]:\n", tensor1_size1)
    print("\ntensor1.reshape(4,2):\n", tensor2)
    pass


# 升维和降维
def tensor_squeeze_unsqueeze():
    print("=" * 40)
    print("升维和降维,在dim维度上新增 1个 维度")
    print("\n升维")
    tensor1 = torch.tensor([1, 2, 3, 4])  # size[4]
    tensor2 = tensor1.unsqueeze(dim=0)  # size[1,4]
    tensor3 = tensor1.unsqueeze(dim=1)  # size[4,1]
    tensor4 = tensor1.unsqueeze(dim=-1)  # -1表示最右边的维度 size[4,1]
    print("\ntensor1:\n", tensor1, "\nshape:\n", tensor1.shape)
    print("\ntensor2 = tensor1.unsqueeze(dim=0):\n", tensor2, "\nshape:\n", tensor2.shape)
    print("\ntensor3 = tensor1.unsqueeze(dim=1):\n", tensor3, "\nshape:\n", tensor3.shape)
    print("\ntensor4 = tensor1.unsqueeze(dim=-1):\n", tensor4, "\nshape:\n", tensor4.shape)

    print("\n降维不生效,移除所有大小为 1 的维度。")
    tensor5 = torch.tensor([[[1, 2, 3, 4], [5, 6, 7, 8]], [[9, 10, 11, 12], [13, 14, 15, 16]]])  # size[2,2,4]
    tensor6 = tensor5.squeeze()  # 不生效，因为原来的数据没有为1的维度
    print("\ntensor5:\n", tensor5, "\nshape:\n", tensor5.shape)
    print("\ntensor6 = tensor5.squeeze():\n", tensor6, "\nshape:\n", tensor6.shape)

    print("\n降维生效,移除所有大小为 1 的维度。")
    tensor7 = torch.tensor([[[1, 2, 3, 4]], [[5, 6, 7, 8]]])  # size[2,1,4]
    tensor8 = tensor7.squeeze()  # size[2,4] 生效，移除所有大小为 1 的维度
    print("\ntensor7:\n", tensor7, "\nshape:\n", tensor7.shape)
    print("\ntensor8 = tensor7.squeeze():\n", tensor8, "\nshape:\n", tensor8.shape)
    pass


# 张量维度交换transpose
def tensor_transpose():
    print("=" * 40)
    print("张量维度交换transpose,只能两个")

    tensor1 = torch.tensor(
        [
            [
                [1, 2, 3, 4],
                [5, 6, 7, 8],
                [5, 6, 7, 8]
            ],
            [
                [9, 10, 11, 12],
                [13, 14, 15, 16],
                [13, 14, 15, 16]
            ]
        ])  # size[2,3,4]
    tensor2 = tensor1.transpose(0, 1)  # size[3,2,4]
    print("\ntensor1:\n", tensor1, "\nshape:\n", tensor1.shape)
    print("\ntensor2 = tensor1.transpose(0, 1):\n", tensor2, "\nshape:\n", tensor2.shape)
    pass


# 张量维度交换permute
def tensor_permute():
    print("=" * 40)
    print("张量维度交换permute")
    tensor1 = torch.tensor(
        [
            [
                [1, 2, 3, 4],
                [5, 6, 7, 8],
                [5, 6, 7, 8]
            ],
            [
                [9, 10, 11, 12],
                [13, 14, 15, 16],
                [13, 14, 15, 16]
            ]
        ])  # size[2,3,4]
    tensor2 = tensor1.permute([2, 1, 0])  # size[4,3,2] 输入维度数组
    print("\ntensor1:\n", tensor1, "\nshape:\n", tensor1.shape)
    print("\ntensor2.permute([4,3,2]):\n", tensor2, "\nshape:\n", tensor2.shape)
    pass


# view和contiugous
# 一个张量经过transpose和permuute操作后，张量在内存中将不再连续存储，此时如果需要将张量转换为numpy数组或者view()或者继续后续操作，需要调用contiguous方法。
def tensor_view_contiguous():
    print("=" * 40)
    print("view和contiugous")
    tensor1 = torch.tensor([[1, 2, 3], [5, 6, 7]])
    print("\ntensor1:\n", tensor1, "\nshape:\n", tensor1.shape)
    print("\ntensor1.is_contiguous():\n", tensor1.is_contiguous())

    tensor2 = tensor1.view(3, 2)
    print("\ntensor2 = tensor1.view(3,2):\n", tensor2, "\nshape:\n", tensor2.shape)
    print("\ntensor2.is_contiguous():\n", tensor2.is_contiguous())

    # 使用transpose后，张量在内存中将不再连续存储
    tensor3 = tensor1.transpose(0, 1)
    print("\ntensor3 = tensor1.transpose(0, 1):\n", tensor3, "\nshape:\n", tensor3.shape)
    print("\ntensor3.is_contiguous():\n", tensor3.is_contiguous())

    # 先使用contiguous整理内存，再使用view
    tensor4 = tensor3.contiguous()
    print("\ntensor4 = tensor3.contiguous():\n", tensor4, "\nshape:\n", tensor4.shape)
    print("\ntensor4.is_contiguous():\n", tensor4.is_contiguous())
    tensor5 = tensor4.view(3, 2)
    print("\ntensor4.view(3,2):\n", tensor5, "\nshape:\n", tensor5.shape)

    pass


if __name__ == '__main__':
    tensor_reshape()
    tensor_squeeze_unsqueeze()
    tensor_transpose()
    tensor_permute()
    tensor_view_contiguous()

```
**结果**  
```text
========================================
reshape,保证张量数据不变，变换指定的形状

tensor1:
 tensor([[1, 2, 3, 4],
        [5, 6, 7, 8]])

tensor1.shape:
 torch.Size([2, 4])

tensor1.shape[0]:
 2

tensor1.shape[1]:
 4

tensor1.size:
 torch.Size([2, 4])

tensor1.size[0]:
 2

tensor1.size[1]:
 4

tensor1.reshape(4,2):
 tensor([[1, 2],
        [3, 4],
        [5, 6],
        [7, 8]])
========================================
升维和降维,在dim维度上新增 1个 维度

升维

tensor1:
 tensor([1, 2, 3, 4]) 
shape:
 torch.Size([4])

tensor2 = tensor1.unsqueeze(dim=0):
 tensor([[1, 2, 3, 4]]) 
shape:
 torch.Size([1, 4])

tensor3 = tensor1.unsqueeze(dim=1):
 tensor([[1],
        [2],
        [3],
        [4]]) 
shape:
 torch.Size([4, 1])

tensor4 = tensor1.unsqueeze(dim=-1):
 tensor([[1],
        [2],
        [3],
        [4]]) 
shape:
 torch.Size([4, 1])

降维不生效,移除所有大小为 1 的维度。

tensor5:
 tensor([[[ 1,  2,  3,  4],
         [ 5,  6,  7,  8]],

        [[ 9, 10, 11, 12],
         [13, 14, 15, 16]]]) 
shape:
 torch.Size([2, 2, 4])

tensor6 = tensor5.squeeze():
 tensor([[[ 1,  2,  3,  4],
         [ 5,  6,  7,  8]],

        [[ 9, 10, 11, 12],
         [13, 14, 15, 16]]]) 
shape:
 torch.Size([2, 2, 4])

降维生效,移除所有大小为 1 的维度。

tensor7:
 tensor([[[1, 2, 3, 4]],

        [[5, 6, 7, 8]]]) 
shape:
 torch.Size([2, 1, 4])

tensor8 = tensor7.squeeze():
 tensor([[1, 2, 3, 4],
        [5, 6, 7, 8]]) 
shape:
 torch.Size([2, 4])
========================================
张量维度交换transpose,只能两个

tensor1:
 tensor([[[ 1,  2,  3,  4],
         [ 5,  6,  7,  8],
         [ 5,  6,  7,  8]],

        [[ 9, 10, 11, 12],
         [13, 14, 15, 16],
         [13, 14, 15, 16]]]) 
shape:
 torch.Size([2, 3, 4])

tensor2 = tensor1.transpose(0, 1):
 tensor([[[ 1,  2,  3,  4],
         [ 9, 10, 11, 12]],

        [[ 5,  6,  7,  8],
         [13, 14, 15, 16]],

        [[ 5,  6,  7,  8],
         [13, 14, 15, 16]]]) 
shape:
 torch.Size([3, 2, 4])
========================================
张量维度交换permute

tensor1:
 tensor([[[ 1,  2,  3,  4],
         [ 5,  6,  7,  8],
         [ 5,  6,  7,  8]],

        [[ 9, 10, 11, 12],
         [13, 14, 15, 16],
         [13, 14, 15, 16]]]) 
shape:
 torch.Size([2, 3, 4])

tensor2.permute([4,3,2]):
 tensor([[[ 1,  9],
         [ 5, 13],
         [ 5, 13]],

        [[ 2, 10],
         [ 6, 14],
         [ 6, 14]],

        [[ 3, 11],
         [ 7, 15],
         [ 7, 15]],

        [[ 4, 12],
         [ 8, 16],
         [ 8, 16]]]) 
shape:
 torch.Size([4, 3, 2])
========================================
view和contiugous

tensor1:
 tensor([[1, 2, 3],
        [5, 6, 7]]) 
shape:
 torch.Size([2, 3])

tensor1.is_contiguous():
 True

tensor2 = tensor1.view(3,2):
 tensor([[1, 2],
        [3, 5],
        [6, 7]]) 
shape:
 torch.Size([3, 2])

tensor2.is_contiguous():
 True

tensor3 = tensor1.transpose(0, 1):
 tensor([[1, 5],
        [2, 6],
        [3, 7]]) 
shape:
 torch.Size([3, 2])

tensor3.is_contiguous():
 False

tensor4 = tensor3.contiguous():
 tensor([[1, 5],
        [2, 6],
        [3, 7]]) 
shape:
 torch.Size([3, 2])

tensor4.is_contiguous():
 True

tensor4.view(3,2):
 tensor([[1, 5],
        [2, 6],
        [3, 7]]) 
shape:
 torch.Size([3, 2])
```

# 5. Tensor拼接操作
```python
# torch.cat()，将多个tensor拼接成一个tensor,维度不变
import torch


def torch_cat():
    print("=" * 40)
    print("torch.cat()，将多个tensor拼接成一个tensor,维度不变")
    tensor1 = torch.randint(0, 10, (1, 2, 3))
    tensor2 = torch.randint(0, 10, (1, 2, 3))
    tensor3 = torch.cat([tensor1, tensor2], dim=0)
    tensor4 = torch.cat([tensor1, tensor2], dim=1)
    tensor5 = torch.cat([tensor1, tensor2], dim=2)
    print("\ntensor1:\n", tensor1, "\ntensor1.shape:\n", tensor1.shape)
    print("\ntensor2:\n", tensor2, "\ntensor2.shape:\n", tensor2.shape)
    print("\ntensor3 = torch.cat([tensor1, tensor2], dim=0):\n", tensor3, "\ntensor3.shape:\n", tensor3.shape)
    print("\ntensor4 = torch.cat([tensor1, tensor2], dim=1):\n", tensor4, "\ntensor4.shape:\n", tensor4.shape)
    print("\ntensor5 = torch.cat([tensor1, tensor2], dim=2):\n", tensor5, "\ntensor5.shape:\n", tensor5.shape)
    pass


# torch.stack() 可以将多个tensor拼接成一个tensor,维度会增加一维，要求tensor形状一致
def torch_stack():
    print("=" * 40)
    print("torch.stack() 可以将多个tensor拼接成一个tensor,维度会增加一维，要求tensor形状一致")
    tensor1 = torch.randint(0, 10, (1, 2, 3))
    tensor2 = torch.randint(0, 10, (1, 2, 3))
    tensor3 = torch.stack([tensor1, tensor2], dim=0)
    tensor4 = torch.stack([tensor1, tensor2], dim=1)
    tensor5 = torch.stack([tensor1, tensor2], dim=2)
    print("\ntensor1:\n", tensor1, "\ntensor1.shape:\n", tensor1.shape)
    print("\ntensor2:\n", tensor2, "\ntensor2.shape:\n", tensor2.shape)
    print("\ntensor3 = torch.stack([tensor1, tensor2], dim=0):\n", tensor3, "\ntensor3.shape:\n", tensor3.shape)
    print("\ntensor4 = torch.stack([tensor1, tensor2], dim=1):\n", tensor4, "\ntensor4.shape:\n", tensor4.shape)
    print("\ntensor5 = torch.stack([tensor1, tensor2], dim=2):\n", tensor5, "\ntensor5.shape:\n", tensor5.shape)

if __name__ == '__main__':
    torch_cat()
    torch_stack()

```

**结果**  
```text
========================================
torch.cat()，将多个tensor拼接成一个tensor,维度不变

tensor1:
 tensor([[[5, 2, 6],
         [2, 7, 6]]]) 
tensor1.shape:
 torch.Size([1, 2, 3])

tensor2:
 tensor([[[3, 1, 4],
         [1, 0, 2]]]) 
tensor2.shape:
 torch.Size([1, 2, 3])

tensor3 = torch.cat([tensor1, tensor2], dim=0):
 tensor([[[5, 2, 6],
         [2, 7, 6]],

        [[3, 1, 4],
         [1, 0, 2]]]) 
tensor3.shape:
 torch.Size([2, 2, 3])

tensor4 = torch.cat([tensor1, tensor2], dim=1):
 tensor([[[5, 2, 6],
         [2, 7, 6],
         [3, 1, 4],
         [1, 0, 2]]]) 
tensor4.shape:
 torch.Size([1, 4, 3])

tensor5 = torch.cat([tensor1, tensor2], dim=2):
 tensor([[[5, 2, 6, 3, 1, 4],
         [2, 7, 6, 1, 0, 2]]]) 
tensor5.shape:
 torch.Size([1, 2, 6])
========================================
torch.stack() 可以将多个tensor拼接成一个tensor,维度会增加一维，要求tensor形状一致

tensor1:
 tensor([[[5, 8, 6],
         [1, 9, 1]]]) 
tensor1.shape:
 torch.Size([1, 2, 3])

tensor2:
 tensor([[[7, 3, 3],
         [8, 5, 8]]]) 
tensor2.shape:
 torch.Size([1, 2, 3])

tensor3 = torch.stack([tensor1, tensor2], dim=0):
 tensor([[[[5, 8, 6],
          [1, 9, 1]]],


        [[[7, 3, 3],
          [8, 5, 8]]]]) 
tensor3.shape:
 torch.Size([2, 1, 2, 3])

tensor4 = torch.stack([tensor1, tensor2], dim=1):
 tensor([[[[5, 8, 6],
          [1, 9, 1]],

         [[7, 3, 3],
          [8, 5, 8]]]]) 
tensor4.shape:
 torch.Size([1, 2, 2, 3])

tensor5 = torch.stack([tensor1, tensor2], dim=2):
 tensor([[[[5, 8, 6],
          [7, 3, 3]],

         [[1, 9, 1],
          [8, 5, 8]]]]) 
tensor5.shape:
 torch.Size([1, 2, 2, 3])
```

# 6. 自动微分
## 6.1 原理
**first,在深度学习中，向前传播计算预测值**  
$y_{pred\_i} = X_i * W + b$  

$y_{pred\_i}$  是预测值，是一个标量   
$X_i$  是特征向量，可以是多维的(如m*n,比图像处理中)  
$W$  是权重矩阵，是一个向量，是所有  $X_i$  向量共享的   


**second,计算损失**  
以均方误差为例，  
$Loss = \frac{1}{n}\sum_{i=1}^{n}(y_{pred\_i}-y_{true\_i})^2$  

**third，计算权重梯度，并反向传播更新权重**  
损失函数对W的梯度为：  $\frac{\partial Loss}{\partial W}$  
更新后的权重：  $W_{new}  = W_{old} - \eta * \frac{\partial Loss}{\partial W}$  

偏置同理  

**依次循环多轮**  
一般情况下，在训练过程中，模型会依次循环多轮（即多个训练周期，epoch）。每一轮训练结束后，模型的预测值会逐步逼近真实值，均方误差（MSE）的损失值也会逐渐减小。随着损失值的下降，梯度的大小也会逐步变小，从而使得权重的更新幅度越来越小模型，逐渐趋于收敛。 

特殊情况，如模型太简单，欠拟合，学习率设置不合理，数据噪声很多，损失函数设计不合理等，都会影响最终的收敛   

## 6.2 pytorch-API
### 6.2.3 demo1:.backward()
**代码**  
```python
def train_step():
    # requires_grad 默认为False，不会自动计算梯度
    # 为True，自动计算梯度，并保存到grad中
    w = torch.tensor(10, requires_grad=True, dtype=torch.float32)
    print("w->", w)

    # 定义loss，表示损失函数
    loss = 2 * w ** 2
    print("loss->", loss)
    print("loss对w的导数->", loss.grad_fn)

    #
    print("loss.sum()->", loss.sum())
    loss.sum().backward()  # 这里的sum()可以省略，因为loss本身就是标量
    print("w.grad梯度->", w.grad)
    pass


if __name__ == '__main__':
    train_step()
```
**结果**  
```text
w-> tensor(10., requires_grad=True)
loss-> tensor(200., grad_fn=<MulBackward0>)
loss对w的导数-> <MulBackward0 object at 0x0000021169E99FF0>
loss.sum()-> tensor(200., grad_fn=<SumBackward0>)
w.grad梯度-> tensor(40.)
```

### 6.2.4 demo2- grad.zero_()
记得清空每一轮后的权重的梯度值， 否则会累加。  
**代码**  
```python
# 多轮收敛权重
def train_multi_step():
    print("=" * 40)
    print("多轮收敛权重")
    w = torch.tensor(10, requires_grad=True, dtype=torch.float32)
    loss = w ** 2 + 20
    print("初始时：", "w的值->", w.data, "\tloss的值->", loss.data)

    for i in range(1, 101):
        # 前向传播
        loss = w ** 2 + 20

        # 梯度清零
        if w.grad is not None:
            w.grad.zero_()  # 清空权重

        # 反向传播
        loss.backward()

        # 更新权重
        w.data = w.data - 0.01 * w.grad
        print("w的值->", w.data, "\tw的梯度->", w.grad)

    # 最终结果
    print("w最终值->", w.data, "\tloss最终值->", loss, "\tw的梯度->", w.grad)
    pass
```

**结果**  
```text
========================================
多轮收敛权重
初始时： w的值-> tensor(10.) 	loss的值-> tensor(120.)
w的值-> tensor(9.8000) 	w的梯度-> tensor(20.)
w的值-> tensor(9.6040) 	w的梯度-> tensor(19.6000)
w的值-> tensor(9.4119) 	w的梯度-> tensor(19.2080)
w的值-> tensor(9.2237) 	w的梯度-> tensor(18.8238)
...
w的值-> tensor(1.3809) 	w的梯度-> tensor(2.8181)
w的值-> tensor(1.3533) 	w的梯度-> tensor(2.7618)
w的值-> tensor(1.3262) 	w的梯度-> tensor(2.7065)
w最终值-> tensor(1.3262) 	loss最终值-> tensor(21.8313, grad_fn=<AddBackward0>) 	w的梯度-> tensor(2.7065)

```

### 6.2.5 demo3- detach
```python
# 对一个张量（Tensor）设置了 requires_grad=True，那么这个张量就与计算图关联，PyTorch 会记录它的计算过程，以便进行反向传播。
# 但当你想把这个张量转换为 NumPy 数组（numpy.ndarray）时，就会遇到问题，因为： NumPy 不支持带有梯度信息的张量。

# detach ,解决一个张量设置自动微分(requires_grad=True)后，无法转为numpy的nparray的问题
# 从计算图中“分离”张量，即不再记录它的梯度信息。
# 分离后的张量是一个普通的张量，不再参与反向传播。
# 分离后的张量可以安全地转换为 NumPy 数组。
def tensor_detach():
    print("=" * 40)
    print("tensor detach")

    tensor1 = torch.tensor([10, 20], requires_grad=True, dtype=torch.float32)
    print("\nbefore detach,tensor1->", tensor1, "\ntensor1.type->", type(tensor1))

    # RuntimeError: Can't call numpy() on Tensor that requires grad. Use tensor.detach().numpy() instead.
    # numpy1 = tensor1.numpy()

    # 不会改变tensor1的类型，tensor1依旧可以自动微分，但是底层数据和tensor2是共享的
    tensor2 = tensor1.detach()
    tensor2.data[0] = 100
    numpy2 = tensor2.numpy()
    print("\nafter detach,tensor1->", tensor1, "\ntensor1.type->", type(tensor1))
    print("\ntensor2->", tensor2, "\ntensor2.type->", type(tensor2))
    print("\nnumpy2->", numpy2, "\nnumpy2.type->", type(numpy2))
    pass
```
**结果**  
```text
========================================
tensor detach

before detach,tensor1-> tensor([10., 20.], requires_grad=True) 
tensor1.type-> <class 'torch.Tensor'>

after detach,tensor1-> tensor([100.,  20.], requires_grad=True) 
tensor1.type-> <class 'torch.Tensor'>

tensor2-> tensor([100.,  20.]) 
tensor2.type-> <class 'torch.Tensor'>

numpy2-> [100.  20.] 
numpy2.type-> <class 'numpy.ndarray'>
```
### demo4
```python
def demo():
    # 特征向量
    x = torch.ones(2, 5)
    # 真实值
    y_true = torch.zeros(2, 3)

    # 初始化权重和偏置
    w = torch.randn(5, 3, requires_grad=True)
    b = torch.randn(3, requires_grad=True)
    print("\nbefore:\nw的值：", w, "\nw的梯度：", w.grad)
    print("\nbefore:\nb的值：", b, "\nb的梯度：", b.grad)

    # 设置网络的预测值
    y_pred = torch.matmul(x, w) + b
    # 设置损失函数
    loss = torch.nn.MSELoss()
    loss = loss(y_pred, y_true)

    # 自动微分
    loss.backward()
    print("\nafter: \nw的值：", w, "\nw的梯度：", w.grad)
    print("\nafter \nb的值：", b, "\nb的梯度：", b.grad)

    # 后续继续更新w和b
    # 若手写，则w新 = w旧 - 学习率*w梯度
    pass
```
**结果**  
```text
before:
w的值： tensor([[ 1.4458, -0.6366, -0.2436],
        [ 1.8789,  0.2167,  0.7270],
        [-1.1901, -0.6494, -1.7013],
        [ 0.0775,  0.0111,  0.1468],
        [ 0.1479,  0.3075, -0.2417]], requires_grad=True) 
w的梯度： None

before:
b的值： tensor([-0.5552, -1.0517, -0.9311], requires_grad=True) 
b的梯度： None

after: 
w的值： tensor([[ 1.4458, -0.6366, -0.2436],
        [ 1.8789,  0.2167,  0.7270],
        [-1.1901, -0.6494, -1.7013],
        [ 0.0775,  0.0111,  0.1468],
        [ 0.1479,  0.3075, -0.2417]], requires_grad=True) 
w的梯度： tensor([[ 1.2032, -1.2017, -1.4960],
        [ 1.2032, -1.2017, -1.4960],
        [ 1.2032, -1.2017, -1.4960],
        [ 1.2032, -1.2017, -1.4960],
        [ 1.2032, -1.2017, -1.4960]])

after 
b的值： tensor([-0.5552, -1.0517, -0.9311], requires_grad=True) 
b的梯度： tensor([ 1.2032, -1.2017, -1.4960])
```

# 7. pytorch-案例-训练线性回归模型
```python
# torch.nn.MSELoss : 损失函数
# torch.nn.Linear  : 假设函数  类似x*w+b
# torch.optim.SGD  : 优化器，来更新权重和偏置

# 为什么需要 batch_size？
# 内存限制：如果数据集很大，一次性加载所有数据可能会超出内存限制。
# 训练效率：使用 batch_size 可以利用 GPU 的并行计算能力，提高训练速度。
# 梯度更新频率：batch_size 越小，梯度更新越频繁，训练过程越“抖动”；batch_size 越大，训练越稳定，但需要更多内存。
import matplotlib.pyplot as plt
import torch
from sklearn.datasets import make_regression
from torch.utils.data import TensorDataset, DataLoader


# 生成数据集
def create_dataset():
    x, y, coef = make_regression(n_samples=100, n_features=1, noise=10, coef=True, bias=14.5, random_state=12)

    # 转换为张量
    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)

    plt.scatter(x, y)
    x1 = torch.linspace(x.min(), x.max(), 1000)
    y1 = torch.tensor([v * coef + 14.5 for v in x1])
    plt.plot(x1, y1, label='real', color='red')
    plt.legend()
    plt.grid()
    plt.show()

    return x, y, coef
    pass


# 模型训练
def train_model(x, y, coef):
    # 1. 创建数据集
    dataset = TensorDataset(x, y)

    # 2. 封装数据集对象
    # 训练集打乱，测试集不打乱
    dataloader = DataLoader(dataset, batch_size=10, shuffle=True)

    # 3.初始一个线性回归模型
    model = torch.nn.Linear(1, 1)

    # 4.创建损失函数对象
    criterion = torch.nn.MSELoss()

    # 5. 创建优化器对象
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # 6. 开始训练
    # 6.1 定义训练轮数，每轮的平均损失
    epochs, loss_list = 100, []
    # 6.2 开始训练
    for epoch in range(epochs):
        # 6.2.1 定义每轮的训练总损失，训练的批次数
        total_loss, total_samples = 0.0, 0
        # 6.2.2 每一轮的训练,共10 轮
        for batch_x, batch_y_true in dataloader:
            # 每一个batch的训练, 共100个样本，batch_size为10，一共10 批
            # 1） 模型预测
            batch_y_pred = model(batch_x)
            # 2） 计算当前批的平均损失
            loss = criterion(batch_y_pred, batch_y_true.reshape(-1, 1))
            # 3） 损失相加
            total_loss += loss.item()
            # 4）训练样本 批数 +1
            total_samples += 1
            # 5） 梯度归零
            optimizer.zero_grad()
            # 6） 反向传播，计算梯度
            loss.backward()
            # 7） 更新参数
            optimizer.step()
        # 6.2.3 计算当前轮的平均损失
        epoch_loss = total_loss / total_samples
        loss_list.append(epoch_loss)
        print(f'Epoch [{epoch + 1}/{epochs}], mean Loss: {epoch_loss:.4f}')

    # 7 最终结果
    print('权重:', model.weight.item(), '偏置:', model.bias.item())

    # 8 绘制损失曲线
    plt.plot(range(epochs), loss_list)
    plt.title('loss')
    plt.grid()
    plt.show()
    pass


if __name__ == '__main__':
    x, y, coef = create_dataset()
    train_model(x, y, coef)
    pass
```

**结果**  

```text
Epoch [1/100], mean Loss: 1427.9020
Epoch [2/100], mean Loss: 977.7131
Epoch [3/100], mean Loss: 682.6126
...
Epoch [97/100], mean Loss: 100.2006
Epoch [98/100], mean Loss: 100.2468
Epoch [99/100], mean Loss: 100.3301
Epoch [100/100], mean Loss: 100.1055
权重: 37.02631759643555 偏置: 13.730884552001953
```
