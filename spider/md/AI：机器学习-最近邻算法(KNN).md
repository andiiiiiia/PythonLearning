# 1. 算法简介
**无监督最近邻**是许多其他学习方法的基础，特别是流形学习(manifold learning)和光谱聚类(spectral clustering)。  
有监督的 neighbors-based (基于邻居的) 学习有两种方式：离散标签数据的分类和连续标签数据的回归。  

## 1.2 基本算法思想：
找到在距离上离新样本最近的一些样本， 并且从这些样本中预测标签。  
最近邻的样本数可以是用户定义的常数(k-最近邻)，也可以根据不同的点的局部密度(基于半径的近邻学习)确定。  

### 1.2.1 距离计算
**欧式距离**：  
维度差值平方和 开根号  
如：d = √((x1-x2)²+(y1-y2)²+(z1-z2)²)    

**曼哈顿距离**：  
维度差值的绝对值求和  
如：d = |x1-x2|+|y1-y2|+|z1-z2|  

**切比雪夫距离**  
max(对应维度差值的绝对值)  
如：max(|x1-x2|,|y1-y2|,|z1-z2|)   
使用场景：国际象棋，国王八方向，从A点到B点最短距离   

**闵式距离**  
一个总结公式： 
$d_{12} = \sqrt[p]{\sum_{k=1}^n{|x_{1k}-x_{2k}|}^p}$

p=1:  曼哈顿
p=2： 欧式
p->∞： 切比雪夫

**等等**   

### 1.2.2 权重
基本的最近邻算法使用一样的**权重**：也就是说，分配给查询点的值是根据最近邻居的简单多数投票计算的。   
在某些情况下，最好是对邻居进行加权，这样更靠近的邻居对拟合来说贡献更大或者更小或者其他。  

### 1.2.3 K值选择
K太小：模型容易受到噪声影响，预测结果不稳定。过拟合。  
K太大：模型会变得过于平滑，可能忽略局部特征。欠拟合。  
通常使用 **交叉验证**（Cross Validation） 来选择最优的K值。  

## 1.3 KNN分类算法思想
### 1.3.1 算法思想
1.计算未知样本到每一个训练样本的距离  
2.将训练样本根据距离大小升序排列  
3.取出距离最近的K个训练样本  
4.进行多数表决,统计K个样本中哪个类别的样本个数最多的  
5.将未知的样本归属到出现次数最多的类别  

## 1.4 KNN回归算法思想
### 1.4.1 算法思想
1.计算未知样本到每一个训练样本的距离  
2.将训练样本根据距离大小升序排列  
3.取出距离最近的 K 个训练样本  
4.把这个 K 个样本的目标值计算其平均值  
5.作为将未知的样本预测的值    

# 2. sklearn应用 

## 2.1 Knn分类实现

### 2.1.1 描述
scikit-learn实现了两个不同的最近邻分类器：   
KNeighborsClassifier 分类器根据每个查询点的k个最近邻实现学习，其中k是用户指定的整数值。  
RadiusNeighborsClassifier分类器根据每个训练点的固定半径内的邻居数实现学习，其中  是用户指定的浮点值。 
### 2.1.2 实现流程

获取数据集  
数据基本处理  
数据集预处理  
机器学习  
模型评估  
模型预测  

### 2.1.3 测试1
**测试用例**  
```python
from sklearn.neighbors import KNeighborsClassifier
x_train = [[0],[1],[2],[3]]
y_train = [0,0,1,1]
x_test= [[5]]
estimator = KNeighborsClassifier(n_neighbors=2)
#训练
estimator.fit(x_train,y_train)
# 预测
y_pred = estimator.predict(x_test)

print(y_pred)
```
**测试结果**  
```text
[1]
```

### 2.1.4 测试2：鸢尾花分类
**测试用例**
```python
from sklearn.datasets import load_iris
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


def load_dataset_iris():
    dataset = load_iris()
    print("数据集中所有的键：\n", dataset.keys())
    print("数据集中前5行数据：\n", dataset.data[:5])
    print("数据集中数据的特征名称：\n", dataset.feature_names)
    print("数据集中前5行数据的标签：\n", dataset.target[:5])
    print("数据集中的标签名称：\n", dataset.target_names)


def show_iris():
    dataset = load_iris()
    # 构造df
    iris_df = pd.DataFrame(dataset['data'], columns=dataset.feature_names)
    iris_df['ResultType'] = dataset.target
    print(iris_df)
    # 绘制
    x_name = 'sepal width (cm)'
    y_name = 'sepal length (cm)'
    # fit_reg:拟合线
    sns.lmplot(x=x_name, y=y_name, data=iris_df, hue='ResultType', fit_reg=True)
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.title("iris")
    plt.tight_layout()
    plt.show()


def traintest_split():
    dataset = load_iris()
    # random_state: int or RandomState instance, default = None 。在应用切分之前，控制应用于数据的无序处理。为多个函数调用传递可重复输出的int值。
    # shuffle: bool, default = True 。切分前是否对数据进行打乱。如果shuffle = False，则stratify必须为None。
    x_train, x_test, y_train, y_test = train_test_split(
        dataset.data,
        dataset.target,
        test_size=0.3,
        random_state=22,
        shuffle=True
    )
    print("x_train前5条:\n", x_train[:5])
    print("x_train条数:\n", len(x_train))
    print("y_train前5条:\n", y_train[:5])
    print("y_train条数:\n", len(y_train))

    print("x_test前5条:\n", x_test[:5])
    print("x_test长度:\n", len(x_test))
    print("y_test前5条:\n", y_test[:5])
    print("y_test长度:\n", len(y_test))

def iris_evaluate_test():
    # 1. 加载数据集
    iris_dataset = load_iris()

    # 2. 数据预处理
    x_train, x_test, y_train, y_test = train_test_split(
        iris_dataset.data,
        iris_dataset.target,
        test_size=0.3,
        random_state=23
    )
    print("测试集的target：\n", y_test)

    # 3. 特征工程
    # 3.1 特征提取： 略
    # 3.2 特征预处理：
    scaler = StandardScaler()
    # 部分数据先拟合fit，找到该part的整体指标，如均值、方差、最大值最小值等等（根据具体转换的目的），然后对该trainData进行转换transform，从而实现数据的标准化、归一化等等。
    x_train = scaler.fit_transform(x_train)
    # 在fit的基础上，进行标准化，降维，归一化等操作
    x_test = scaler.transform(x_test)

    # 4. 模型训练
    # 4.1 创建模型
    classifier = KNeighborsClassifier(n_neighbors=5, weights='uniform')
    # 4.2 训练模型
    classifier.fit(x_train, y_train)

    # 5. 模型预测
    # 5.1 对30条测试集进行预测
    y_predict = classifier.predict(x_test)
    print("测试集的预测target：\n", y_predict)
    # 5.2 对新数据进行测试
    x_my_data = [[6.6, 3.3, 5.5, 1.1]]
    x_my_data = scaler.transform(x_my_data)
    y_my_data_predict = classifier.predict(x_my_data)
    print("自定义数据集的预测target:\n", y_my_data_predict)
    # 5.3 查看上述数据集，每种分类的概率
    y_my_data_predict_proba = classifier.predict_proba(x_my_data)
    print("自定义数据集的预测target的概率:\n", y_my_data_predict_proba)

    # 6. 模型评估
    # 方式1：基于 训练集的特征 和 训练集的标签   进行评分
    print("classifier.score 训练集 正确率(准确率):\n", classifier.score(x_train, y_train))
    # 方式1：基于 测试集的特征 和 测试集的标签   进行评分
    print("classifier.score 测试集 正确率(准确率):\n", classifier.score(x_test, y_test))
    # 方式3：基于 测试集的标签 和 预测结果      进行评分
    print("accuracy_score 正确率(准确率):\n", accuracy_score(y_test, y_predict))


if __name__ == '__main__':
    # load_dataset_iris()
    # show_iris()
    # traintest_split()
    iris_evaluate_test()

```
**测试结果：load_dataset_iris()**
```text
数据集中所有的键：
 dict_keys(['data', 'target', 'frame', 'target_names', 'DESCR', 'feature_names', 'filename', 'data_module'])
数据集中前5行数据：
 [[5.1 3.5 1.4 0.2]
 [4.9 3.  1.4 0.2]
 [4.7 3.2 1.3 0.2]
 [4.6 3.1 1.5 0.2]
 [5.  3.6 1.4 0.2]]
数据集中数据的特征名称：
 ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
数据集中前5行数据的标签：
 [0 0 0 0 0]
数据集中的标签名称：
 ['setosa' 'versicolor' 'virginica']
```
**测试结果：show_iris()**
```text
     sepal length (cm)  sepal width (cm)  ...  petal width (cm)  ResultType
0                  5.1               3.5  ...               0.2           0
1                  4.9               3.0  ...               0.2           0
2                  4.7               3.2  ...               0.2           0
3                  4.6               3.1  ...               0.2           0
4                  5.0               3.6  ...               0.2           0
..                 ...               ...  ...               ...         ...
145                6.7               3.0  ...               2.3           2
146                6.3               2.5  ...               1.9           2
147                6.5               3.0  ...               2.0           2
148                6.2               3.4  ...               2.3           2
149                5.9               3.0  ...               1.8           2
```

**测试结果：traintest_split()**
```text
x_train前5条:
 [[5.6 3.  4.1 1.3]
 [5.7 2.5 5.  2. ]
 [6.5 3.  5.8 2.2]
 [5.  3.6 1.4 0.2]
 [6.1 2.8 4.  1.3]]
x_train条数:
 105
y_train前5条:
 [1 2 2 0 1]
y_train条数:
 105
x_test前5条:
 [[5.4 3.7 1.5 0.2]
 [6.4 3.2 5.3 2.3]
 [6.5 2.8 4.6 1.5]
 [6.3 2.5 5.  1.9]
 [6.1 2.9 4.7 1.4]]
x_test长度:
 45
y_test前5条:
 [0 2 1 2 1]
y_test长度:
 45
```
**测试结果：iris_evaluate_test**
```text
测试集的target：
 [2 2 1 0 2 1 0 2 0 1 1 0 2 0 0 2 1 1 2 0 2 0 0 0 2 0 0 2 1 1 0 1 0 2 0 0 1
 1 1 2 2 0 1 0 1]
测试集的预测target：
 [2 2 1 0 2 1 0 2 0 1 1 0 2 0 0 1 1 1 2 0 2 0 0 0 2 0 0 2 1 1 0 1 0 2 0 0 1
 1 1 2 2 0 1 0 1]
自定义数据集的预测target:
 [1]
自定义数据集的预测target的概率:
 [[0. 1. 0.]]
classifier.score 训练集 正确率(准确率):
 0.9523809523809523
classifier.score 测试集 正确率(准确率):
 0.9777777777777777
accuracy_score 正确率(准确率):
 0.9777777777777777
```

### 2.1.5 测试3：鸢尾花分类-交叉验证与网格化搜索
**测试代码**
```python
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler


def iris_evaluate_test():
    # 1. 加载数据集
    iris_dataset = load_iris()

    # 2. 数据预处理
    x_train, x_test, y_train, y_test = train_test_split(
        iris_dataset.data,
        iris_dataset.target,
        test_size=0.3,
        random_state=23
    )
    print("测试集的target：\n", y_test)

    # 3. 特征工程
    # 3.1 特征提取： 略
    # 3.2 特征预处理：
    scaler = StandardScaler()
    # 部分数据先拟合fit，找到该part的整体指标，如均值、方差、最大值最小值等等（根据具体转换的目的），然后对该trainData进行转换transform，从而实现数据的标准化、归一化等等。
    x_train = scaler.fit_transform(x_train)
    # 在fit的基础上，进行标准化，降维，归一化等操作
    x_test = scaler.transform(x_test)

    # 4. 模型训练
    # 4.1 创建模型
    model = KNeighborsClassifier(weights='uniform')
    # 4.2 定义网格化参数
    param_grid = {'n_neighbors': [1, 3, 5, 7]}
    model = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)
    # 4.3 训练模型
    model.fit(x_train, y_train)
    # 4.4 交叉验证网格化搜索结果查看
    print("model.best_score_:\n", model.best_score_)
    print("model.best_estimator_:\n", model.best_estimator_)
    print("model.best_params_:\n", model.best_params_)
    print("model.cv_results_:\n", model.cv_results_)
    # 4.5 保存交叉验证结果集
    my_cv_result = pd.DataFrame(model.cv_results_)
    my_cv_result.to_csv(path_or_buf='./my_cv_result.csv')


if __name__ == '__main__':
    iris_evaluate_test()

```
**测试结果**
```text
测试集的target：
 [2 2 1 0 2 1 0 2 0 1 1 0 2 0 0 2 1 1 2 0 2 0 0 0 2 0 0 2 1 1 0 1 0 2 0 0 1
 1 1 2 2 0 1 0 1]
 
model.best_score_:
 0.9619047619047618

model.best_estimator_:
 KNeighborsClassifier(n_neighbors=7)

model.best_params_:
 {'n_neighbors': 7}

model.cv_results_:
{
	'mean_fit_time': array([0.00079746, 0.00019941, 0.        , 0.00019941]), 
	'std_fit_time': array([0.00074596, 0.00039883, 0.        , 0.00039883]), 
	'mean_score_time': array([0.00179558, 0.0010087 , 0.00099659, 0.002213  ]), 
	'std_score_time': array([7.48598976e-04, 2.25919186e-05, 2.61174468e-07, 9.84770248e-04]), 
	'param_n_neighbors': masked_array(data=[1, 3, 5, 7],mask=[False, False, False, False],fill_value=999999), 
	'params': [{'n_neighbors': 1}, {'n_neighbors': 3}, {'n_neighbors': 5}, {'n_neighbors': 7}], 
	'split0_test_score': array([0.9047619 , 0.85714286, 0.9047619 , 0.95238095]), 
	'split1_test_score': array([0.9047619 , 0.9047619 , 0.9047619 , 0.95238095]), 
	'split2_test_score': array([0.9047619 , 0.9047619 , 0.95238095, 0.95238095]), 
	'split3_test_score': array([1.        , 0.95238095, 0.95238095, 1.        ]), 
	'split4_test_score': array([0.95238095, 0.95238095, 0.95238095, 0.95238095]), 
	'mean_test_score': array([0.93333333, 0.91428571, 0.93333333, 0.96190476]), 
	'std_test_score': array([0.03809524, 0.03563483, 0.02332847, 0.01904762]), 
	'rank_test_score': array([2, 4, 2, 1])
}
```
### 2.1.6 测试4：手写数字识别
**测试代码**
```python
import joblib
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
import numpy as np


def show_digit(index):
    # 1. 加载数据集
    df_digits = pd.read_csv("./手写数字识别.csv")
    if index < 0 or index > len(df_digits) - 1:
        return
    # 2. 打印基本信息
    # Index(['label', 'pixel0', 'pixel1', 'pixel783'],dtype='object', length=785)
    print("df_digits.keys:\n", df_digits.keys())
    x = df_digits.iloc[:, 1:]
    y = df_digits.iloc[:, 0]
    print("前10个数字的label:\n", y[:10])
    print("前10个数字的features:\n", x[:10])
    print("当前数字的标签:\n", y[index])

    # 3. 显示图片
    # reshape x[index]为28*28
    x_index_reshape = x.iloc[index].values.reshape(28, 28)
    print("x_index_reshape:\n", x_index_reshape)
    plt.imshow(x_index_reshape, cmap='gray')
    plt.show()


def train_model():
    # 1. 加载数据
    data_digits = pd.read_csv(".\手写数字识别.csv")
    x = data_digits.iloc[:, 1:]
    y = data_digits.iloc[:, 0]

    # 2. 特征工程：
    # 特征提取
    # 特征预处理（标准化，归一化）
    # x = x / 255
    x = x.div(x.max(axis=1), axis=0)

    # 3. 数据预处理：
    # 划分测试集和数据集
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, stratify=y, random_state=0)

    # 4. 模型训练
    # 创建模型
    model = KNeighborsClassifier()
    grid_param = {'n_neighbors': [1, 3, 5, 7, 9]}
    model = GridSearchCV(estimator=model, param_grid=grid_param, cv=5)
    # 训练模型
    model.fit(x_train, y_train)
    print("model.best_estimator_:\n", model.best_estimator_)
    model = model.best_estimator_
    model.fit(x_train, y_train)

    # 5. 模型预测
    y_predict = model.predict(x_test)

    # 6. 模型评估
    accuracy = accuracy_score(y_test, y_predict)
    print("accuracy:\n", accuracy)

    # 7. 模型保存
    joblib.dump(model, ".\\model\\digits.pth")
    # return


def use_model_predict(path):
    # 1. 加载模型
    model = joblib.load(".\\model\\digits.pth")

    # 2. 读取图片，并转换
    img = plt.imread(path)
    # 转单通道
    if img.ndim == 3 and img.shape[2] == 3:
        # 使用加权平均法将RGB转为灰度
        img = np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])
    else:
        print("图像不是3通道，无法转换。")
    # 归一化
    img = (img - img.min()) / (img.max() - img.min())
    # 模型输入格式
    img = img.reshape(1, -1)

    # 3. 预测图片数字
    y_predict = model.predict(img)
    print("y_predict:\n",y_predict)


if __name__ == '__main__':
    # show_digit(9)
    # train_model()
    use_model_predict('digit_3.png')

```
**测试结果:show_digit**
```text
df_digits.keys:
 Index(['label', 'pixel0', 'pixel1', 'pixel2', 'pixel3', 'pixel4', 'pixel5',
       'pixel6', 'pixel7', 'pixel8',
       ...
       'pixel774', 'pixel775', 'pixel776', 'pixel777', 'pixel778', 'pixel779',
       'pixel780', 'pixel781', 'pixel782', 'pixel783'],
      dtype='object', length=785)

前10个数字的label:
 0    1
1    0
2    1
3    4
4    0
5    0
6    7
7    3
8    5
9    3
Name: label, dtype: int64

前10个数字的features:
    pixel0  pixel1  pixel2  pixel3  ...  pixel780  pixel781  pixel782  pixel783
0       0       0       0       0  ...         0         0         0         0
1       0       0       0       0  ...         0         0         0         0
2       0       0       0       0  ...         0         0         0         0
3       0       0       0       0  ...         0         0         0         0
4       0       0       0       0  ...         0         0         0         0
5       0       0       0       0  ...         0         0         0         0
6       0       0       0       0  ...         0         0         0         0
7       0       0       0       0  ...         0         0         0         0
8       0       0       0       0  ...         0         0         0         0
9       0       0       0       0  ...         0         0         0         0
[10 rows x 784 columns]


当前数字的标签:
 3

x_index_reshape:
[
[  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0]
[  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0]
[  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0]
[  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0]
[  0   0   0   0   0   0   0   5  60 136 136 147 254 255 199 111  18   9   0   0   0   0   0   0   0   0   0   0]
[  0   0   0   0   0   0  25 152 253 253 253 253 253 253 253 253 253 124   0   0   0   0   0   0   0   0   0   0]
[  0   0   0   0   0   0 135 225 244 253 202 200 181 164 216 253 253 211   151   0   0   0   0   0   0   0   0   0]
[  0   0   0   0   0   0   0  30 149  78   3   0   0   0  20 134 253 253   224   0   0   0   0   0   0   0   0   0]
[  0   0   0   0   0   0   0   0   0   0   0   0   0   0  28 206 253 253   224   0   0   0   0   0   0   0   0   0]
[  0   0   0   0   0   0   0   0   0   0   0   0   0   0  78 253 253 253   224   0   0   0   0   0   0   0   0   0]
[  0   0   0   0   0   0   0   0   0   0   0   0   0   5  99 234 253 253   224   0   0   0   0   0   0   0   0   0]
[  0   0   0   0   0   0   0   0   0   0   0  14 142 220 219 236 253 253   240 121   7   0   0   0   0   0   0   0]
[  0   0   0   0   0   0   0   0   0   0   0  24 253 253 253 253 235 233   253 253 185  53   0   0   0   0   0   0]
[  0   0   0   0   0   0   0   0   0   0   0   8 150 194 194 194  53  40   97 253 253 170   0   0   0   0   0   0]
[  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   122 253 253 170   0   0   0   0   0   0]
[  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0  55   237 253 253 170   0   0   0   0   0   0]
[  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0 130   253 253 253 170   0   0   0   0   0   0]
[  0   0   0   0   0   0   0   0   0   0   0   0   0   0   4  12 120 193   253 253 214  28   0   0   0   0   0   0]
[  0   0   0   0   0   0   0   0   0   0   0   0   0   7 153 253 253 253   253 212  30   0   0   0   0   0   0   0]
[  0   0   0   0   0   0  33 136  70   6   0  27  67 186 253 253 253 253   234  31   0   0   0   0   0   0   0   0]
[  0   0   0   0   0  26 231 253 253 191 183 223 253 253 253 253 172 216   112   0   0   0   0   0   0   0   0   0]
[  0   0   0   0   0  36 215 253 253 253 253 253 253 253 253 253  47  25   0   0   0   0   0   0   0   0   0   0]
[  0   0   0   0   0   5  87 223 253 253 253 244 152 223 223 109   4   0   0   0   0   0   0   0   0   0   0   0]
[  0   0   0   0   0   0  67  50 176 148  78  16   0  12  12   0   0   0   0   0   0   0   0   0   0   0   0   0]
[  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0]
[  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0]
[  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0]
[  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0]
```

**测试结果：train_model**
```text
model.best_estimator_:
 KNeighborsClassifier(n_neighbors=3)

accuracy:
 0.965
```

**测试结果：use_model_predict**
```text
y_predict:
[3]
```
## 2.2 Knn回归实现

### 2.2.1 描述
scikit-learn实现了两个不同的近邻回归器：  
KNeighborsRegressor实现基于每个查询点的k个最近邻的学习，其中k是用户指定的整数值。   
RadiusNeighborsRegressor实现了基于查询点的固定半径r内的近邻的学习，其中r是用户指定的浮点数值。

### 2.2.2 测试1
**测试用例**  
```python
from sklearn.neighbors import KNeighborsRegressor

x_train = [[0, 0, 1], [1, 1, 0], [3, 5, 6], [7, 6, 8]]
y_train = [0.2, 0.1, 0.3, 0.4]

x_test = [[5, 7, 6]]

# 初始化
regressor = KNeighborsRegressor(n_neighbors=3)
# 训练
regressor.fit(x_train,y_train)
# 预测
y_pre  = regressor.predict(x_test)
# 打印
print(y_pre)
```
**测试结果**  
```text
[0.26666667]
```
