# 1. 逻辑回归简介
## 1.1 应用场景
二分类  
预测阳性  
是否放贷  
情感分析：正向&负向  
## 1.2 数学基础
### 1.2.1 sigmoid激活函数(S型函数)：  
公式：  
$f(x) = \frac{1}{1+e^{-x}}$  

作用：将 $(-\\infty,\infty)$ 映射到(0,1)上，划分正样本和负样本    
  
数学性质：单调递增函数，拐点在(0,0.5)的位置  

导函数公式：  
$f'(x) = f(x)(1-f(x))$   

f(x) > 0.5 : 正样本  
f(x) < 0.5 : 负样本  

### 1.2.2 概率
**概率：**  
事件发生的可能性  
eg: 北京早上堵车的可能性 $P_A$ = 0.7。   中午堵车的可能性 $P_B$ = 0.3。	   晚上堵车的可能性 $P_C$=0.4   

**联合概率：**  
指两个或多个随机变量同时发生的概率   
eg:周一早上和周二早上同时堵车的概率为：  
$P_AP_B = 0.7*0.7 = 0.49$   

**条件概率：**  
表示事件A在另外一个事件B已经发生条件下的发生概率  
eg：周一早上堵车的情况下，中午再堵车的概率为：  
$P_{B|A} = 0.7*0.3 = 0.21$  

### 1.2.3 极大似然估计
**思想：**  
在已知样本数据(观测到的结果)的前提下，寻找最可能产生这些数据的模型参数。
假设我们有一个概率模型，它由一个或多个未知参数组成，比如参数为θ。我们观察到了一组样本数据$x_1,x_2,x_3...x_n$,这些数据是根据某个概率分布（比如正态分布、泊松分布等）生成的。  

**极大似然估计的目标**：  
找到一个参数θ，使得**这组样本数据出现的概率**最大。  

**似然函数**：  
即这组样本数据出现的概率。  
$L(θ) = P(x_1,x_2,...x_n|θ) = \prod_{i=1}^{n}P(x_i|θ)$  

**例子：正态分布**  
假设一组样本数据  $x_1,x_2,...x_n$  服从正态分布  $N(μ,\sigma^2)$  ，其中μ是均值，$\sigma^2$是方差，两参数未知。  
假设概率密度函数为：  
$P(x_i|μ,\sigma^2) = \frac{1}{\sqrt{2π\sigma^2}}e^{-\frac{(x_i-μ)^2}{2\sigma^2}}$  

似然函数：  
$L(μ,\sigma^2) = \prod_{i=1}^{n}\frac{1}{\sqrt{2π\sigma^2}}e^{-\frac{(x_i-μ)^2}{2\sigma^2}}$  
    
对数似然函数：  
$ln(L(μ,\sigma^2)) = -\frac{n}{2}ln(2π\sigma^2)-\frac{1}{2\sigma^2}\sum_{i=1}^{n}(x_i-μ)^2$  
  
对μ求导：  
$μ = \frac{1}{n}\sum_{i=1}^{n}x_i$  
   
对  $\sigma$  求导：  
$\sigma^2 = \frac{1}{n}\sum_{i=1}^{n}(x_i-μ)^2$  

**例子：**  
问：假设有一枚不均匀的硬币,出现正面的概率和反面的概率是不同的。假定出现正面的概率为θ，抛了6次得到如下现象D={正面,反面,反面,正面,正面,正面}。每次投掷事件都是相互独立的。则根据产生的现象D,来估计参数θ是多少?  
解：  
$L(θ) = θ^4(1-θ)^2$   
$L^{'}(θ) = 4θ^3(1-θ)^2-θ^42(1-θ) = θ^3(1-θ)(4-6θ)$   
求解可得当θ=2/3时，取最大概率。


# 2. 逻辑回归原理
## 2.1 逻辑回归概念 Logistic Regression
一种分类模型,把线性回归的输出,作为逻辑回归的输入。  
输出是(0,1)之间的值  
## 2.2 基本思想
1. 利用线性模型 $f(x)=w^Tx+b$根据特征的重要性计算出一个值  
2. 再使用 sigmoid 函数将 f(x)的输出值映射为概率值  
    设置阚值(eg: 0.5),输出概率值大于 0.5,则将未知样本输出为 1 类  
    否则输出为 0 类  
## 2.3 逻辑回归的激活函数
$h(w) = sigmoid(w^Tx + b )$  
线性回归的输出,作为逻辑回归的输入  
## 2.4 逻辑回归计算案例
## 2.5 损失函数
### 2.5.1 损失函数的基本原理
$Loss(L) = -\sum_{i=1}^{n}(y_ilog(p_i) + (1-y_i)log(1-p_i))$   
y:逻辑回归预估的类别，A->1  B-> 0  
$p_i = sigmoid(w^T + b)$是逻辑回归的输出结果  
原理：每个样本预测值有A、B两个类别，真实类别对应的位置，概率值越大越好。  
### 2.5.2 损失函数和极大似然函数的关系
**一个样本的概率**  
假设，有0,1两个类别，样本为1类别的概率为p。  
则 概率  $L = p^y(1-p)^{1-y}$  

**多个样本的概率**  
假设，有0,1两个类别，样本特征和类别为(x1,y1),(x2,y2),...(xn,yn)。    
则 所有样本都预测正确的概率  $P = \prod_{i=1}^{n}p^{y_i}(1-p)^{1-y_i}$   

**关系**  
对所有样本概率取对数：  $log(P) = \sum_{i=1}^{n}(y_ilog(p_i) + (1-y_i)log(1-p_i))$  

所以，**Loss = -log(P)**  

使用梯度下降优化算法，更新逻辑回归算法的权重参数  
# 3. 逻辑回归分类问题评估
## 3.1 混淆矩阵
| |预测值(正例) | 预测值(反例)|
| - | - | - |
|真实值(正例) |真正例(TP) | 伪反例(FN)|
| 真实值(反例)|伪正例(FP) | 真反例(TN)|

### 3.1.1 案例
样本有10个样本，其中6个恶性肿瘤，4个良性肿瘤，假设恶性肿瘤为正例。  
模型A:预测对了3个恶性肿瘤，4个良性肿瘤，则TP,FP,TN,FN分别为什么？  
解：  
| |预测值(正例) | 预测值(反例)|
| - | - | - |
|真实值(正例) |真正例(TP):3 | 伪反例(FN):3|
| 真实值(反例)|伪正例(FP):0 | 真反例(TN):4|

## 3.2 精确率
tp/(tp+fp)  
### 3.2.1 案例
根据混淆矩阵中案例计算，精确率 = 3/3 = 100%   

## 3.3 召回率
tp/(tp+fn)  
### 3.3.1 案例
根据混淆矩阵中案例计算，召回率 = 3/6 = 50%   

## 3.4 F1-score
$F1-score = (2*Precision*recall)/(precision + recall)$  

### 3.4.1 案例
根据混淆矩阵中案例计算，F1-score = (2*50%)/(100%+50%) = 66.7%   

## 3.5 AUC指标
## 3.6 ROC曲线

# 4. sklearn:逻辑回归API函数与案例
## 4.1 LogisticRegression介绍
```python
class sklearn.linear_model.LogisticRegression(penalty='l2', *, dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=None, solver='lbfgs', max_iter=100, multi_class='auto', verbose=0, warm_start=False, n_jobs=None, l1_ratio=None)
```
Logistic回归（又名logit，MaxEnt）分类器。  

在多类情况下，如果将“ multi_class”选项设置为“ ovr”，则训练算法将使用“一对剩余”（OvR）方案；  
如果将“ multi_class”选项设置为“multinomial(多项式)”，则使用交叉熵损失。（当前，只有“ lbfgs”，“ sag”，“ saga”和“ newton-cg”求解器支持“多项式”选项。）    

**默认将类别数量少的作为正例，亦可自己指定**  

### 4.1.1 penalty
{‘L1’, ‘L2’, ‘elasticnet’, ‘none’}, default=’L2’  用于指定处罚中使用的规范。  
'newton-cg'，'sag'和'lbfgs'求解器仅支持L2惩罚。  
仅“ saga”求解器支持“ elasticnet”。  
如果为“ none”（liblinear求解器不支持），则不应用任何正则化。  

### 4.1.2 C
float, default=1.0  
正则强度的倒数；必须为正浮点数。与支持向量机一样，较小的值指定更强的正则化。  

### 4.1.3 slover  
此类使用'liblinear'库，'newton-cg'，'sag'，'saga'和'lbfgs'求解器实现**正则逻辑回归**。    
**liblinear**：是这些求解器的底层库，提供了高效实现的优化算法。它支持 L1 和 L2 正则化，以及 L1/L2 的混合。
**newton-cg、lbfgs、sag、saga**：是 scikit-learn 提供的求解器选项，它们调用了 liblinear 库的不同优化方法，或在某些情况下使用了 scikit-learn 自己实现的优化器。  

|求解器名称 | 优化方法 | 支持正则化 | 适用数据集 | 是否支持多分类 | 是否适合大规模数据 | 备注 |
|--| --| --| --| --| --| --|
|liblinear | 内部实现（L1/L2） | L1 和 L2 | 小到中等数据 | 是 | 否 | 默认求解器，适合稀疏数据 |
|newton-cg | 牛顿共轭梯度法 | L2 | 中等数据 | 是 | 否 | 计算 Hessian 矩阵，适合小数据集 |
|lbfgs | L-BFGS（有限内存拟牛顿法） | L2 | 小到中等数据 | 是 | 否 | 收敛速度快，适合小数据和非稀疏数据 |
|sag | 随机平均梯度法 | L2 | 大规模数据 | 是 | 是 | 适合大量样本和特征，但不支持 L1 |
|saga | 随机平均梯度法的扩展 | L1、L2、弹性网络（L1+L2） | 大规模数据 | 是 | 是 | sag 的改进版，支持 L1 和弹性网络正则化|

## 4.2 案例：癌症分类预测
**测试代码**  
```python
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import pandas as pd
import numpy as np


def load_data():
    data = load_breast_cancer()
    print("data.keys():\n", data.keys(), "\n")
    print("data.feature_names:\n", data.feature_names, "\n")
    print("data.data[:5]:\n", data.data[:5], "\n")
    print("data.target_names:\n", data.target_names, "\n")
    # 0-恶性，1-良性
    print("data.target[:5]:\n", data.target[:5], "\n")
    data_data_df = pd.DataFrame(data.data)
    print(data_data_df.info())


def train_model():
    # 加载数据集
    breast_cancer_data = load_breast_cancer()
    x = breast_cancer_data.data
    y = breast_cancer_data.target
    # 数据预处理：缺失值/异常值处理，划分训练集合测试集
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=22, test_size=0.2)
    # 特征工程：特征提取，特征预处理(归一化，标准化)，特征降维，特征选择，特征组合
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)
    # 训练模型
    model = LogisticRegression()
    model.fit(x_train, y_train)
    # 模型预测
    y_pred = model.predict(x_test)
    print("y_pred-y_test:\n", y_pred - y_test, "\n")
    # 模型评估：准确率
    # 能用准确率来评估吗？可以，但是不够精确，因为逻辑回归用来解答是或否的问题
    result = accuracy_score(y_true=y_test, y_pred=y_pred)
    print("accuracy_score:\n", result, "\n")
    # 模型评估:后续优化：基于混淆矩阵的精确率，召回率，F1值，ROC曲线，AUC值
    conf_matrix = confusion_matrix(y_true=y_test, y_pred=y_pred, labels=[0, 1])
    print("混淆矩阵:\n", conf_matrix, "\n")
    # precision:pos_label表示哪个值为正例
    p_score = precision_score(y_true=y_test, y_pred=y_pred, pos_label=0)
    print("精确率:\n", p_score, "\n")
    # 召回率:pos_label表示哪个值为正例
    r_score = recall_score(y_true=y_test, y_pred=y_pred, pos_label=0)
    print("召回率:\n", r_score, "\n")
    # f1_score:pos_label表示哪个值为正例
    f_score = f1_score(y_true=y_test, y_pred=y_pred, pos_label=0)
    print("f1-score:\n", f_score, "\n")


if __name__ == '__main__':
    load_data()
    # train_model()

```
**测试结果-load_data()**  
```text
data.keys():
 dict_keys(['data', 'target', 'frame', 'target_names', 'DESCR', 'feature_names', 'filename', 'data_module']) 

data.feature_names:
 ['mean radius' 'mean texture' 'mean perimeter' 'mean area'
 'mean smoothness' 'mean compactness' 'mean concavity'
 'mean concave points' 'mean symmetry' 'mean fractal dimension'
 'radius error' 'texture error' 'perimeter error' 'area error'
 'smoothness error' 'compactness error' 'concavity error'
 'concave points error' 'symmetry error' 'fractal dimension error'
 'worst radius' 'worst texture' 'worst perimeter' 'worst area'
 'worst smoothness' 'worst compactness' 'worst concavity'
 'worst concave points' 'worst symmetry' 'worst fractal dimension'] 

data.data[:5]:
 [[1.799e+01 1.038e+01 1.228e+02 1.001e+03 1.184e-01 2.776e-01 3.001e-01
  1.471e-01 2.419e-01 7.871e-02 1.095e+00 9.053e-01 8.589e+00 1.534e+02
  6.399e-03 4.904e-02 5.373e-02 1.587e-02 3.003e-02 6.193e-03 2.538e+01
  1.733e+01 1.846e+02 2.019e+03 1.622e-01 6.656e-01 7.119e-01 2.654e-01
  4.601e-01 1.189e-01]
 [2.057e+01 1.777e+01 1.329e+02 1.326e+03 8.474e-02 7.864e-02 8.690e-02
  7.017e-02 1.812e-01 5.667e-02 5.435e-01 7.339e-01 3.398e+00 7.408e+01
  5.225e-03 1.308e-02 1.860e-02 1.340e-02 1.389e-02 3.532e-03 2.499e+01
  2.341e+01 1.588e+02 1.956e+03 1.238e-01 1.866e-01 2.416e-01 1.860e-01
  2.750e-01 8.902e-02]
 [1.969e+01 2.125e+01 1.300e+02 1.203e+03 1.096e-01 1.599e-01 1.974e-01
  1.279e-01 2.069e-01 5.999e-02 7.456e-01 7.869e-01 4.585e+00 9.403e+01
  6.150e-03 4.006e-02 3.832e-02 2.058e-02 2.250e-02 4.571e-03 2.357e+01
  2.553e+01 1.525e+02 1.709e+03 1.444e-01 4.245e-01 4.504e-01 2.430e-01
  3.613e-01 8.758e-02]
 [1.142e+01 2.038e+01 7.758e+01 3.861e+02 1.425e-01 2.839e-01 2.414e-01
  1.052e-01 2.597e-01 9.744e-02 4.956e-01 1.156e+00 3.445e+00 2.723e+01
  9.110e-03 7.458e-02 5.661e-02 1.867e-02 5.963e-02 9.208e-03 1.491e+01
  2.650e+01 9.887e+01 5.677e+02 2.098e-01 8.663e-01 6.869e-01 2.575e-01
  6.638e-01 1.730e-01]
 [2.029e+01 1.434e+01 1.351e+02 1.297e+03 1.003e-01 1.328e-01 1.980e-01
  1.043e-01 1.809e-01 5.883e-02 7.572e-01 7.813e-01 5.438e+00 9.444e+01
  1.149e-02 2.461e-02 5.688e-02 1.885e-02 1.756e-02 5.115e-03 2.254e+01
  1.667e+01 1.522e+02 1.575e+03 1.374e-01 2.050e-01 4.000e-01 1.625e-01
  2.364e-01 7.678e-02]] 

data.target_names:
 ['malignant' 'benign'] 

data.target[:5]:
 [0 0 0 0 0] 

data_data_df.info():
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 569 entries, 0 to 568
Data columns (total 30 columns):
 #   Column  Non-Null Count  Dtype  
---  ------  --------------  -----  
 0   0       569 non-null    float64
 1   1       569 non-null    float64
 2   2       569 non-null    float64
 3   3       569 non-null    float64
 4   4       569 non-null    float64
 5   5       569 non-null    float64
 6   6       569 non-null    float64
 7   7       569 non-null    float64
 8   8       569 non-null    float64
 9   9       569 non-null    float64
 10  10      569 non-null    float64
 11  11      569 non-null    float64
 12  12      569 non-null    float64
 13  13      569 non-null    float64
 14  14      569 non-null    float64
 15  15      569 non-null    float64
 16  16      569 non-null    float64
 17  17      569 non-null    float64
 18  18      569 non-null    float64
 19  19      569 non-null    float64
 20  20      569 non-null    float64
 21  21      569 non-null    float64
 22  22      569 non-null    float64
 23  23      569 non-null    float64
 24  24      569 non-null    float64
 25  25      569 non-null    float64
 26  26      569 non-null    float64
 27  27      569 non-null    float64
 28  28      569 non-null    float64
 29  29      569 non-null    float64
dtypes: float64(30)
memory usage: 133.5 KB
None
```
**测试结果train-model()**  
```text
y_pred-y_test:
 [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0] 

accuracy_score:
 0.9736842105263158 

混淆矩阵:
 [[40  3]
 [ 0 71]] 

精确率:
 1.0 

召回率:
 0.9302325581395349 

f1-score:
 0.963855421686747 
```
## 4.3 案例：电信客户流失
**测试代码**  
```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.metrics import classification_report


def load_process_data():
    # 加载数据
    churn_pd = pd.read_csv("./WA_Fn-UseC_-Telco-Customer-Churn.csv")
    print("初始 --> churn_pd.info():")
    churn_pd.info()
    print("\n")
    print("初始 --> churn_pd.describe():\n", churn_pd.describe(), "\n")
    print("初始 --> churn_pd[:5]:\n", churn_pd[:5], "\n")
    # 数据基本处理：类别数据做one-hot编码
    churn_pd = pd.get_dummies(churn_pd, columns=['gender', 'Churn'])
    print("--> gender和Churn做one hot编码 --> churn_pd[:5]:\n", churn_pd[:5], "\n")
    print("--> gender和Churn做one hot编码 --> churn_pd.info():")
    churn_pd.info()
    # 提取保留特征列
    churn_pd.drop(['customerID', 'Churn_No', 'gender_Female'], axis=1, inplace=True)
    print("\n--> 删除customerID列、Churn_No列、gender_Female列 --> churn_pd.info():")
    churn_pd.info()
    # 修改列名
    churn_pd.rename(columns={'Churn_Yes': 'Flag'}, inplace=True)
    print("\n--> 修改列名Churn_Yes为Flag --> churn_pd.info():")
    churn_pd.info()
    # 查看数据分布
    print("\n--> 修改列名Churn_Yes为Flag --> churn_pd.Flag.value_counts():\n", churn_pd.Flag.value_counts(), "\n")
    return churn_pd


def show_data(data: pd.DataFrame):
    sns.countplot(data=data, x='Contract', hue='Flag')
    plt.show()


def train_model(data: pd.DataFrame):
    # 数据基本处理
    x = data[['MonthlyCharges', 'tenure', 'SeniorCitizen']]
    y = data['Flag']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=100)
    # 特征工程
    # 模型训练
    model = LogisticRegression()
    model.fit(x_train, y_train)
    # 模型预测
    y_pred = model.predict(x_test)
    # 模型评估
    a_score = accuracy_score(y_true=y_test, y_pred=y_pred)
    print("a_score\n", a_score, "\n")
    r_a_score = roc_auc_score(y_true=y_test, y_score=y_pred)
    print("r_a_score:\n", r_a_score, "\n")
    c_report = classification_report(y_test, y_pred, target_names=['flag0', 'flag1'])
    print("c_report:\n", c_report)


if __name__ == '__main__':
    data_df = load_process_data()
    show_data(data_df)
    train_model(data_df)
```
**测试结果load_process_data**  
```text
初始 --> churn_pd.info():
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 7043 entries, 0 to 7042
Data columns (total 21 columns):
 #   Column            Non-Null Count  Dtype  
---  ------            --------------  -----  
 0   customerID        7043 non-null   object 
 1   gender            7043 non-null   object 
 2   SeniorCitizen     7043 non-null   int64  
 3   Partner           7043 non-null   object 
 4   Dependents        7043 non-null   object 
 5   tenure            7043 non-null   int64  
 6   PhoneService      7043 non-null   object 
 7   MultipleLines     7043 non-null   object 
 8   InternetService   7043 non-null   object 
 9   OnlineSecurity    7043 non-null   object 
 10  OnlineBackup      7043 non-null   object 
 11  DeviceProtection  7043 non-null   object 
 12  TechSupport       7043 non-null   object 
 13  StreamingTV       7043 non-null   object 
 14  StreamingMovies   7043 non-null   object 
 15  Contract          7043 non-null   object 
 16  PaperlessBilling  7043 non-null   object 
 17  PaymentMethod     7043 non-null   object 
 18  MonthlyCharges    7043 non-null   float64
 19  TotalCharges      7043 non-null   object 
 20  Churn             7043 non-null   object 
dtypes: float64(1), int64(2), object(18)
memory usage: 1.1+ MB


初始 --> churn_pd.describe():
        SeniorCitizen       tenure  MonthlyCharges
count    7043.000000  7043.000000     7043.000000
mean        0.162147    32.371149       64.761692
std         0.368612    24.559481       30.090047
min         0.000000     0.000000       18.250000
25%         0.000000     9.000000       35.500000
50%         0.000000    29.000000       70.350000
75%         0.000000    55.000000       89.850000
max         1.000000    72.000000      118.750000 

初始 --> churn_pd[:5]:
    customerID  gender  SeniorCitizen  ... MonthlyCharges TotalCharges  Churn
0  7590-VHVEG  Female              0  ...          29.85        29.85     No
1  5575-GNVDE    Male              0  ...          56.95       1889.5     No
2  3668-QPYBK    Male              0  ...          53.85       108.15    Yes
3  7795-CFOCW    Male              0  ...          42.30      1840.75     No
4  9237-HQITU  Female              0  ...          70.70       151.65    Yes

[5 rows x 21 columns] 

--> gender和Churn做one hot编码 --> churn_pd[:5]:
    customerID  SeniorCitizen Partner  ... gender_Male  Churn_No Churn_Yes
0  7590-VHVEG              0     Yes  ...       False      True     False
1  5575-GNVDE              0      No  ...        True      True     False
2  3668-QPYBK              0      No  ...        True     False      True
3  7795-CFOCW              0      No  ...        True      True     False
4  9237-HQITU              0      No  ...       False     False      True

[5 rows x 23 columns] 

--> gender和Churn做one hot编码 --> churn_pd.info():
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 7043 entries, 0 to 7042
Data columns (total 23 columns):
 #   Column            Non-Null Count  Dtype  
---  ------            --------------  -----  
 0   customerID        7043 non-null   object 
 1   SeniorCitizen     7043 non-null   int64  
 2   Partner           7043 non-null   object 
 3   Dependents        7043 non-null   object 
 4   tenure            7043 non-null   int64  
 5   PhoneService      7043 non-null   object 
 6   MultipleLines     7043 non-null   object 
 7   InternetService   7043 non-null   object 
 8   OnlineSecurity    7043 non-null   object 
 9   OnlineBackup      7043 non-null   object 
 10  DeviceProtection  7043 non-null   object 
 11  TechSupport       7043 non-null   object 
 12  StreamingTV       7043 non-null   object 
 13  StreamingMovies   7043 non-null   object 
 14  Contract          7043 non-null   object 
 15  PaperlessBilling  7043 non-null   object 
 16  PaymentMethod     7043 non-null   object 
 17  MonthlyCharges    7043 non-null   float64
 18  TotalCharges      7043 non-null   object 
 19  gender_Female     7043 non-null   bool   
 20  gender_Male       7043 non-null   bool   
 21  Churn_No          7043 non-null   bool   
 22  Churn_Yes         7043 non-null   bool   
dtypes: bool(4), float64(1), int64(2), object(16)
memory usage: 1.0+ MB

--> 删除customerID列、Churn_No列、gender_Female列 --> churn_pd.info():
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 7043 entries, 0 to 7042
Data columns (total 20 columns):
 #   Column            Non-Null Count  Dtype  
---  ------            --------------  -----  
 0   SeniorCitizen     7043 non-null   int64  
 1   Partner           7043 non-null   object 
 2   Dependents        7043 non-null   object 
 3   tenure            7043 non-null   int64  
 4   PhoneService      7043 non-null   object 
 5   MultipleLines     7043 non-null   object 
 6   InternetService   7043 non-null   object 
 7   OnlineSecurity    7043 non-null   object 
 8   OnlineBackup      7043 non-null   object 
 9   DeviceProtection  7043 non-null   object 
 10  TechSupport       7043 non-null   object 
 11  StreamingTV       7043 non-null   object 
 12  StreamingMovies   7043 non-null   object 
 13  Contract          7043 non-null   object 
 14  PaperlessBilling  7043 non-null   object 
 15  PaymentMethod     7043 non-null   object 
 16  MonthlyCharges    7043 non-null   float64
 17  TotalCharges      7043 non-null   object 
 18  gender_Male       7043 non-null   bool   
 19  Churn_Yes         7043 non-null   bool   
dtypes: bool(2), float64(1), int64(2), object(15)
memory usage: 1004.3+ KB

--> 修改列名Churn_Yes为Flag --> churn_pd.info():
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 7043 entries, 0 to 7042
Data columns (total 20 columns):
 #   Column            Non-Null Count  Dtype  
---  ------            --------------  -----  
 0   SeniorCitizen     7043 non-null   int64  
 1   Partner           7043 non-null   object 
 2   Dependents        7043 non-null   object 
 3   tenure            7043 non-null   int64  
 4   PhoneService      7043 non-null   object 
 5   MultipleLines     7043 non-null   object 
 6   InternetService   7043 non-null   object 
 7   OnlineSecurity    7043 non-null   object 
 8   OnlineBackup      7043 non-null   object 
 9   DeviceProtection  7043 non-null   object 
 10  TechSupport       7043 non-null   object 
 11  StreamingTV       7043 non-null   object 
 12  StreamingMovies   7043 non-null   object 
 13  Contract          7043 non-null   object 
 14  PaperlessBilling  7043 non-null   object 
 15  PaymentMethod     7043 non-null   object 
 16  MonthlyCharges    7043 non-null   float64
 17  TotalCharges      7043 non-null   object 
 18  gender_Male       7043 non-null   bool   
 19  Flag              7043 non-null   bool   
dtypes: bool(2), float64(1), int64(2), object(15)
memory usage: 1004.3+ KB

--> 修改列名Churn_Yes为Flag --> churn_pd.Flag.value_counts():
 Flag
False    5174
True     1869
Name: count, dtype: int64 
```
**测试结果show_data**  

**测试结果train_model**  
```text
a_score
 0.7757274662881476 

r_a_score:
 0.6764837398373984 

c_report:
               precision    recall  f1-score   support

       flag0       0.82      0.89      0.85      1025
       flag1       0.62      0.46      0.53       384

    accuracy                           0.78      1409
   macro avg       0.72      0.68      0.69      1409
weighted avg       0.76      0.78      0.76      1409
```
