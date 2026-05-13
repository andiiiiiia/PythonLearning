# 1. RNN介绍
## 1.1 RNN简介
RNN（循环神经网络）是一种专门用于处理序列数据的神经网络结构。与传统的神经网络不同，RNN具有记忆能力，可以记住之前输入的信息，并将这些信息用于当前的计算。当前的输出不仅依赖于当前的输入，还依赖于之前的状态。    

在RNN中，每个时间步都会有一个“隐藏状态”（hidden state），这个状态会从一个时间步传递到下一个时间步，从而实现对序列信息的记忆。  

在RNN网络中，一般只有一个隐藏层。  

我们假设每个时间步的输入为  $x_{t}$  ,每个时间步的隐藏状态为  $h_{t}$  ,每个时间步的输出为  $y_{t}$  。  

**则每个时间步的隐藏层状态计算公式为:**   

$h_{t} = f(W_{xh} * x_{t} + W_{hh} * h_{t-1} + b_{h})$   

$W_{xh}$  ：输入到隐藏层的权重矩阵  
$W_{hh}$  ：隐藏层到隐藏层的权重矩阵（循环连接）  
$b_h$  ：隐藏层的偏置项  
$f$  ：激活函数，通常使用 tanh 或 ReLU  

**注意：**  

这里的 $h_{t}$  表示向量。  

**则每个时间步的输出计算公式为:**   

$y_{t} = W_{hy} * h_{t} + b_{y}$  

$W_{hy}$ ：隐藏层到输出层的权重矩阵

$b_{y}$：输出层的偏置项

**注意：**  
这里的  $y_{t}$   通常表示模型预测下一个词的概率分布，就是**词表中所有词的概率**。假设你的词表中有 1000 个词，那么，它是一个长度为 10000 的向量，每个元素代表概率可能的概率，所有元素和为1。  

## 1.2 RNN应用场景
自然语言处理:机器翻译,情感分析  
语音识别:语音转文字  
时间序列预测：股票价格预测，天气预测  
语音合成  
代码生成   
变体：LSTM/gru  

# 2. 词嵌入层 embedding
## 2.1 是什么

将离散的词（通常是**整数索引**）转换为连续的向量表示，从而让神经网络能够更好地理解和处理语言信息。   

将每个词映射为一个固定长度的向量。这些向量可以捕捉词语之间的语义关系，比如相似性、上下文关系等。  

## 2.2 作用

- 将离散的词转换为连续的向量。例如，将词 "猫" 转换为 [0.1, 0.3, -0.2] 这样的向量。  

- 捕捉词语之间的语义关系。语义相近的词在向量空间中距离较近，例如 "猫" 和 "狗" 的向量可能比较接近。  

- 降维。将高维的 one-hot 编码词向量（如 10000 维）转换为低维的词嵌入向量（如 100 维）。  

- 提升模型性能。使用词嵌入可以显著提升模型在 NLP 任务中的表现。

## 2.3 工作流程

源文档  ->  jieba分词+去重  ->  所有行的词列表(all_line_words) + 所有去重后的词列表(uniq_words)  

去重后的词列表  ->  enumerate  ->  词表(uniq_words_2_index) { 词 , index }    

原文档的所有行的词列表(all_line_words)  +  词表(uniq_words_2_index)  ->  映射转换  ->  源文档的词索引列表(corpus_index)   

定义词嵌入层embed = nn.embedding(len(uniq_words) , embedding_dim)     

inputs( batch_size , seq_len )  ->  embed() -> 词向量( batch_size , seq_len , embedding_dim ) 


## 2.4 案例：词嵌入层nn.Embedding

```python
import jieba
import torch

if __name__ == '__main__':
    text = "词嵌入层（Embedding Layer）是深度学习中处理文本数据时非常关键的一个组件，它的核心作用是将离散的词（通常是整数索引）转换为连续的向量表示，从而让神经网络能够更好地理解和处理语言信息。"
    print("\n原文本:\n", text)
    words = jieba.lcut(text)
    print("\n分词结果:\n", words)
    # uniq
    uniq_words = list(set(words))
    print("\n去重后分词结果:\n", uniq_words)
    # 创建embedding层
    embed = torch.nn.Embedding(len(uniq_words), 4)
    print("\nEmbedding层:\n", embed)
    # 通过embed将词索引转为向量
    for i, word in enumerate(uniq_words):
        word_vector = embed(torch.tensor(i)) # 随机初始化
        print(f"\n{word}的向量表示:\n", word_vector)
```

**执行结果**  

```text
原文本:
 词嵌入层（Embedding Layer）是深度学习中处理文本数据时非常关键的一个组件，它的核心作用是将离散的词（通常是整数索引）转换为连续的向量表示，从而让神经网络能够更好地理解和处理语言信息。
Loading model cost 0.408 seconds.
Prefix dict has been built successfully.

分词结果:
 ['词', '嵌入', '层', '（', 'Embedding', ' ', 'Layer', '）', '是', '深度', '学习', '中', '处理', '文本', '数据', '时', '非常', '关键', '的', '一个', '组件', '，', '它', '的', '核心作用', '是', '将', '离散', '的', '词', '（', '通常', '是', '整数', '索引', '）', '转换', '为', '连续', '的', '向量', '表示', '，', '从而', '让', '神经网络', '能够', '更好', '地', '理解', '和', '处理', '语言', '信息', '。']

去重后分词结果:
 ['时', '通常', '转换', '学习', '地', '将', '整数', '深度', '关键', '处理', '层', '向量', '从而', '和', '一个', '索引', '语言', '）', ' ', '连续', '表示', '离散', '它', 'Embedding', '词', '嵌入', '，', '理解', '是', '中', '数据', '让', '非常', '文本', '信息', '更好', '的', '组件', '能够', '（', '核心作用', 'Layer', '。', '为', '神经网络']

Embedding层:
 Embedding(45, 4)

时的向量表示:
 tensor([-1.0634,  0.1218, -0.7351, -0.0115], grad_fn=<EmbeddingBackward0>)

通常的向量表示:
 tensor([-0.5348,  0.3350, -0.4949, -0.7021], grad_fn=<EmbeddingBackward0>)

转换的向量表示:
 tensor([-0.6266, -0.6247, -0.5498,  1.2492], grad_fn=<EmbeddingBackward0>)

学习的向量表示:
 tensor([ 0.3728, -0.6975, -0.2273,  0.8428], grad_fn=<EmbeddingBackward0>)

地的向量表示:
 tensor([ 0.0414, -0.2242, -1.3842,  1.5525], grad_fn=<EmbeddingBackward0>)

........

。的向量表示:
 tensor([-1.5797, -1.5140, -0.8331,  0.3552], grad_fn=<EmbeddingBackward0>)

为的向量表示:
 tensor([-0.4637,  1.1501, -1.3131, -2.3106], grad_fn=<EmbeddingBackward0>)

神经网络的向量表示:
 tensor([ 0.6168, -0.1670,  2.1379,  0.6653], grad_fn=<EmbeddingBackward0>)
```

# 3. 循环网络层
## 3.1 API：

**定义：rnn = torch.nn.RNN(input_size=,hidden_size=,num_layers=)**    

input_size： 输入的词向量的embedding_dim,   
hidden_size: 想要得到的隐藏层的维度，也就是rnn层的神经元个数   
num_layers： 隐藏层个数，一般就1个隐藏层  

**使用：output,h1 = rnn(h0,x)**   

x的表示形式为[seq_len, batch, input_size)，即[句子的长度，batch的大小，词向量的维度)  

h0的表示形式为[num_layers, batch, hidden_size)，即[隐藏层的层数，batch的大，隐藏层h的维数)  

output的表示形式与输入x类似，为[seq_len, batch, hidden_size)，即[句子的长度，batch的大小，输出向量的维度)  

hn的表示形式与输入h0一样，为(num_layers, batch, hidden_size)，即[隐藏层的层数，batch的大，隐藏层h的维度)  

## 3.2 案例-demo
```python
    # 1. 定义网络模型
    rnn = torch.nn.RNN(input_size=128, hidden_size=256, num_layers=1)

    # 2. 定义词向量
    # seq_len:每个样本有 5 个时间步
    # batch:即一次处理 32 个样本。
    # input_size: 输入的词向量维度
    x = torch.randn(size=(5, 32, 128))

    # 3. 初始化隐藏层状态h0
    # num_layer = 1, batch = 32, hidden_size = 256
    h0 = torch.randn(size=(1, 32, 256))

    # 4. 执行
    output, h1 = rnn(x, h0)
    print("\noutput.shape:\n", output.shape)
    print("\nh1.shape:\n", h1.shape)
```

**执行结果**  
```text
output.shape:
 torch.Size([5, 32, 256])

h1.shape:
 torch.Size([1, 32, 256])

```

# 4. 文本生成模型案例

## 4.1 构建唯一词表 和 所有词的索引列表

```python
# 源文档 -> jieba分词 -> 去重 -> enumerate转换 -> 构建词表(uniq_word_2_index)
# 根据词表 -> 将原文档转换为 -> 纯索引文档(corpus_index)
def build_vocab():
    # 1. 构造all_line_words和uniq_words
    # 记录jieba分词结果，记录jieba分词结果去重后的结果
    uniq_words, all_line_words = [], []
    for line in open("./data/jaychou_lyrics.txt", 'r', encoding="utf-8"):
        line_words = list(jieba.cut(line))
        all_line_words.append(line_words)
        for word in line_words:
            # 当word不在uniq_words中，则添加到uniq_words中
            if word not in uniq_words:
                uniq_words.append(word)
    print("\n前2行歌词：all_line_words:\n", all_line_words[:2])

    # 2. 根据uniq_words生成词表uniq_word_2_idnex
    # 用uniq_words构建 **词表** ：生成词-索引
    uniq_word_2_index = {word: index for index, word in enumerate(uniq_words)}

    # 3. 根据uniq_word_2_idnex和all_words生成corpus_index
    # 将原歌词文件内容all_words用词表索引表示，得到 **索引文档**
    corpus_index = []
    for line_words in all_line_words:
        tmp = []
        for word in line_words:
            tmp.append(uniq_word_2_index[word])
        # 每行之间添加分隔符
        tmp.append(uniq_word_2_index[" "])
        corpus_index.extend(tmp)

    # 4. 返回uniq_words，count(uniq_words), uniq_word_2_index，corpus_index
    # 后续重点关注 uniq_word_2_index 和 corpus_index
    return uniq_words, len(uniq_words), uniq_word_2_index, corpus_index
```

## 4.2 继承dataset

```python
# 最终得到：x(batch_size,seq_len) , y(batch_size,seq_len)
class LyricDataset(torch.utils.data.Dataset):
    # sql_len: 每条数据(句子)的词个数，即时间步数量
    def __init__(self, corpus_index, seq_len):
        super(LyricDataset, self).__init__()
        # 每条数据(句子)的词个数
        self.seq_len = seq_len
        # 原始歌词索引文档长度
        self.corpus_len = len(corpus_index)
        # 原始歌词索引文档
        self.corpus_index = corpus_index
        # 可以生成的所有样本数量(所有句子数量): 采用可重叠的样本数策略，通过窗口滑动，整个歌词索引文档可以被分割成多少个完整的序列
        self.seq_num = self.corpus_len - self.seq_len

    def __len__(self):
        print("=" * 20, "begin LyricDataset : __len__", "=" * 20)
        print("self.seq_num:\n", self.seq_num)
        print("=" * 20, "end LyricDataset : __len__", "=" * 20)
        return self.seq_num

    def __getitem__(self, index):
        # 开始索引
        start_index = min(max(index, 0), self.corpus_len - self.seq_len)
        # 结束索引
        end_index = start_index + self.seq_len
        # 输入值x
        x = self.corpus_index[start_index: end_index]
        # 输出值y，真实值
        y = self.corpus_index[start_index + 1: end_index + 1]
        return torch.tensor(x), torch.tensor(y)
```


## 4.3 模型结构

```python
# 搭建RNN神经网络
class TextGeneratorRNNModel(torch.nn.Module):
    def __init__(self, uniq_words_count, embedding_dim, hidden_dim, n_layers):
        super(TextGeneratorRNNModel, self).__init__()
        self.uniq_words_count = uniq_words_count
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        # 词嵌入层
        self.embedding = torch.nn.Embedding(uniq_words_count, embedding_dim)
        # RNN层
        self.rnn = torch.nn.RNN(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=n_layers)
        # 输出层
        self.out = torch.nn.Linear(hidden_dim, uniq_words_count)

    def forward(self, inputs, hidden):
        print("=" * 20, "begin : TextGeneratorRNNModel : forward", "=" * 20)
        print("\ninputs.shape:\n", inputs.shape)

        # 词嵌入
        embeds = self.embedding(inputs)
        print("\nembeds.shape:\n", embeds.shape)

        # RNN
        y, hidden = self.rnn(embeds.transpose(0, 1), hidden)
        print("\ny.shape:\n", y.shape)
        print("\nhidden.shape:\n", hidden.shape)

        # 全连接输出层
        y = y.reshape(-1, y.shape[-1])
        print("\ny = y.reshape(-1, y.shape[-1]) : shape:\n", y.shape)

        output = self.out(y)
        print("\noutput = self.out(y) : shape:\n", output.shape)

        print("=" * 20, "end : TextGeneratorRNNModel : forward", "=" * 20)
        # 返回
        return output, hidden

    def init_hidden(self, batch_size):
        print("=" * 20, "begin : TextGeneratorRNNModel : init_hidden", "=" * 20)
        hidden = torch.zeros(self.n_layers, batch_size, self.hidden_dim)
        print("\nhidden.shape:\n", hidden.shape)

        print("=" * 20, "end : TextGeneratorRNNModel : init_hidden", "=" * 20)
        return hidden
```

## 4.4 模型训练

```python
def train(seq_len, batch_size, embedding_dim, hidden_dim, n_layers):
    print("=" * 20, "begin : 模型训练 train", "=" * 20) 
    # 1. 构建词表
    uniq_words, uniq_words_count, uniq_word_2_index, corpus_index = build_vocab()
    # 2. 构建数据集
    dataset = LyricDataset(corpus_index, seq_len=seq_len)
    # 3. 构建数据加载器
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    # 4. 构建模型
    model = TextGeneratorRNNModel(uniq_words_count, embedding_dim=embedding_dim, hidden_dim=hidden_dim,
                                  n_layers=n_layers)
    # 5. 构建损失函数
    criterion = torch.nn.CrossEntropyLoss()
    # 6. 构建优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 7. 模型训练
    EPOCHS = 10
    for epoch in range(EPOCHS):
        # 定义记录变量，开始时间，迭代次数，训练总损失
        start_time, batch_count, epoch_total_loss = time.time(), 0, 0

        # 轮询每个批次
        for x, y in data_loader:
            print("=" * 20, "begin : batch开始", "=" * 20)

            print("\ndata_loader : x.shape:\n", x.shape)
            print("\ndata_loader : y.shape:\n", y.shape)

            # 隐藏值初始化，考虑到整除问题，方案1：这里的batch_size最好用动态的，取x的批次就行。
            # 方案2：取固定值，但是需要丢弃最后一个批次，或者填充最后一个批次为batch_size大小，否则可能会报错。
            # PS: 这里有个问题，就是每个批次都初始化了hidden，导致丢失前一批次的记忆,正常情况应该使用上述方案2，并将此行代码放到当前循环外。
            hidden = model.init_hidden(x.size(0))

            # 前向传播
            output, hidden = model(x, hidden)

            # 计算损失
            # 先转置，假设batch_size=10, seq_len=32, 从(10,32)调整为(32,10),再做展开，为什么？因为output也是按照这个顺序展开的，要保证一致。
            y = torch.transpose(y,0,1).reshape(-1)
            print("\n y = torch.transpose(y,0,1).reshape(-1) : shape:\n", y.shape)
            
            loss = criterion(output, y)

            # 梯度清零，反向传播，更新权重
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 计算
            epoch_total_loss += loss.item()
            batch_count += 1
            print("=" * 20, "begin : batch结束", "=" * 20)
        print("epoch:{}, loss:{:.4f}, time:{:.4f}".format(epoch, epoch_total_loss / batch_count,
                                                          time.time() - start_time))
    # 保存模型
    torch.save(model.state_dict(), "./model/TextGeneratorRNNModel.pkl")
```

**train的执行**  

```python
train(seq_len=32, batch_size=10, embedding_dim=128, hidden_dim=256, n_layers=1)
```

**执行结果**  

因为会执行很多次batch，节省篇幅，只取第一个batch展示。重点关注一下数据在各个阶段中的shape以及转换。

```text
==================== begin LyricDataset : __len__ ====================
self.seq_num:
 49104
==================== end LyricDataset : __len__ ====================
==================== begin LyricDataset : __len__ ====================
self.seq_num:
 49104
==================== end LyricDataset : __len__ ====================
==================== begin LyricDataset : __len__ ====================
self.seq_num:
 49104
==================== end LyricDataset : __len__ ====================
==================== begin LyricDataset : __len__ ====================
self.seq_num:
 49104
==================== end LyricDataset : __len__ ====================
==================== begin : batch开始 ====================

data_loader : x.shape:
 torch.Size([10, 32])

data_loader : y.shape:
 torch.Size([10, 32])
==================== begin : TextGeneratorRNNModel : init_hidden ====================

hidden.shape:
 torch.Size([1, 10, 256])
==================== end : TextGeneratorRNNModel : init_hidden ====================
==================== begin : TextGeneratorRNNModel : forward ====================

inputs.shape:
 torch.Size([10, 32])

embeds.shape:
 torch.Size([10, 32, 128])

y.shape:
 torch.Size([32, 10, 256])

hidden.shape:
 torch.Size([1, 10, 256])

y = y.reshape(-1, y.shape[-1]) : shape:
 torch.Size([320, 256])

output = self.out(y) : shape:
 torch.Size([320, 5703])
==================== end : TextGeneratorRNNModel : forward ====================

 y = torch.transpose(y,0,1).reshape(-1) : shape:
 torch.Size([320])
==================== begin : batch结束 ====================
```

## 4.5 模型测试
```python
# 模型预测
# 前提：start_word只能是当前文档中jieba分好的已有的词，并且只能有1个
def evaluate(start_word, seq_len):
    print("=" * 20, "begin : 模型预测 evaluate", "=" * 20)
    # 1. build词表
    uniq_words, uniq_words_count, uniq_word_2_index, corpus_index = build_vocab()
    # 2. 使用模型
    model = TextGeneratorRNNModel(uniq_words_count, embedding_dim=128, hidden_dim=256, n_layers=1)
    model.load_state_dict(torch.load("./model/TextGeneratorRNNModel.pkl"))
    # 3. 获取隐藏层初始值
    hidden = model.init_hidden(batch_size=1)
    # 4. 将输入词转换为索引
    start_words_index = [uniq_word_2_index[start_word]]
    # 实际上可以输入多个词，通过下述方式输入，但是不能超过模型训练时的时间步长度
    # start_words_index = [uniq_word_2_index[i] for i in start_word]

    # 5. 模型预测
    predict_sentence_index = [start_words_index]
    for i in range(seq_len):
        x = torch.tensor([predict_sentence_index[-1]])
        print("\n模型预测输入 x:\n",x)
        output, hidden = model(x, hidden)
        predict_index = torch.argmax(output, dim=1).item()
        predict_sentence_index.append([predict_index])
    # 将index转换为词
    predict_sentence = [uniq_words[i[0]] for i in predict_sentence_index]
    print("".join(predict_sentence))
    print("=" * 20, "end : 模型预测 evaluate", "=" * 20)
```
**执行结果**  
```python
前2行歌词：all_line_words:
 [['想要', '有', '直升机', '\n'], ['想要', '和', '你', '飞到', '宇宙', '去', '\n']]
==================== begin : TextGeneratorRNNModel : init_hidden ====================

hidden.shape:
 torch.Size([1, 1, 256])
==================== end : TextGeneratorRNNModel : init_hidden ====================

模型预测输入 x:
 tensor([[117]])
==================== begin : TextGeneratorRNNModel : forward ====================

inputs.shape:
 torch.Size([1, 1])

embeds.shape:
 torch.Size([1, 1, 128])

y.shape:
 torch.Size([1, 1, 256])

hidden.shape:
 torch.Size([1, 1, 256])

y = y.reshape(-1, y.shape[-1]) : shape:
 torch.Size([1, 256])

output = self.out(y) : shape:
 torch.Size([1, 5703])
==================== end : TextGeneratorRNNModel : forward ====================

模型预测输入 x:
 tensor([[3]])
==================== begin : TextGeneratorRNNModel : forward ====================

inputs.shape:
 torch.Size([1, 1])

embeds.shape:
 torch.Size([1, 1, 128])

y.shape:
 torch.Size([1, 1, 256])

hidden.shape:
 torch.Size([1, 1, 256])

y = y.reshape(-1, y.shape[-1]) : shape:
 torch.Size([1, 256])

output = self.out(y) : shape:
 torch.Size([1, 5703])
==================== end : TextGeneratorRNNModel : forward ====================

...省略


星星
 一颗两颗三颗四颗 连成线乘著风 游荡在蓝天边
 一片云掉落在我面前
 捏成你的形状
 随风跟著我
 一口一口吃掉忧愁
 载著你 彷彿载著阳光
 不管到哪里都是晴天
 蝴蝶自在飞
 花也布满天
 一朵一朵因你而香
 试图让夕阳飞翔
 带领你我环绕大自然
 迎著风 开始共渡每一天
 手牵手
==================== end : 模型预测 evaluate ====================
```
