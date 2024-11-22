import torch
import nltk
from tqdm import tqdm
import torch.nn as nn
from torchtext.vocab import build_vocab_from_iterator
import torch.nn.functional as F
from torch.utils.data import Dataset
import matplotlib.pyplot as plt


BOS_TOKEN='<bos>'  
EOS_TOKEN='<eos>'
PAD_TOKEN='<pad>'


# 加载 Reuters 语料库并构建数据集
def load_reuters():
    from nltk.corpus import reuters
    from collections import Counter
    # 获取reuters中所有句子
    text = reuters.sents()
    text = [[word.lower() for word in sentence] for sentence in text]
    
    special_tokens = [BOS_TOKEN, EOS_TOKEN, PAD_TOKEN]
    
    # 构建词汇表迭代器
    def yield_tokens(data_iter):
        for sentence in data_iter:
            yield sentence
    
    vocab = build_vocab_from_iterator(yield_tokens(text), specials=special_tokens)

    # 利用词表将文本数据转换为id表示
    corpus = [[vocab[token] for token in sentence] for sentence in text]
    return corpus, vocab


# 创建FFN的数据处理类 NGramDataset
class NGramDataset(Dataset):
    def __init__(self, corpus, vocab, context_size=2):
        self.data = []
        self.bos = vocab[BOS_TOKEN]
        self.eos = vocab[EOS_TOKEN]
        self.pad = vocab[PAD_TOKEN]
        for sentence in tqdm(corpus, desc="Dataset Construction"):
            # 插入句首句尾标记符
            sentence = [self.bos] + sentence + [self.eos]
            if len(sentence) < context_size:
                continue
            for i in range(context_size, len(sentence)):
                # 模型输入为长度为context_size的上下文，输出为当前词
                context = sentence[i-context_size:i]
                target = sentence[i]
                # 当前训练样本由（context, target）构成
                self.data.append((context, target))

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, i):
        return self.data[i]
    
    def collate_fn(self, examples):
        # 从独立样本中构建批次的输入输出，并转换为pytorch张量
        inputs = torch.tensor([ex[0] for ex in examples], dtype=torch.long)
        targets = torch.tensor([ex[1] for ex in examples], dtype=torch.long)
        return inputs, targets
    

# 创建FNN模型
class FNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, context_size, hidden_dim):
        super(FNN, self).__init__()
        # 词向量层
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # 线性变换:词向量->隐藏层
        self.linear1 = nn.Linear(context_size*embedding_dim, hidden_dim)
        # 线性变换：隐藏层->输出层
        self.linear2 = nn.Linear(hidden_dim, vocab_size)
        # 使用ReLU激活函数
        self.activate = F.relu
    
    def forward(self, inputs):        
        # 将输入序列映射为词向量，并通过 view 函数对映射后的词向量序列组成的三维张量进行重构，以完成词向量的拼接
        embeds = self.embedding(inputs).view((inputs.shape[0],-1))
        hidden = self.activate(self.linear1(embeds))
        output = self.linear2(hidden)
        log_probs = F.log_softmax(output, dim=1)
        return log_probs

# 保存词表以及训练得到的词向量
def save_pretrained(vocab, embeds, save_path):
    with open(save_path, "w") as writer:
        writer.write(f"{embeds.shape[0]} {embeds.shape[1]}\n")
        for idx, token in enumerate(vocab.get_itos()):
            vec = "".join([f"{x:.6f}" for x in embeds[idx]])
            # 每一行对应一个单词以及由空格分隔的词向量
            writer.write(f"{token} {vec}\n")
    

# 设置超参数，对模型进行训练
embedding_dim = 128
hidden_dim = 256
batch_size = 1024
context_size = 3  # 输入上下文词长度（？？？）
# num_epoch = 10
num_epoch = 5

# 读取文本数据，构建FNN训练数据
corpus, vocab = load_reuters()
dataset = NGramDataset(corpus, vocab, context_size)
data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, collate_fn=dataset.collate_fn)
# 负对数似然损失函数
nll_loss = nn.NLLLoss() 

# 构建FNN模型
model = FNN(len(vocab), embedding_dim, context_size, hidden_dim)
# 使用 Adam 优化器
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 训练模型
model.train()
total_losses=[]
for epoch in range(num_epoch):
    total_loss = 0
    for batch in tqdm(data_loader, desc=f"Training Epoch {epoch}"):
        inputs, targets = batch
        # print(f"Inputs type: {type(inputs)}")  # 打印数据类型
        optimizer.zero_grad()
        log_probs = model(inputs)
        loss = nll_loss(log_probs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Loss:{total_loss}")
    total_losses.append(total_loss)

# 将词向量保存到文件
save_pretrained(vocab, model.embedding.weight.data, "ffnnlm.vec")
# 绘制损失函数曲线
plt.plot(range(num_epoch), total_losses)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss Curve during Training')
plt.show()
