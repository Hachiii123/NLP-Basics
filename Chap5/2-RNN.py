import torch
import nltk
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


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

# 创建数据集
class RnnlmDataset(Dataset):
    def __init__(self, corpus, vocab):
        self.data = []
        self.bos = vocab[BOS_TOKEN]
        self.eos = vocab[EOS_TOKEN]
        self.pad = vocab[PAD_TOKEN]
        for sentence in corpus:
            input = [self.bos] + sentence 
            target = sentence + [self.eos]
            self.data.append((input, target))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, i):
        return self.data[i]
    
    # 从独立样本集合中构建批次输入输出
    def collate_fn(self, examples):
        inputs = [torch.tensor(ex[0]) for ex in examples]
        targets = [torch.tensor(ex[1] for ex in examples)]
        
        # 长度补齐
        inputs = pad_sequence(inputs, batch_first=True, padding_valus=self.pad)
        targets = pad_sequence(targets, batch_first=True, padding_valus=self.pad)
        return inputs, targets
    

# 创建RNNLM模型
class RNNLM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
         super(RNNLM, self).__init__()
         self.embedding = nn.Embedding(vocab_size, embedding_dim)
         self.rnn = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
         self.output = nn.Linear(hidden_dim, vocab_size)

    def forward(self, inputs):
        embeds = self.embedding(inputs)
        hidden, _ = self.rnn(embeds)
        output = self.output(hidden)
        log_probs = F.log_softmax(output, dim=2)
        return log_probs
    
# 保存词表以及训练得到的词向量
def save_pretrained(vocab, embeds, save_path):
    with open(save_path, "w") as writer:
        writer.write(f"{embeds.shape[0]} {embeds.shape[1]}\n")
        for idx, token in enumerate(vocab.get_itos()):
            vec = "".join([f"{x:.6f}" for x in embeds[idx]])
            # 每一行对应一个单词以及由空格分隔的词向量
            writer.write(f"{token} {vec}\n")


# 模型训练
# 设置超参数
embedding_dim = 128
hidden_dim = 256
batch_size = 128
num_epoch = 5

# 读取reuters文本数据
corpus, vocab = load_reuters
dataset = RnnlmDataset(corpus, vocab)
data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, collate_fn=dataset.collate_fn)

# 构建FNN模型
model = RNNLM(len(vocab), embedding_dim, hidden_dim)
nll_loss = nn.NLLLoss(ignore_index=dataset.pad)   # 忽略PAD_TOKEN处的损失
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 开始训练
model.train()
total_losses=[]
for epoch in range(num_epoch):
    total_loss = 0
    for batch in tqdm(data_loader, desc=f"Training Epoch {epoch}"):
        inputs, targets = batch
        # print(f"Inputs type: {type(inputs)}")  # 打印数据类型
        optimizer.zero_grad()
        log_probs = model(inputs)
        loss = nll_loss(log_probs.view(-1, log_probs.shape[-1]), targets.view(-1))
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




