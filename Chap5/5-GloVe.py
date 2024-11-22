import torch
import nltk
from tqdm import tqdm
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import Dataset
from collections import defaultdict



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


class GloveDataset(Dataset):
    def __init__(self, corpus, vocab, context_size=2):
        # 共现矩阵记录西与上下文的共现次数
        self.cooccur_counts = defaultdict(float)
        self.bos = vocab[BOS_TOKEN]
        self.eos = vocab[EOS_TOKEN]
        for sentence in tqdm(corpus, desc="Dataset COnstruction"):
            sentence = [self.bos] + sentence + [self.eos]
            for i in range(1, len(sentence)-1):
                w = sentence[i]
                left_contexts = sentence[max(0, i-context_size):i]
                right_contexts = sentence[i+1:min(len(sentence), i+context_size+1)]
                # 共现次数随距离衰减
                for k,c in enumerate(left_contexts[::-1]):
                    self.cooccur_counts[(w, c)] += 1/(k+1)
                for k,c in enumerate(right_contexts):
                    self.cooccur_counts[(w, c)] += 1/(k+1)
        
        self.data = [(w, c, count) for (w, c), count in self.cooccur_counts.items()]

    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    def collate_fn(self, examples):
        words = torch.tensor([ex[0] for ex in examples])
        contexts = torch.tensor([ex[1] for ex in examples])
        counts = torch.tensor([ex[2] for ex in examples])
        return (words, contexts, counts)
    

# 定义GloVe模型
class GloveModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(GloveModel, self).__init__()
        # 词向量及偏置向量
        self.w_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.w_biases = nn.Embedding(vocab_size, 1)
        # 上下文向量及偏置向量
        self.c_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.c_biases = nn.Embedding(vocab_size, 1)

    def forward_w(self, words):
        w_embeds = self.w_embeddings(words)
        w_biases = self.w_biases(words)
        return w_embeds, w_biases
    
    def forward_c(self, contexts):
        c_embeds = self.c_embeddings(contexts)
        c_biases = self.c_biases(contexts)
        return c_embeds, c_biases
    

# 模型训练
# 超参数设置
embedding_dim = 128
context_size = 2
batch_size = 1024
num_epoch = 5

#样本权重计算
m_max = 100
alpha = 0.75

# 构建GloVe数据集
corpus, vocab = load_reuters()
dataset = GloveDataset(corpus, vocab, context_size=context_size)
data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, collate_fn=dataset.collate_fn)

# 构建GloVe模型
model = GloveModel(len(vocab), embedding_dim)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

model.train()
for epoch in range(num_epoch):
    total_loss = 0
    for batch in tqdm(data_loader, desc="Training Epoch {}".format(epoch)):
        words, contexts, counts = batch
        
        # 提取批次内词
        word_embeds, word_biases = model.forward_w(words)
        context_embeds, context_biases = model.forward_c(contexts)

        # 回归目标值
        log_counts = torch.log(counts)

        # 样本权重
        weight__factor = torch.clamp(torch.pow(counts / m_max, alpha), max=1.0)
    
        optimizer.zero_grad()
    
        # 计算批次内每个样本的L2损失
        loss = (torch.sum(word_embeds * context_embeds, dim=1) + word_biases + context_biases - log_counts) ** 2
        # 样本加权损失
        wavg_loss = (weight__factor * loss).mean()
        wavg_loss.backward()
        optimizer.step()
        total_loss += wavg_loss.item()
    print("Epoch {} Loss: {}".format(epoch, total_loss))


# 保存词表以及训练得到的词向量
def save_pretrained(vocab, embeds, save_path):
    with open(save_path, "w") as writer:
        writer.write(f"{embeds.shape[0]} {embeds.shape[1]}\n")
        for idx, token in enumerate(vocab.get_itos()):
            vec = "".join([f"{x:.6f}" for x in embeds[idx]])
            # 每一行对应一个单词以及由空格分隔的词向量
            writer.write(f"{token} {vec}\n")


# 合并词向量矩阵和上下文向量矩阵，作为最终的预训练词向量
combined_embeds = model.w_embeddings.weight + model.c_embeddings.weight

# 保存预训练词向量
save_pretrained(vocab, combined_embeds, "glove.vec")
