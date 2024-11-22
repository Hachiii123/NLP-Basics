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

# 定义SkipGram模型的数据构建与存取模块
class SkipGramDataset(Dataset):
    def __init__(self, corpus, vocab, context_size=2):
        self.data = []
        self.bos = vocab[BOS_TOKEN]
        self.eos = vocab[EOS_TOKEN]
        for sentence in tqdm(corpus, desc="Dateset Construction"):
            sentence = [self.bos] + sentence + [self.eos]
            for i in range (1, len(sentence)-1):
                # 模型输入：当前词
                w = sentence[i]
                # 模型输出: 窗口大小内的共现词对
                left_context_index = max(0, i-context_size)
                right_context_index = min(len(sentence)-1, i+context_size)
                context = sentence[left_context_index:i] + sentence[i+1:right_context_index+1]
                self.data.extend([(w,c) for c in context])


# 定义SkipGram模型
class SkipGramModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(SkipGramModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.output = nn.Linear(embedding_dim, vocab_size, bias=False)

    def forward(self, inputs):
        embeds = self.embedding(inputs)
        # 根据当前词，预测上下文
        output = self.output(embeds)
        log_probs = F.log_softmax(output, dim=1)
        return log_probs


# 基于负采样的Skip-Gram模型
# 数据
class SGNDataset(Dataset):
    def __init__(self, corpus, vocab, context_size=2, n_negatives=5, ns_dist=None):
        self.data = []
        self.bos = vocab[BOS_TOKEN]
        self.eos = vocab[EOS_TOKEN]
        self.pad = vocab[PAD_TOKEN]
        for sentence in tqdm(corpus, desc="Dateset Construction"):
            sentence = [self.bos] + sentence + [self.eos]
            for i in range (1, len(sentence)-1):
                # 模型输入：（v，context）;
                # 输出：0/1，表示context是否为负样本
                w = sentence[i]
                left_context_index = max(0, i-context_size)
                right_context_index = min(len(sentence)-1, i+context_size)
                context = sentence[left_context_index:i] + sentence[i+1:right_context_index+1]
                context += [self.pad] * (2*context_size-len(context))
                self.data.append((w, context))
            
            # 负样本数量
            self.n_negatives = n_negatives
            # 负样本分布: ns_dist=None时，均匀分布
            self.ns_dist = ns_dist if self.ns_dist else torch.ones(len(vocab))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, i):
        return self.data[i]
    
    def collate_fn(self, examples):
        words = torch.tensor([ex[0] for ex in examples], dtype=torch.long)
        contexts = torch.tensor([ex[1] for ex in examples], dtype=torch.long)
        batch_size, context_size = contexts.shape
        neg_contexts = []

        # 对批次内样本进行负采样
        for i in range(batch_size):
            # 保证负样本不包含当前上下文内样本
            ns_dist = self.ns_dist.index_fill(0, contexts[i], .0)
            neg_contexts.append(torch.multinomial(ns_dist, self.n_negatives * context_size, replacement=True))
            neg_contexts = torch.stack(neg_contexts, dim=0)
        
        return words, contexts, neg_contexts
    

# 模型
class SGNModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(SGNModel, self).__init__()
        # 词向量
        self.w_embedding = nn.Embedding(vocab_size, embedding_dim)
        # 上下文向量
        self.c_embedding = nn.Embedding(vocab_size, embedding_dim)
        
    def forward_w(self, words):
        w_embeds = self.w_embedding(words)
        return w_embeds
    
    def forward_c(self, contexts):
        c_embeds = self.c_embedding(contexts)
        return c_embeds

# 训练
# 从给定预料中统计Unigram出现次数并计算概率分布
def get_unigram_distribution(corpus, vocab_size):
    token_counts = torch.tensor([0] * vocab_size)
    total_count = 0
    for sentence in corpus:
        total_count += len(sentence)
        for token in sentence:
            token_counts[token] += 1
    
    unigram_dist = torch.div(token_counts, total_count)
    return unigram_dist

# 保存词表以及训练得到的词向量
def save_pretrained(vocab, embeds, save_path):
    with open(save_path, "w") as writer:
        writer.write(f"{embeds.shape[0]} {embeds.shape[1]}\n")
        for idx, token in enumerate(vocab.get_itos()):
            vec = "".join([f"{x:.6f}" for x in embeds[idx]])
            # 每一行对应一个单词以及由空格分隔的词向量
            writer.write(f"{token} {vec}\n")

# 具体训练过程
# 超参数
embedding_dim = 128
context_size = 3
batch_size = 1024
n_negatives = 5  # 负样本数量
num_epoch = 5

corpus, vocab = load_reuters()
unigram_dist = get_unigram_distribution(corpus, len(vocab))
# 计算负采样分布
negative_sampling_dist = unigram_dist ** 0.75
negative_sampling_dist = negative_sampling_dist / torch.sum(negative_sampling_dist)

# 构建SGNS训练数据集
dataset = SGNDataset(corpus, vocab, context_size=context_size, n_negatives=n_negatives, ns_dist=negative_sampling_dist)
data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, collate_fn=dataset.collate_fn)

# 构建SGNS模型
model = SGNModel(len(vocab), embedding_dim)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 训练过程
for epoch in range(num_epoch):
    total_loss = 0
    for batch in tqdm(data_loader, desc="Training Epoch {}".format(epoch)):
        words, contexts, neg_contexts = batch
        optimizer.zero_grad()
        batch_size = words.shape(0)
        word_embeds = model.forward_w(words).unsqueeze(dim=2)
        context_embeds = model.forward_c(contexts)
        neg_context_embeds = model.forward_c(neg_contexts)
        context_loss = F.logsigmoid(torch.bmm(context_embeds, word_embeds).squeeze(dim=2))
        # 负样本的分类对数似然
        neg_context_loss = F.logsigmoid(-torch.bmm(neg_context_embeds, word_embeds).squeeze(dim=2).neg())
        neg_context_loss = neg_context_loss.view(batch_size, -1, n_negatives).sum(dim=2)
        neg_context_loss = neg_context_loss.mean(dim=1)
        # 总体损失
        loss = -(context_loss + neg_context_loss).mean()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print("Epoch {} Loss: {}".format(epoch, total_loss))

# 合并词向量与上下文向量矩阵
combined_embeds = model.w_embedding.weight + model.c_embedding.weight
# 将词向量保存到文件
save_pretrained(vocab, combined_embeds.data, "ffnnlm.vec")
