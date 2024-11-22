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

# 定义CBOW模型的数据构建与存取模块
class CbowDataset(Dataset):
    def __init__(self, corpus, vocab, context_size=2):
        self.data = []
        self.bos = vocab[BOS_TOKEN]
        self.eos = vocab[EOS_TOKEN]
        for sentence in tqdm(corpus, desc="Dataset Construction"):
            sentence = [self.bos] + sentence + [self.eos]
            if len(sentence) < context_size*2+1:
                continue
            for i in range(context_size, len(sentence) - context_size):
                # 构建中心词的上下文
                context = sentence[i-context_size:i] + sentence[i+1:i+context_size+1]
                target = sentence[i]
                self.data.append((context, target))


# 定义CBOW模型
class CbowModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(CbowModel, self).__init__()
        # 词向量层
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # 输出层
        self.output = nn.Linear(embedding_dim, vocab_size, bias=False)

    def forward(self, inputs):
        embeds = self.embedding(inputs)
        # 计算上下文向量：取平均
        hidden = embeds.mean(dim=1)
        output = self.output(hidden)
        log_probs = F.log_softmax(output, dim=1)
        return log_probs

