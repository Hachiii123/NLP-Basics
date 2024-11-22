import spacy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchtext.vocab import build_vocab_from_iterator
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence




# 加载宾州树库的词性标注语料库
def load_treebank():
    from nltk.corpus import treebank

    # 获取标注好的句子和标签
    sents, postags = zip(*(zip(*sent) for sent in treebank.tagged_sents()))

    # 使用 build_vocab_from_iterator 来构建词汇表
    word_vocab = build_vocab_from_iterator((sentence for sentence in sents), specials=["<pad>"])
    tag_vocab = build_vocab_from_iterator((tags for tags in postags), specials=["<pad>"])

    # 设置为默认的未知词索引
    word_vocab.set_default_index(word_vocab["<unk>"])
    tag_vocab.set_default_index(tag_vocab["<unk>"])

    # 前3000条作为训练数据
    train_data = [(torch.tensor([word_vocab[token] for token in sentence]),
                   torch.tensor([tag_vocab[tag] for tag in tags])) for sentence, tags in zip(sents[:3000], postags[:3000])]
    # 其余作为测试数据
    test_data = [(torch.tensor([word_vocab[token] for token in sentence]),
                  torch.tensor([tag_vocab[tag] for tag in tags])) for sentence, tags in zip(sents[3000:], postags[3000:])]


    
    return train_data, test_data, word_vocab, tag_vocab

# 批处理函数：用于将不同长度的句子组成批次，并进行填充和打包
def collate_fn(examples, vocab, tag_vocab):
    # 每个输入序列的长度
    lengths = torch.tensor([len(ex[0]) for ex in examples])

    # 将输入和目标转换为张量
    inputs = torch.tensor(ex[0] for ex in examples)
    targets = torch.tensor(ex[1] for ex in examples)

    # 对输入和输出序列进行填充
    inputs = pad_sequence(inputs, batch_first=True, padding_value=vocab["<pad>"])
    targets = pad_sequence(targets, batch_first=True, padding_value=tag_vocab["<pad>"])

    return inputs, lengths, targets, inputs != vocab["<pad>"]

# 定义 LSTM 模型
class LSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes):
        super(LSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.output = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, inputs, lengths):
        # 1.将输入的token转换为嵌入表示
        embeddings = self.embedding(inputs)

        # 2.对嵌入序列进行打包，即还原成结尾经过补齐的多个序列
        # 打包：将每个序列中实际的词或标记打包成一个紧凑的数据结构，并保存每个序列的长度。
        # LSTM 接收到打包后的数据时，能自动忽略填充位置，仅对有效部分进行前向传播和梯度计算
        x_pack = pack_padded_sequence(embeddings, lengths, batch_first=True, enforce_sorted=False)
        
        # 3.LSTM 前向传播
        packed_output, (hn, cn) = self.lstm(x_pack)
        
        # 4.解包，获得补齐后的序列
        # 解包：打包的序列通过 LSTM 后，返回的也是一个紧凑的、打包的输出，包含每个时间步上的输出信息。
        # 为了方便后续处理（如计算损失或送入其他层），我们需要将打包数据重新解包回补齐的形状
        unpacked_output, _ = pad_packed_sequence(packed_output, batch_first=True)

        # 5.输出层计算，输出最后一个时刻的隐藏状态
        outputs = self.output(unpacked_output)

        # 6.使用 log_softmax得到每个标记的概率分布
        log_probs = F.log_softmax(outputs, dim=-1)
        
        return log_probs
    

# 1.加载数据并构造词汇表
train_data, test_data, word_vocab, tag_vocab = load_treebank()

# 2.定义模型超参数
vocab_size = len(train_data[0][0])
embedding_dim = 100
hidden_dim = 64
num_classes = len(tag_vocab)

# 3.实例化模型
model = LSTM(vocab_size, embedding_dim, hidden_dim, num_classes)

# 4.定义DataLoader
train_loader = DataLoader(train_data, batch_size=32, shuffle=True, collate_fn=lambda x: collate_fn(x, word_vocab, tag_vocab))
test_loader = DataLoader(test_data, batch_size=32, shuffle=False, collate_fn=lambda x: collate_fn(x, word_vocab, tag_vocab))

# 5. 获取一个批次数据并进行前向传播
for inputs, lengths, targets, mask in train_loader:
    # 前向传播
    outputs = model(inputs, lengths)
    
    # 打印输出的形状
    print("输出形状:", outputs.shape)  # 应为 (batch_size, max_seq_len, num_classes)
    print("输出示例:", outputs[0])     # 输出第一个句子的词性标签概率分布
    break  # 只测试一个批次

