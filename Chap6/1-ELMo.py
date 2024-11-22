import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from tqdm import tqdm
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import Dataset
from collections import defaultdict
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence


BOS_TOKEN='<bos>'  
EOS_TOKEN='<eos>'
PAD_TOKEN='<pad>'
EOW_TOKEN='<eow>'
BOW_TOKEN='<bow>'


def load_corpus(path, max_tok_len=None, max_seq_len=None):
    """
    从生文本预料中加载数据并构建词表
    max_tok_len: 词的长度（字符数目）上限
    max_seq_len: 序列长度（词数）上限
    """
    text = []
    charset = {BOS_TOKEN, EOS_TOKEN, PAD_TOKEN, EOW_TOKEN}
    with open(path, "r") as f:
        for line in tqdm(f):
            tokens = line.rstrip().split(" ")
            # 截断长序列
            if max_seq_len is not None and len(tokens) + 2 > max_seq_len:
                tokens = line[:max_seq_len-2]
            sent = [BOS_TOKEN]

            # 截断字符数目过多的词
            for token in tokens:
                if max_tok_len is not None and len(token) + 2 > max_tok_len:
                    token = token[:max_tok_len-2]
                sent.append(token)
                for ch in token:
                    charset.add(ch)
            sent.append(EOS_TOKEN)
            text.append(sent)
    
    special_tokens = [BOS_TOKEN, EOS_TOKEN, PAD_TOKEN]
    # 构建词汇表迭代器
    def yield_tokens(data_iter):
        for sentence in data_iter:
            yield sentence
    
    # 构建词表
    vocab_w = build_vocab_from_iterator(yield_tokens(text), min_freq=2, specials=special_tokens)
    # 构建字符级词表
    vocab_c = build_vocab_from_iterator(tokens=list(charset))

    # 构建词级别预料
    corpus_w = [vocab_w.convert_tokens_to_ids(sent) for sent in text]
    # 构建字符级预料
    corpus_c = []
    bow = vocab_c[BOW_TOKEN]
    eow = vocab_c[EOW_TOKEN]
    for i,sent in enumerate(text):
        sent_c = [bow]
        for token in sent:
            if token == BOS_TOKEN or token == EOS_TOKEN:
                token_c = [bow, vocab_c[token], eow]
            else:
                token_c = [bow] + vocab_c.convert_tokens_to_ids(token)+ [eow]
            
            sent_c.append(token_c)
        corpus_c.append(sent_c)
    
    return corpus_w, corpus_c, vocab_w, 



# 构建双向语言模型的数据类BiLMDatset
class BiLMDataset(Dataset):
    def __init__(self, corpus_w, corpus_c, vocab_w, vocab_c):
        super(BiLMDataset, self).__init__()
        self.pad_w = vocab_w[PAD_TOKEN]
        self.pad_c = vocab_c[PAD_TOKEN]

        self.data=[]
        for sent_w,sent_c in zip(corpus_w, corpus_c):
            self.data.append(sent_w, sent_c)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    def collate_fn(self, examples):
        # 当前批次中个样本序列的长度
        seq_lens = torch.LongTensor([len(ex[0]) for ex in examples])

        # 词级别输入：batch_size * max_seq_len
        inputs_w = [torch.tensor(ex[0] for ex in examples)]
        # 对batch内样本进行长度补齐
        inputs_w = pad_sequence(inputs_w, batch_first=True, padding_values=self.pad_w)

        # 计算当前批次中的最大序列长度以及单词的最大字符数目
        batch_size , max_seq_len = inputs_w.shape
        max_tok_len = max([max([len(tok) for tok in ex[1]]) for ex in examples])

        # 字符级输入：batch_size * max_seq_len * max_tok_len
        inputs_c = torch.LongTensor(batch_size, max_seq_len, max_tok_len).fill_(self.pad_c)

        for i,(sent_w, setn_c) in enumerate(examples):
            for j,tok in enumerate(setn_c):
                inputs_c[i,j,:len(tok)] = torch.LongTensor(tok)
        
        targets_fw = torch.LongTensor(inputs_w.shape).fill(self.pad_w)
        targets_bw = torch.LongTensor(inputs_w.shape).fill(self.pad_w)
        for i,(sent_w,sent_c) in enumerate(inputs_w):
            targets_fw[i][:len(sent_w)-1] = torch.LongTensor(sent_w[1:])
            targets_bw[i][1:len(sent_w)] = torch.LongTensor(sent_w[:len(sent_w)-1])
        
        return inputs_w, inputs_c, seq_lens, targets_fw, targets_bw

# 输入表示层依赖的 Highway 网络
class Highway(nn.Module):
    def __init__(self, input_dim, num_layers=1, activation=F.relu):
        super(Highway, self).__init__()
        self.input_dim = input_dim
        self.layers = torch.nn.ModuleList(
            nn.Linear(input_dim, input_dim * 2) for _ in range(num_layers)
        )
        self.activate = activation
        for layer in self.layers:
            layer.bias[input_dim].data.fill_(1)

    def forward(self, inputs):
        curr_inputs = inputs
        for layer in self.layers:
            projected_inputs = layer(curr_inputs)

            # 输出向量的前半部分作为当前隐含层的输出
            hidden = self.activate(projected_inputs[:, 0: self.input_dim])

            # 输出向量的后半部分用于计算门控向量
            gate = torch.sigmoid(projected_inputs[:, self.input_dim:])

            # 线性插值
            curr_inputs = gate * curr_inputs + (1 - gate) * hidden
        
        return curr_inputs
    

# 基于字符卷积的词表示层
class ConvTokenEmbedder(nn.Module):
    """
    vocab_c: 字符级词表
    char_embedding_dim: 字符向量维度
    char_conv_filters: 字符卷积核大小
    num_highways: highway网络层数
    """
    def __init__(self, vocab_c, char_embedding_dim, char_conv_filters, num_highways, output_dim, pad="<pad>"):
        super(ConvTokenEmbedder, self).__init__()
        self.vocab_c = vocab_c
        self.char_embedding_dim = nn.Embedding(
            len(vocab_c), 
            char_embedding_dim, 
            padding_idx=vocab_c[pad]
        )
        self.char_embeddings.data.uniform(-0.25, 0.25)

        # 为每个卷积核分别构建卷积神经网络
        self.convolutions = nn.ModuleList()
        for kernel_size, out_channels in char_conv_filters:
            conv = torch.nn.Conv1d(
                in_channels = char_embedding_dim,    # 向量维度为作为通道数
                out_channels = out_channels,        # 输出通道数
                kernel_size = kernel_size,          # 卷积核大小
                bias=True
            )
            self.convolutions.append(conv)
        
        # 构建highway网络
        self.num_filters = sum(f[1] for f in char_conv_filters)
        self.num_highways = num_highways
        self.highways = Highway(self.num_filters, self.num_highways, activation=F.relu)

        # ELMo 的向量表示是多层表示的插值，因此须要保证各层向量表示的维度一致
        self.projection = nn.Linear(self.num_filters, output_dim, bias=True)

    def forward(self, inputs):
        batch_size, seq_len, token_len = inputs.shape
        inputs = inputs.view(batch_size * seq_len, -1)
        char_embeds = self.char_embeddings(inputs)
        char_embeds = char_embeds.transpose(1, 2)

        conv_hiddens = []
        for i in range(len(self.convolutions)):
            conv_hidden = self.convolutions[i](char_embeds)
            conv_hidden, _ = torch.max(conv_hidden, dim=-1)
            conv_hiddens = F.relu(conv_hidden)
            conv_hiddens.append(conv_hidden)
        
        # 将不同卷积核下得到的向量表示进行拼接
        token_embeds = torch.cat(conv_hiddens, dim=-1)
        token_embeds = self.highways(token_embeds)
        token_embeds = self.projection(token_embeds)
        token_embeds = token_embeds.view(batch_size, seq_len, -1)

        return token_embeds
    
# 创建双向 LSTM 编码器，获取序列每一时刻、每一层的前向表示和后向表示
class ELMoLstmEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(ELMoLstmEncoder, self).__init__()
        self.projection_dim = input_dim
        self.num_layers = num_layers

        # 前向 LSTM (多层)
        self.forward_layers = nn.ModuleList()
        self.forward_projections = nn.ModuleList()
        
        # 后向 LSTM (多层)
        self.backward_layers = nn.ModuleList()
        self.backward_projections = nn.ModuleList()

        lstm_input_dim = input_dim

        for _ in range(num_layers):
            # 单层前向 LSTM 层及投射层
            forward_layer = nn.LSTM(lstm_input_dim, hidden_dim, num_layers=1, batch_first=True)
            forward_projection = nn.Linear(hidden_dim, self.projection_dim, bias=True)

            # 单层后向 LSTM 层及投射层
            backward_layer = nn.LSTM(lstm_input_dim, hidden_dim, num_layers=1, batch_first=True)
            backward_projection = nn.Linear(hidden_dim, self.projection_dim, bias=True)

            lstm_input_dim = self.projection_dim

            self.forward_layers.append(forward_layer)
            self.forward_projections.append(forward_projection)
            self.backward_layers.append(backward_layer)
            self.backward_projections.append(backward_projection)
    
    def forward(self, inputs, lengths):
        batch_size, seq_len, input_dim = inputs.shape
        # 根据前向输入批次以及序列长度信息，构建后向输入批次
        rev_idx = torch.arange(seq_len).unsqueeze(0).repeat(batch_size, 1)
        for i in range(lengths.shape[0]):
            rev_idx[i,:lengths[i]] = torch.arange(lengths[i]-1, -1, -1)
        rev_idx = rev_idx.unsqueeze(2).expand_as(inputs)
        rev_inputs = inputs.gather(1, rev_idx)

        # 前向、后向 LSTM 输入
        forward_inputs, backward_inputs = inputs, rev_inputs
        # 保存每层前向、后向隐含层状态
        stacked_forward_states, stacked_backward_states = [], []

        for layer_index in range(self.num_layers):
            packed_forward_inputs = pack_padded_sequence(
                forward_inputs, lengths, batch_first=True, enforce_srtoed=False
            )
            packed_backward_inputs = pack_padded_sequence(
                backward_inputs, lengths, batch_first=True, enforce_sorted=False
            )

            # 计算前向 LSTM 
            forward_layer = self.forward_layers[layer_index]
            packed_forward, _ = forward_layer(packed_forward_inputs)
            forward = pad_packed_sequence(packed_forward, batch_first=True)[0]
            forward = self.forward_projections[layer_index](forward)
            stacked_forward_states.append(forward)

            # 计算后向 LSTM
            backward_layer = self.backward_layers[layer_index]
            packed_backward, _ = backward_layer(packed_backward_inputs)
            backward = pad_packed_sequence(packed_backward, batch_first=True)[0]
            backward = self.backward_projections[layer_index](backward)
            
            # 恢复至序列的原始顺序
            stacked_backward_states.append(backward.gather(1, rev_idx))
        
        return stacked_forward_states, stacked_backward_states
    

# 超参数
configs = {
    'max_tok_len':50,
    'train_file': './train.txt',   # 经过预处理的训练语料文本，每一行是一段独立文本
    'model_path':'./elmo_bilm',    # 模型保存路径
    'char_embedding_dim':50,       # 字符向量维度
    'char_conv_filters':[[1,32], [2,32], [3,64], [4,128], [5,256], [6,512], [7,1024]],    # 字符卷积核大小: [宽度，输出维度]
    'num_highways':2,              # highway网络层数
    'projection_dim':512,
    'hidden_dim':4096,
    'num_layers':2,
    'batch_size':32,
    'dropout':32,
    'learning_rate':0.0004,
    'clip_grad':5,
    'num_epochs':10,
}

# 创建双向语言模型
class BiLM(nn.Module):
    def __init__(self, configs, vocab_w, vocab_c):
        super(BiLM, self).__init__()
        self.dropout = configs['dropout']
        self.num_classes = len(vocab_w)  # 输出层目标维度为词表大小
        self.token_embedder = ConvTokenEmbedder(
            vocab_c,
            configs['char_embedding_dim'],
            configs['char_conv_filters'],
            configs['num_highways'],
            configs['projection_dim']
        )

        # ELMo LSTM 编码器
        self.encoder = ELMoLstmEncoder(
            configs['projection_dim'],
            configs['hidden_dim'],
            configs['num_layers']
        )
        
        # 分类器（输出层）
        self.classifier = nn.Linear(configs['projection_dim'] * 2, self.num_classes)

    def forward(self, inputs, lengths):
        # 词级别输入
        token_embeds = self.token_embedder(inputs)
        token_embeds = F.dropout(token_embeds, self.dropout, self.training)
        forward, backward = self.encoder(token_embeds, lengths)
        
        return self.classifier(forward[-1], self.classifier(backward[-1]))
    

    # 保存编码器参数
    def save_pretrained(self, path):
        os.makedirs(path, exist_ok=True)
        torch.save(self.token_embedder.state_dict(), os.path.join(path, 'token_embedder.pth'))
        torch.save(self.encoder.state_dict(), os.path.join(path, 'token_embedder.pth'))


# 训练
corpus_w, corpus_c, vocab_w, vocab_c = load_corpus(configs['train_file'])
train_data = BiLMDataset(corpus_w, corpus_c, vocab_w, vocab_c)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=configs['batch_size'])

# 交叉熵损失
criterion = nn.CrossEntropyLoss(
    ignore_index = vocab_w[PAD_TOKEN],
    reduction='sum'
)

# 创建模型
model = BiLM(configs, vocab_w, vocab_c)
optimizer = torch.optim.Adam(
    filter(lambda x: x.requires_grad, model.parameters()),
    lr=configs['learning_rate']
)

# 训练过程
model.train()
for epoch in range(configs['num_epochs']):
    total_loss = 0
    total_tags = 0
    for batch in tqdm(train_loader, desc="Training Epoch {epoch}"):
        inputs_w, inputs_c, seq_lens, targets_fw, targets_bw = batch

        optimizer.zero_grad()
        outputs_fw, outputs_bw = model(inputs_w, seq_lens)
        
        # 前向语言模型损失
        loss_fw = criterion = (
            outputs_fw.view(-1, outputs_fw.shape[-1]), 
            targets_fw.view(-1)
        )

        # 后向语言模型损失
        loss_bw = criterion = (
            outputs_bw.view(-1, outputs_bw.shape[-1]), 
            targets_bw.view(-1)
        )

        # 总损失
        loss = (loss_fw + loss_bw) / 2.0
        loss.backward()
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), configs['clip_grad'])
        optimizer.step()

        total_loss += loss.item()
        total_tags += seq_lens.sum().item()

        # 以前向语言模型的困惑度（PPL）作为模型的当前性能指标
        train_ppl = np.exp(total_loss / total_tags)
        print("Train PPL: {train_ppl:.2f}")

def save_vocab(vocab, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        for word in vocab:
            f.write(word + '\n')


# 保存模型编码器参数
model.save_pretrained(configs['model_path'])
# 保存超参数
json.dump(configs, open(os.path.join(configs['model_path'], 'config.json'), 'w'))
# 保存词表
save_vocab(vocab_w, os.path.join(configs['model_path'], 'vocab_w.dic'))
save_vocab(vocab_c, os.path.join(configs['model_path'], 'vocab_c.dic'))
         




