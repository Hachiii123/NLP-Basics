import torch
import math
from torch import nn
from torch.nn import functional as F
from torch.nn import RNN
from torch.nn import LSTM

# # 直接调用函数实现RNN
# rnn = RNN(input_size=4, hidden_size=5, batch_first=True)   # 输入维度为4，输出维度为5
# inputs = torch.randn(2, 3, 4)
# print(inputs)
# outputs, hidden = rnn(inputs)
# print(outputs)
# print(hidden)

# new_rnn = RNN(input_size=4, hidden_size=5, batch_first=True, bidirectional=False, num_layers=3)  # 隐藏层层数为3
# outputs, hidden = new_rnn(inputs)
# print(outputs)
# print(hidden)

# # 直接调用函数实现LSTM
# lstm = LSTM(input_size=4, hidden_size=5, batch_first=True)
# outputs, (h_n, c_n) = lstm(inputs)
# print(h_n)
# print(c_n)


class ScaleDotProductAttention(nn.Module):
    def __init__(self, d_k):
        super(ScaleDotProductAttention, self).__init__()
        self.d_k = d_k

    def forward(self, Q, K, V, mask=None):
        # 1.计算 Q 和 V 的点击，除以缩放系数 sqrt(d_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # 2.如果 mask 存在，则将 mask 乘以一个很小的负无穷，使得 mask 对应的位置的得分为负无穷
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # 3.计算 softmax 值，获得注意力权重
        attention_weights = F.softmax(scores, dim=-1)

        # 4.使用注意力权重对 V 进行加权求和，得到输出
        output = torch.matmul(attention_weights, V)

        return output, attention_weights

# 定义完整的注意力网络模块
class AttentionNetwork(nn.Module):
    def __init__(self, input_dim, d_k, d_v, d_model):
        super(AttentionNetwork, self).__init__()

        # 定义Query、Key、Value矩阵（对输入向量 x 进行线性变换，将其映射到 Query、Key 和 Value 空间中）
        # input_dim是输入维度，d_k是 Query 和 Key 的目标维度，d_v 是 Value 的目标维度
        self.query = nn.Linear(input_dim, d_k)
        self.key = nn.Linear(input_dim, d_k)
        self.value = nn.Linear(input_dim, d_v)

        # 定义缩放点积注意力层
        self.attention = ScaleDotProductAttention(d_k)

        # 定义一个线性层将注意力模块的输出维度映射为模型维度,确保注意力网络的输出与模型的整体结构兼容，便于与其他模块进行叠加
        self.fc = nn.Linear(d_v, d_model)
    
    def forward(self, x, mask=None):
        # 计算Q、K、V
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        # 计算注意力输出
        attention_output, attention_weights = self.attention(Q, K, V, mask=mask)

        # 通过全连接层调整维度
        output = self.fc(attention_output)

        return output, attention_weights
    

input = torch.rand(2, 4, 8)

# 实例化注意力网络
attention_network = AttentionNetwork(input_dim=8, d_k=16, d_v=16, d_model=8)
output, attention_weights = attention_network(input)
print("Output shape:", output.shape)  # 应为 (2, 4, 8)
print("Attention weights shape:", attention_weights.shape)  # 应为 (2, 4, 4)
print("Output:", output)
