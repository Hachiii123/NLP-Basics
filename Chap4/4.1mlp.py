import torch
from torch import nn
from torch.nn import functional as F

# linear = nn.Linear(10, 2)  # 10 is the input size and 2 is the output size
# inputs = torch.randn(5, 10)  # 5 is the batch size
# print(inputs)
# outputs = linear(inputs)
# print(outputs)

# activation = F.sigmoid(outputs)         # sigmoid：输出某一个类别的概率值（0-1），适用于二元分类
# print(activation)
# activation = F.softmax(outputs, dim=1)  # softmax：输出各个类别的概率值且总和为1，dim=1 means we apply softmax on the second dimension (the output dimension)
# print(activation)
# activation = F.relu(outputs)            # relu：f(x)=max(0, x)，非线性激活函数
# print(activation)

# 自定义神经网络模型
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MLP, self).__init__()
        
        # 线性变换：输入层 -> 隐藏层
        self.linear1 = nn.Linear(input_size, hidden_size)

        # 激活函数
        self.activate = F.relu

        # 线性变换：隐藏层 -> 输出层
        self.linear = nn.Linear(hidden_size, num_classes)
    
    # 前向传播过程
    def forward(self, inputs):
        print("inputs:",inputs)
        hidden = self.linear1(inputs)
        print("hidden:",hidden)
        activation = self.activate(hidden)
        print("activation:",activation)
        outputs = self.linear(activation)
        print("outputs:",outputs)
        probs = F.softmax(outputs, dim=1)  # 输出各个类别的概率值且总和为1
        return probs

mlp = MLP(input_size=4, hidden_size=5, num_classes=2)
inputs = torch.randn(3, 4)
probs = mlp(inputs)
print(probs)