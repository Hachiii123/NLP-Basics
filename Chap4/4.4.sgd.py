import torch
import torch.nn as nn
import torch.nn.functional as F

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
        # print("inputs:",inputs)
        hidden = self.linear1(inputs)
        # print("hidden:",hidden)
        activation = self.activate(hidden)
        # print("activation:",activation)
        outputs = self.linear(activation)
        # print("outputs:",outputs)
        # 获得每个输入属于某一类别的概率(Softmax)，再取对数
        # 取对数的目的是避免计算 Softmax 时产生数值溢出
        log_probs = F.log_softmax(outputs, dim=1)  # 输出各个类别的概率值且总和为1
        return log_probs

# mlp = MLP(input_size=4, hidden_size=5, num_classes=2)
# inputs = torch.randn(3, 4)
# probs = mlp(inputs)
# print(probs)

# 异或问题的四个输入
x_train = torch.tensor([[0.0, 0.0],[0.0, 1.0],[1.0, 0.0],[1.0, 1.0]])
# 每个输入对应的输出类别
y_train = torch.tensor([0, 1, 1, 0])

mlp = MLP(input_size=2, hidden_size=5, num_classes=2)
# 调用负对数似然损失
critic = nn.NLLLoss()
# 采用随机梯度下降法
optimizer = torch.optim.SGD(mlp.parameters(), lr=0.05)

for epoch in range(500):
    # 调用模型进行预测
    y_pred = mlp(x_train)
    # 计算损失值
    loss = critic(y_pred, y_train)
    # 调用反向传播前先将梯度置为0
    optimizer.zero_grad()
    # 反向传播计算参数梯度
    loss.backward()
    # 更新参数
    optimizer.step()

print("Parameters: ")
for name, param in mlp.named_parameters():
    print(name, param.data)

y_pred = mlp(x_train)
print("Predictions: ", y_pred.argmax(dim=1))


