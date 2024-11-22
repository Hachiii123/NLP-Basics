import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, Conv1d, MaxPool1d


# conv1 = nn.Conv1d(5, 2, 4)  # input_channels, output_channels, kernel_size
# conv2 = nn.Conv1d(5, 2, 3)

# inputs = torch.randn(2, 5, 6) # batch_size, input_channels, sequence_length
# print("inputs:", inputs)

# outputs1 = conv1(inputs)
# outputs2 = conv2(inputs)
# print("outputs1:", outputs1)
# print("outputs2:", outputs2)

# pool1 = MaxPool1d(3)  # kernel_size=3
# pool2 = MaxPool1d(4)
# outputs_pool1 = pool1(outputs1)  # apply max pooling to the output of the first convolutional layer
# outputs_pool2 = pool2(outputs2)
# print("outputs_pool1:", outputs_pool1)
# print("outputs_pool2:", outputs_pool2)

# outputs_pool1 = F.max_pool1d(outputs1, kernel_size=outputs_pool1.shape[2])  # apply max pooling to the output of the first convolutional layer, using the last dimension of the output as the kernel_size
# outputs_pool2 = F.max_pool1d(outputs2, kernel_size=outputs_pool2.shape[2])
# print("outputs_pool1:", outputs_pool1)
# print("outputs_pool2:", outputs_pool2)

# outputs_pool_squeeze1 = outputs_pool1.squeeze(dim=2)  # remove the last dimension of the output
# print("outputs_pool_squeeze1:", outputs_pool_squeeze1)
# outputs_pool_squeeze2 = outputs_pool2.squeeze(dim=2)
# print("outputs_pool_squeeze2:", outputs_pool_squeeze2)
# outputs_pool = torch.cat([outputs_pool_squeeze1, outputs_pool_squeeze2], dim=1)  # concatenate the two squeezed outputs along the channel dimension
# print("outputs_pool:", outputs_pool)


# linear = Linear(4,2)
# outputs_linear = Linear(outputs_pool)
# print("outputs_linear:", outputs_linear)


class CNN(nn.Module):
    def __init__(self, input_dim, output_dim, num_classes, kenerl_size):
        super(CNN, self).__init__()
    
        self.conv = nn.Conv1d(input_dim, output_dim, kenerl_size)
        self.pool = F.max_pool1d
        self.linear = nn.Linear(output_dim, num_classes)

    def forward(self, inputs):
        print("inputs:", inputs)

        conv = self.conv(inputs)
        print("conv:", conv)
        pool = self.pool(conv, kernel_size=conv.shape[2])
        print("pool:", pool)
        pool_squeeze = pool.squeeze(dim=2)
        outputs = self.linear(pool_squeeze)
        print("outputs:", outputs)
        return outputs

cnn = CNN(5, 2, 2, 4)  # 
inputs = torch.rand(2,5,6)  # batch_size=2, input_dim=5, sequence_length=6
probs = cnn(inputs)
print("outputs:", probs)

