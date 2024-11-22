import torch
print(torch.empty(2,3)) #创建一个shape为(2,3)的空张量，未初始化
print(torch.rand(2,3)) #创建一个shape为(2，3)的随机张量，每个值从[0,1]的均匀分布中产生
print(torch.zeros(2,3,dtype=torch.long)) #创建一个shape为(2，3)的0张量,数据类型为整数

print(torch.tensor([[1.0,3.4,11],[0.8,2.1,77.9]]))  #通过Python列表创建张量
print(torch.arange(10)) 

#张量的基本运算
x = torch.tensor([1,2,3])
y = torch.tensor([4,5,6])

#四则运算是按元素进行运算的
print(x*y)

print(x.dot(y))  #向量x与y的点积
print(x.sin())
print(x.exp())  #对x按元素求e^x

#张量的维
x = torch.tensor([[1,2,3],[4,5,6]],dtype=torch.float)
# print(x.mean())  #对所有元素求平均值
print(x.mean(dim=0,keepdim=True)) #按列求平均值,为保持输出的维度正确显示，设置keepdim=True
print(x.mean(dim=1,keepdim=True)) #按行求平均值

#拼接
y = torch.tensor([[7,8,9],[10,11,12]],dtype=torch.float)
print(torch.cat((x,y),dim=0))
print(torch.cat((x,y),dim=1))

复杂数学计算表达式
x = torch.tensor([2.])
y = torch.tensor([3.])
z = (x+y)*(y-2)
print(z)

自动微分
requires_grad=True要求pytroch对该张量进行梯度计算和跟踪
x = torch.tensor([2.],requires_grad=True)  
y = torch.tensor([3.],requires_grad=True)
z = (x+y)*(y-2)
#反向传播算法
z.backward()
print("z 对 x 的梯度：",x.grad)
print("z 对 x 的梯度：",y.grad)

#调整张量形状
#View函数，要求张量必须连续
x = torch.tensor([1,2,3,4,5,6])
print(x,x.shape)

print(x.view(2,3))  #将x的形状调整为（2，3）
print(x.view(3,2))  #将x的形状调整为（3，2）

x = torch.tensor([[1,2,3],[4,5,6]])
print(x)
print(x.transpose(0,1))  #交换第一维和第二维，transpose()只能交换两个维度

x = torch.tensor([[[1,2,3],[4,5,6]]])  #原本维度顺序：（1,2,3）
x = x.permute(2,0,1)  #permute可同时交换多个维度，交换后维度顺序（3，1，2）
print(x)

# #广播机制
x = torch.range(1,4).view(3,1)  
y = torch.range(4,6).view(1,2)

print(x)  # x 的维度为：（3，1）
print(y)  # y 的维度为：（1，2）
#执行运算前，通过广播机制将x，y扩展为（3，2）的张量
print(x+y) #将x的第1列复制到第2列,将y的第1行复制到第2、3行


#索引和切片
x = torch.arange(12).view(3,4)
print(x)
print(x[1,3])  #第二行第四列的元素
print(x[1:3])  #第二、三行元素
print(x[:,2])  #第三列全部元素
print(x[:,2:4]) #第三、四列元素
x[:,2:4] = 100
print(x)

#升维和降维:升维即在指定dim插入维度1；降维在不指定dim时，张量中所有形状为1的维度将被除去，指定dim后，则只操作在给定dim上
a = torch.tensor([1,2,3,4])
print(a.shape)

b = torch.unsqueeze(a,dim=0)
print(b,b.shape)

b = a.unsqueeze(dim=0)
print(b,b.shape)

c = b.squeeze()
print(c,c.shape)





