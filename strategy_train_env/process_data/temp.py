import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import os
import time
import kan

# data = pd.read_csv("./data/pulp_v2/period-7.csv")
# # 生成1-10之间的10个随机数
torch.manual_seed(42)
input_data = torch.tensor(range(0,48),dtype=torch.float32).view(-1,1)
labels = torch.rand((48,1),dtype=torch.float32)
# print(labels)

# 计算 labels 中所有值减去中位数后的期望
# median = torch.median(labels)
# absolute_deviation = torch.mean(torch.abs(labels - median))
# print(f'Absolute Median Deviation: {absolute_deviation.item():.4f}')
# 定义神经网络
variance = torch.var(labels)
print(f'Variance of labels: {variance.item():.4f}')
time.sleep(1)
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(1, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 初始化网络、损失函数和优化器
model = SimpleNet()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练网络
epochs = 500000
for epoch in range(epochs):
    # 前向传播
    outputs = model(input_data)
    loss = criterion(outputs, labels)

    # 反向传播和优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(loss.item())
    # if (epoch + 1) % 100 == 0:
    # print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

# 测试模型
# with torch.no_grad():
#     test_input = torch.tensor([[]], dtype=torch.float32)
#     prediction = model(test_input)
#     print(f'Prediction for input 10: {prediction.item():.4f}')