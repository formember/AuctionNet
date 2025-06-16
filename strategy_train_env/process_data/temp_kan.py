import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import os
import time
import kan

# data = pd.read_csv("./data/pulp_v2/period-7.csv")
# 生成1-10之间的10个随机数
torch.manual_seed(42)
input_data = torch.rand((64,1))
labels = torch.rand((64,1))

# 计算 labels 中所有值减去中位数后的期望
# median = torch.median(labels)
# absolute_deviation = torch.mean(torch.abs(labels - median))
# print(f'Absolute Median Deviation: {absolute_deviation.item():.4f}')
# 定义神经网络


# 初始化网络、损失函数和优化器
model = kan.KAN(width=[1,8,8,1],grid = 3, k=3,seed=42,device='cpu')
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
model(input_data)
# model.plot()
# 训练网络
epochs = 5000
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