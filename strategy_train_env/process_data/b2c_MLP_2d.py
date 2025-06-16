import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import os
import time
import kan
from torch.utils.data import DataLoader, TensorDataset
data = pd.read_csv("./linear_solution/alpha_cpa.csv")
train_data = data[(data["budget"] <= 15000) & (data["budget"] % 10 ==0)]

input_data = torch.tensor(train_data["advertiserNumber"].values, dtype=torch.float32).view(-1, 1)
budget = torch.tensor(train_data["budget"].values, dtype=torch.float32).view(-1, 1)
budget = budget / 20000.0  # Normalize budget to [0, 1]

labels = torch.tensor(train_data["alpha"].values, dtype=torch.float32).view(-1, 1)
labels = labels / 300.0  # Normalize labels to [0, 1]
eval_data = data[(data["budget"]>15000)]

eval_input_data = torch.tensor(eval_data["advertiserNumber"].values, dtype=torch.float32).view(-1, 1)
eval_budget = torch.tensor(eval_data["budget"].values, dtype=torch.float32).view(-1, 1)
eval_budget = eval_budget / 20000.0  # Normalize budget to [0, 1]
eval_labels = torch.tensor(eval_data["alpha"].values, dtype=torch.float32).view(-1, 1)
eval_labels = eval_labels / 300.0  # Normalize labels to [0, 1]
# Create TensorDataset for training data
train_dataset = TensorDataset(input_data, budget, labels)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

# Create TensorDataset for evaluation data
eval_dataset = TensorDataset(eval_input_data, eval_budget, eval_labels)
eval_loader = DataLoader(eval_dataset, batch_size=128, shuffle=False)

# label = [[0, 0.012137764410940773, 33.98270216292129], [1, 0.010380899371583683, 55.22316800125827], [2, 0.011936386096267768, 23.669761560355845], [3, 0.011822148080930649, 64.58246195057586], [4, 0.008558708998402114, 51.69234978020519], [5, 0.011749582776835908, 59.57374559763331], [6, 0.013700708976205244, 2.721172598960626], [7, 0.01068911027404277, 31.187956506907557], [8, 0.007726010755608194, 63.547297320831255], [9, 0.006802288203705872, 49.94441556511027], [10, 0.006567695538295101, 93.4595420317068], [11, 0.00856753011207318, 74.62692663549474], [12, 0.010610358197459013, 71.91814667093865], [13, 0.006391709566825048, 62.574530583608215], [14, 0.007695830178747801, 47.921472042483636], [15, 0.008572635904869145, 83.45410932847807], [16, 0.0052797950934017, 60.75578927697089], [17, 0.00620870812775599, 58.058410291365455], [18, 0.005711911614478749, 53.77060090941103], [19, 0.006504791840206848, 55.92324671205796], [20, 0.007438736935817523, 87.16648563998181], [21, 0.004838200305769046, 38.09795920779896], [22, 0.007187299901668049, 56.9180482375531], [23, 0.00643901751728923, 55.585941748412544], [24, 0.006752127089515672, 53.244342243763406], [25, 0.009104966292029887, 72.41094562385393], [26, 0.007420819263293755, 74.91688752797792], [27, 0.006112323059330704, 88.12885865806214], [28, 0.00860766248979186, 40.70384968648327], [29, 0.00966163844786201, 48.441055501877265], [30, 0.00662818913355801, 76.31331086001423], [31, 0.007543209220709082, 45.649478722669], [32, 0.005760324588930232, 52.21782001699158], [33, 0.006066118498955066, 28.623691041967724], [34, 0.0050320567222360705, 71.02106239970402], [35, 0.008178361410803789, 21.0949316644269], [36, 0.005594594572777021, 35.49959899400297], [37, 0.0061578440455667734, 79.98708081441407], [38, 0.006061592077329764, 31.93643304172843], [39, 0.005524474938097181, 64.7280881350269], [40, 0.005663782961473944, 94.44257273259728], [41, 0.00755819642775651, 66.31773211010166], [42, 0.0056494772331444724, 50.76586687658163], [43, 0.00971268197520565, 87.72806529495115], [44, 0.006041939194883365, 78.05126690833504], [45, 0.004769427281115754, 72.700325572273], [46, 0.008085208543060324, 78.96789622020349], [47, 0.00774847843353657, 63.392466153859765]]

# 计算 labels 中所有值减去中位数后的期望
# median = torch.median(labels)
# absolute_deviation = torch.mean(torch.abs(labels - median))
# print(f'Absolute Median Deviation: {absolute_deviation.item():.4f}')
# 定义神经网络
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.id_embedding = nn.Embedding(48, 8) 
        self.k_linear = nn.Linear(8, 32)
        self.k_linear2 = nn.Linear(32, 16)
        self.k_linear3 = nn.Linear(16, 1)
        self.b_linear = nn.Linear(8, 32)
        self.b_linear2 = nn.Linear(32, 16)
        self.b_linear3 = nn.Linear(16, 1)
        self.c_linear = nn.Linear(8, 32)
        self.c_linear2 = nn.Linear(32, 16)
        self.c_linear3 = nn.Linear(16, 1)
        self.relu = nn.ReLU()
    def forward(self, x , budget):
        emb = self.id_embedding(x.long())
        k = self.relu(self.k_linear(emb))
        k = self.relu(self.k_linear2(k))
        k = self.k_linear3(k)
        b = self.relu(self.b_linear(emb))
        b = self.relu(self.b_linear2(b))
        b = self.b_linear3(b)
        c = self.relu(self.c_linear(emb))
        c = self.relu(self.c_linear2(c))
        c = self.c_linear3(c)
        k = k.view(-1, 1)
        b = b.view(-1, 1)
        c = c.view(-1, 1)
        return k * budget**2 + b * budget + c

# 初始化网络、损失函数和优化器
model = SimpleNet()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练网络
epochs = 30
for epoch in range(epochs):
    for batch in train_loader:
        input_data, budget, labels = batch
        # 前向传播
        outputs = model(input_data, budget)
        loss = criterion(outputs, labels)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.8f}')
    # 测试模型
    # with torch.no_grad():
    #     total_loss = 0.0
    #     for eval_batch in eval_loader:
    #         eval_input_data, eval_budget, eval_labels = eval_batch
    #         eval_outputs,_,_ = model(eval_input_data,eval_budget)
    #         eval_loss = criterion(eval_outputs, eval_labels)
    #         total_loss += eval_loss.item()
    #     eval_loss = total_loss / len(eval_loader)
    #     print(f'Epoch [{epoch+1}/{epochs}], Evaluation Loss: {eval_loss:.4f}')

budget = [[] for _ in range(48)]
prediction = [[] for _ in range(48)]
label = [[] for _ in range(48)]
for eval_batch in eval_loader:
    eval_input_data, eval_budget, eval_labels = eval_batch
    with torch.no_grad():
        eval_outputs = model(eval_input_data, eval_budget)
        for i in range(len(eval_input_data)):
            advertiser_id = int(eval_input_data[i].item())
            budget[advertiser_id].append(eval_budget[i].item())
            prediction[advertiser_id].append(eval_outputs[i].item())
            label[advertiser_id].append(eval_labels[i].item())
import matplotlib.pyplot as plt

plt.figure(figsize=(20, 15))
for i in range(48):
    plt.subplot(6, 8, i + 1)  # Create a 6x8 grid of subplots
    plt.scatter(budget[i], prediction[i], label="Prediction", color="blue")
    plt.scatter(budget[i], label[i], label="Label", color="red")
    plt.xlabel("Budget")
    plt.ylabel("Alpha")
    plt.title(f"Advertiser {i}")
    plt.legend()
    plt.grid(True)

plt.tight_layout()  # Adjust layout to prevent overlap
plt.savefig("all_advertisers_2d.png")
plt.close()



# 测试模型
# with torch.no_grad():
#     test_input = torch.tensor([[]], dtype=torch.float32)
#     prediction = model(test_input)
#     print(f'Prediction for input 10: {prediction.item():.4f}')