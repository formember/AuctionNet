import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import os
import time
import kan
from torch.utils.data import DataLoader, TensorDataset

# data = pd.read_csv("./linear_solution/alpha_cpa.csv")
# train_data = data[(data["budget"]<=3000) & (data["budget"] % 10 == 0)]
# input_data = torch.tensor(train_data["advertiserNumber"].values, dtype=torch.float32).view(-1, 1)
# budget = torch.tensor(train_data["budget"].values, dtype=torch.float32).view(-1, 1)
# labels = torch.tensor(train_data["alpha"].values, dtype=torch.float32).view(-1, 1)

# eval_data = data[data["budget"]>18000 and data["budget"] % 10 == 0]
# eval_input_data = torch.tensor(eval_data["advertiserNumber"].values, dtype=torch.float32).view(-1, 1)
# eval_budget = torch.tensor(eval_data["budget"].values, dtype=torch.float32).view(-1, 1)
# eval_labels = torch.tensor(eval_data["alpha"].values, dtype=torch.float32).view(-1, 1)


# # Create TensorDataset for training data
# train_dataset = TensorDataset(input_data, budget, labels)
# train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

# # Create TensorDataset for evaluation data
# eval_dataset = TensorDataset(eval_input_data, eval_budget, eval_labels)
# eval_loader = DataLoader(eval_dataset, batch_size=128, shuffle=False)

model = kan.KAN(width=[1,8,8,2], grid = 3, k=3,seed=42,device='cpu')
print(model.act_fun[0].coef.shape)
exit(0)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

epochs = 5000
for epoch in range(epochs):
    # 前向传播
    for batch in train_loader:
        input_data, budget, labels = batch
        input_data = input_data.view(-1, 1)
        outputs = model(input_data)
        k,b = outputs.chunk(2, dim=1)
        outputs = k * budget + b
        loss = criterion(outputs, labels)
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        total_loss = 0.0
        for eval_batch in eval_loader:
            eval_input_data, eval_budget, eval_labels = eval_batch
            eval_input_data = eval_input_data.view(-1, 1)
            eval_outputs = model(eval_input_data)
            eval_k, eval_b = eval_outputs.chunk(2, dim=1)
            eval_outputs = eval_k * eval_budget + eval_b
            eval_loss = criterion(eval_outputs, eval_labels)
            total_loss += eval_loss.item()
        average_eval_loss = total_loss / len(eval_loader)
        print(f'Epoch [{epoch+1}/{epochs}], Evaluation Loss: {average_eval_loss:.4f}')
