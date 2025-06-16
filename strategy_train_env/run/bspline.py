import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import os
import time
import kan
import ast
import sys
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import LinearLR
from torch.utils.tensorboard import SummaryWriter

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bidding_train_env.common.utils import save_normalize_dict
from bidding_train_env.baseline.iql.replay_buffer import ReplayBuffer
from torch.utils.data import Dataset
def safe_literal_eval(val):
    if pd.isna(val):
        return val  # 如果是NaN，返回NaN
    try:
        return ast.literal_eval(val)
    except (ValueError, SyntaxError):
        print(val)
        return val  # 如果解析出错，返回原值

data = pd.read_csv("./data/bspline_data/training_data_bspline_50.csv")
# data = data[(data["period"] == 7) & (data["advertiserNumber"] == 0)]

data["state"] = data["state"].apply(safe_literal_eval)
data["budget_coef"] = data["budget_coef"].apply(
        safe_literal_eval
    )

def normalize_state(training_data, state_dim, normalize_indices):
    """
    Normalize features for reinforcement learning.
    Args:
        training_data: A DataFrame containing the training data.
        state_dim: The total dimension of the features.
        normalize_indices: A list of indices of the features to be normalized.

    Returns:
        A dictionary containing the normalization statistics.
    """
    state_columns = [f'state{i}' for i in range(state_dim)]

    for i, state_col in enumerate(state_columns):
        training_data[state_col] = training_data['state'].apply(
            lambda x: x[i] if x is not None and not np.isnan(x).any() else 0.0)
    stats = {
        i: {
            'min': training_data[state_columns[i]].min(),
            'max': training_data[state_columns[i]].max(),
            'mean': training_data[state_columns[i]].mean(),
            'std': training_data[state_columns[i]].std()
        }
        for i in normalize_indices
    }

    for state_col in state_columns:
        if int(state_col.replace('state', '')) in normalize_indices:
            min_val = stats[int(state_col.replace('state', ''))]['min']
            max_val = stats[int(state_col.replace('state', ''))]['max']
            training_data[f'normalize_{state_col}'] = (
                                                              training_data[state_col] - min_val) / (
                                                              max_val - min_val + 0.01)

        else:
            training_data[f'normalize_{state_col}'] = training_data[state_col]


    training_data['normalize_state'] = training_data.apply(
        lambda row: tuple(row[f'normalize_{state_col}'] for state_col in state_columns), axis=1)

    return stats


def add_to_replay_buffer(replay_buffer, training_data, is_normalize):

    for row in training_data.itertuples():
        state, budget_coef = row.state if not is_normalize else row.normalize_state, row.budget_coef
        for item in budget_coef:
            replay_buffer.push(np.array(state), np.array([item[0] / 20000.0]), np.array([item[1] / 300.0]), np.zeros_like(state),
                               np.array([0]))
state_dim = 39
normalize_indices = [35,36,37,38]
is_normalize = True

save_path = "bc_bspline_k3_20"
normalize_dic = normalize_state(data, state_dim, normalize_indices)
# normalize_reward(training_data, "reward_continuous")
save_normalize_dict(normalize_dic, f"saved_model/{save_path}")

class CustomDataset(Dataset):
    def __init__(self, training_data, is_normalize):
        self.data = training_data
        self.is_normalize = is_normalize

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        state = row.state if not self.is_normalize else row.normalize_state
        budget_coef = row.budget_coef
        samples = []
        # for item in budget_coef:
        samples.append((np.array(state), 
                        np.array([item[0] / 20000.0 for index,item in enumerate(budget_coef) if index%2== 0]), 
                        np.array([item[1] / 300.0 for index,item in enumerate(budget_coef) if index%2== 0]), 
                        np.zeros_like(state), 
                        np.array([0])))
        return samples

dataset = CustomDataset(data, is_normalize)
dataloader = DataLoader(dataset, batch_size=128, shuffle=True, collate_fn=lambda x: [item for sublist in x for item in sublist])


def B_batch(x, grid, k=0, extend=True, device='cpu'):
    '''
    evaludate x on B-spline bases
    
    Args:
    -----
        x : 2D torch.tensor
            inputs, shape (number of splines, number of samples)
        grid : 2D torch.tensor
            grids, shape (number of splines, number of grid points)
        k : int
            the piecewise polynomial order of splines.
        extend : bool
            If True, k points are extended on both ends. If False, no extension (zero boundary condition). Default: True
        device : str
            devicde
    
    Returns:
    --------
        spline values : 3D torch.tensor
            shape (batch, in_dim, G+k). G: the number of grid intervals, k: spline order.
      
    Example
    -------
    >>> from kan.spline import B_batch
    >>> x = torch.rand(100,2)
    >>> grid = torch.linspace(-1,1,steps=11)[None, :].expand(2, 11)
    >>> B_batch(x, grid, k=3).shape
    '''
    x = x.unsqueeze(dim=2)
    grid = grid.unsqueeze(dim=0)
    
    if k == 0:
        value = (x >= grid[:, :, :-1]) * (x < grid[:, :, 1:])
    else:
        B_km1 = B_batch(x[:,:,0], grid=grid[0], k=k - 1)
        
        value = (x - grid[:, :, :-(k + 1)]) / (grid[:, :, k:-1] - grid[:, :, :-(k + 1)]) * B_km1[:, :, :-1] + (
                    grid[:, :, k + 1:] - x) / (grid[:, :, k + 1:] - grid[:, :, 1:(-k)]) * B_km1[:, :, 1:]
    
    # in case grid is degenerate
    value = torch.nan_to_num(value)
    return value

def extend_grid(grid, k_extend=0):
    '''
    extend grid
    '''
    h = (grid[:, [-1]] - grid[:, [0]]) / (grid.shape[1] - 1)

    for i in range(k_extend):
        grid = torch.cat([grid[:, [0]] - h, grid], dim=1)
        grid = torch.cat([grid, grid[:, [-1]] + h], dim=1)

    return grid
class SimpleNet(nn.Module):
    def __init__(self, k=3, grid_size=3, num = 3, hidden_size = 128):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        super(SimpleNet, self).__init__()
        self.id_embedding = nn.Embedding(48, 8) 
        self.period_embedding = nn.Embedding(8, 8)  # period embedding
        self.time_embedding = nn.Embedding(48, 16)
        self.k = k
        self.grid_size = grid_size
        self.input_dim = 8
        grid = torch.linspace(-1.0, 1.0, steps = num + 1)[None,:].expand(self.input_dim, num+1)
        grid = extend_grid(grid, k_extend=k)
        self.grid = torch.nn.Parameter(grid, requires_grad=False)
        self.feature_net = nn.Sequential(
            nn.Linear(68, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU())
        self.coef_net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, self.input_dim * (self.grid.shape[1] - self.k - 1))
        )
        self.scale_sp_net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, self.input_dim)
        )
        self.scale_base_net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )
        # self.k_linear = nn.Linear(75, 32)
        # self.k_linear2 = nn.Linear(32, 8*(self.grid.shape[1]-self.k-1))
        self.relu = nn.ReLU()
        self.base_fun = nn.SiLU()
        # self.scale_base = torch.nn.Parameter(0.0 * 1 / np.sqrt(1) + \
        #                  1.0 * (torch.rand(1, 1)*2-1) * 1/np.sqrt(1)).requires_grad_(True)
        # self.scale_sp = torch.nn.Parameter(torch.ones(self.input_dim, 1) * 1.0 * 1 / np.sqrt(self.input_dim)).requires_grad_(True)  # make scale trainable

    def forward(self, x, budget):
        id_emb = self.id_embedding(x[:,1].long())
        period_emb = self.period_embedding(x[:,0].long())
        time_emb = self.time_embedding(x[:,2].long())

        emb = torch.cat([id_emb, period_emb, time_emb, x[:, 3:]], dim=1)
        feature = self.feature_net(emb)
        coef = self.coef_net(feature)
        coef = coef.view(x.shape[0],self.input_dim, 1, self.grid.shape[1]-self.k-1)
        scale_sp = self.scale_sp_net(feature)  # (batch_size, coefficient_num)
        scale_sp = scale_sp.view(x.shape[0], self.input_dim, 1)
        scale_base = self.scale_base_net(feature)
        scale_base = scale_base.unsqueeze(1)  
        # B = B_batch(budget,self.grid,self.k)
        # base = self.base_fun(budget)
        # # print(B.shape,self.grid.shape)
        # # print(coef.shape)
        # y = torch.einsum('ijk,ijlk->ijl', B, coef)
        # y = scale_base[:,:,:] * base[:,:,None] + scale_sp[:,:,:] * y
        # y_pred = torch.sum(y, dim=1)
        result = []
        Bs = []
        for i in range(budget.shape[0]):
            B = B_batch(budget[i].unsqueeze(1),self.grid,self.k)
            Bs.append(B)
        Bs = torch.stack(Bs, dim=0)  # (batch_size, coefficient_num, num_grid_points)
        base = self.base_fun(budget) 
        # (128,20,8,6) (128,8,1,6)
        y = torch.einsum('bajk,bjlk->bajl', Bs, coef)  # (batch_size, coefficient_num, num_grid_points)
        y = y.squeeze()
        y = scale_base[:,:,:] * base[:,:,None] + scale_base[:,:,:] * y
        # y = scale_base[None,:,:] * base[:,:,None] + self.scale_base[None,:,:] * y
        y_pred = torch.mean(y, dim=-1)
        return y_pred
    
    def save_jit(self, save_path):
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        jit_model = torch.jit.script(self.cpu())
        torch.jit.save(jit_model, f'{save_path}/bc_model.pth')
    def load_net_pkl(self, load_path="saved_model/fixed_initial_budget"):
        file_path = os.path.join(load_path, "bc.pkl")
        self.actor = torch.load(file_path, map_location=self.device)
        self.actor.to(self.device)
        print(f"Model loaded from {self.device}.")

def save_net_pkl(model,save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    file_path = os.path.join(save_path, "bc.pkl")
    torch.save(model, file_path)

# 初始化网络、损失函数和优化器
model = SimpleNet()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
epochs = 100
scheduler = LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=0.1,
            total_iters=epochs
        )
writer = SummaryWriter(log_dir=f"./log/{save_path}")
for i in range(epochs):
    loss_total = 0.0
    for batch in dataloader:
        states, budget, labels, _, _ = zip(*batch)
        states = torch.tensor(states, dtype=torch.float32)
        budget = torch.tensor(budget, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.float32)
        outputs = model(states, budget).squeeze()
        loss = criterion(outputs, labels)
        loss_total += loss.item()
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.25)
        optimizer.step()
    scheduler.step()
    loss_total = loss_total / len(dataloader)
    print(f"Epoch {i+1}/{epochs}, Loss: {loss_total:.8f}")
    writer.add_scalar("Loss/Loss", loss_total, i)

save_net_pkl(model.state_dict(),f"saved_model/{save_path}")




# train_budget = [[] for _ in range(48)]
# train_prediction = [[] for _ in range(48)]
# train_label = [[] for _ in range(48)]
# for train_batch in dataloader:
#     states, budget, labels, _, _ = zip(*batch)
#     states = torch.tensor(states, dtype=torch.float32)
#     budget = torch.tensor(budget, dtype=torch.float32)
#     labels = torch.tensor(labels, dtype=torch.float32)
#     with torch.no_grad():
#         train_outputs = model(states, budget)
#         for i in range(len(states)):
#             advertiser_id = int(states[i][2].item())
#             train_budget[advertiser_id].append(budget[i].item())
#             train_prediction[advertiser_id].append(train_outputs[i].item())
#             train_label[advertiser_id].append(labels[i].item())

# plt.figure(figsize=(20, 15))
# for i in range(48):
#     plt.subplot(6, 8, i + 1)  # Create a 6x8 grid of subplots
#     plt.scatter(train_budget[i], train_prediction[i], label="Train Prediction", color="green")
#     plt.scatter(train_budget[i], train_label[i], label="Train Label", color="orange")
#     plt.xlabel("Budget")
#     plt.ylabel("Alpha")
#     plt.title(f"Advertiser {i} (Train Data)")
#     plt.legend()
#     plt.grid(True)

# plt.tight_layout()  # Adjust layout to prevent overlap
# plt.savefig("time_coef.png")
# plt.close()