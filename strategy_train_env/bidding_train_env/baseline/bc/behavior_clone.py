import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
from torch.optim.lr_scheduler import LinearLR

class Actor(nn.Module):
    def __init__(self, dim_observation, hidden_size=128,isTanh=False):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(dim_observation, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, obs):
        return self.net(obs)


class BC(nn.Module):
    """
        Usage:
        bc = BC(dim_obs=16)
        bc.load_net(load_path="path_to_saved_model")
        actions = bc.take_actions(states)
    """

    def __init__(self, dim_obs, actor_lr=0.001, total_training_steps = 20000,network_random_seed=1, actor_train_iter=3,hidden_size=128,isTanh=False):
        super().__init__()
        self.dim_obs = dim_obs
        self.actor_lr = actor_lr
        self.network_random_seed = network_random_seed
        torch.manual_seed(self.network_random_seed)
        self.isTanh = isTanh
        self.actor = Actor(self.dim_obs, hidden_size=hidden_size,isTanh=isTanh)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.scheduler = LinearLR(
            self.actor_optimizer,
            start_factor=1.0,
            end_factor=0.0,
            total_iters=total_training_steps
        )
        self.actor_train_iter = actor_train_iter
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda:0" if self.use_cuda else "cpu")
        self.actor.to(self.device)
        self.train_episode = 0

    def step(self, states, actions):
        states = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions = torch.tensor(actions, dtype=torch.float32, device=self.device)
        loss_list = []
        for _ in range(self.actor_train_iter):
            predicted_actions = self.actor(states)
            loss = nn.MSELoss()(predicted_actions, actions)
            self.actor_optimizer.zero_grad()
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.7)
            self.actor_optimizer.step()
            loss_list.append(loss.item())
        self.scheduler.step()
        return np.array(loss_list)

    def take_actions(self, states):
        if not isinstance(states, torch.Tensor):
            states = torch.tensor(states, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            actions = self.actor(states)
        if not self.isTanh:
            actions = actions.clamp(min=0).cpu().numpy()
        else:
            actions = actions.cpu().numpy()
        return actions

    def save_net_pkl(self, save_path):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        file_path = os.path.join(save_path, "bc.pkl")
        torch.save(self.actor, file_path)

    def save_jit(self, save_path):
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        jit_model = torch.jit.script(self.cpu())
        torch.jit.save(jit_model, f'{save_path}/bc_model.pth')

    def forward(self, states):

        with torch.no_grad():
            actions = self.actor(states)
        if not self.isTanh:
            actions = torch.clamp(actions, min=0)
        return actions

    def save_net(self, save_path):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        file_path = os.path.join(save_path, "bc.pt")
        torch.save(self.actor.state_dict(), file_path)

    def load_net(self, load_path="saved_model/fixed_initial_budget", device='cpu'):
        file_path = os.path.join(load_path, "bc.pt")
        self.actor.load_state_dict(torch.load(file_path, map_location=device))
        self.actor.to(self.device)
        print(f"Model loaded from {self.device}.")

    def load_net_pkl(self, load_path="saved_model/fixed_initial_budget"):
        file_path = os.path.join(load_path, "bc.pkl")
        self.actor = torch.load(file_path, map_location=self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.actor.to(self.device)
        print(f"Model loaded from {self.device}.")


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

class ActorSpline(nn.Module):
    def __init__(self, dim_observation, hidden_size=128,isTanh=False):
        super(ActorSpline, self).__init__()
        self.id_embedding = nn.Embedding(48, 8)
        self.time_embedding = nn.Embedding(48, 8)
        self.period_embedding = nn.Embedding(7, 8)
        num = 3
        self.k = 3
        self.coefficient_num = 8
        grid = torch.linspace(-1.0, 1.0, steps = num + 1)[None,:].expand(self.coefficient_num, num+1)
        grid = extend_grid(grid, k_extend=self.k)
        self.grid = torch.nn.Parameter(grid, requires_grad=False)
        self.feature_net = nn.Sequential(
            nn.Linear(8, hidden_size),
            nn.ReLU(),
        )
        self.net = nn.Sequential(
            nn.Linear(hidden_size, self.coefficient_num * (self.grid.shape[1]-self.k-1))
        )
        self.scale_sp_net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, self.coefficient_num)
            
        )
        self.scale_base_net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )
        self.base_fun = nn.SiLU()
        self.scale_base = torch.nn.Parameter(0.0 * 1 / np.sqrt(1) + \
                         1.0 * (torch.rand(1, 1)*2-1) * 1/np.sqrt(1)).requires_grad_(True)
        self.scale_sp = torch.nn.Parameter(torch.ones(self.coefficient_num, 1) * 1.0 * 1 / np.sqrt(self.coefficient_num)).requires_grad_(True)  # make scale trainable

    def forward(self, obs, budget):
        # # (batch_size, obs_dim)
        # obs_id = obs[:, 1].long()  # Assuming the first dimension represents IDs
        # obs_time = obs[:, 2].long()  # The rest are the features
        # obs_period = (obs[:, 0] % 7).long()  # Assuming the third dimension represents periods
        # time_embedded = self.time_embedding(obs_time)  # Embedding the time
        # id_embedded = self.id_embedding(obs_id)
        # period_embedded = self.period_embedding(obs_period)  # Embedding the period
        # obs = torch.cat([id_embedded], dim=1)  # Concatenate the features with the embeddings
        obs = self.id_embedding(obs.long())  # Assuming obs is a tensor of IDs
        feature = self.feature_net(obs)
        coef = self.net(feature)  # (batch_size, coefficient_num * (num_grid_points - k - 1))
        coef = coef.view(obs.shape[0],self.coefficient_num, 1, self.grid.shape[1]-self.k-1)
        scale_sp = self.scale_sp_net(feature)  # (batch_size, coefficient_num)
        scale_sp = scale_sp.view(obs.shape[0], self.coefficient_num, 1)
        scale_base = self.scale_base_net(feature)
        scale_base = scale_base.unsqueeze(1)  # (batch_size, 1)
        # for i in range(budget.shape[0]):
        #     B = B_batch(budget[i].unsqueeze(1),self.grid,self.k)
        #     base = self.base_fun(budget[i].unsqueeze(1)) 
        #     y = torch.einsum('ajk,bjlk->abjl', B, coef[i].unsqueeze(0))  # (batch_size, coefficient_num, num_grid_points)
        #     y = y.squeeze(1)
        #     # y = scale_base[i,:,:] * base[:,:,None] + scale_base[i,:,:] * y
        #     y = self.scale_base[None,:,:] * base[:,:,None] + self.scale_base[None,:,:] * y
        #     y_pred = torch.mean(y, dim=1)
        #     result.append(y_pred)
        # result = torch.stack(result, dim=0)
        B = B_batch(budget,self.grid,self.k)
        base = self.base_fun(budget)
        y = torch.einsum('ijk,ijlk->ijl', B, coef)
        y = self.scale_base[None,:,:] * base[:,:,None] + self.scale_sp[None,:,:] * y
        result = torch.sum(y, dim=1)
        return result

class SimpleNet(nn.Module):
    def __init__(self, k=3, grid_size=3, num = 3):
        super(SimpleNet, self).__init__()
        self.id_embedding = nn.Embedding(48, 8) 
        
        self.k = k
        self.grid_size = grid_size
        self.input_dim = 8
        grid = torch.linspace(-1.0, 1.0, steps = num + 1)[None,:].expand(self.input_dim, num+1)
        grid = extend_grid(grid, k_extend=k)
        self.grid = torch.nn.Parameter(grid, requires_grad=False)
        self.k_linear = nn.Linear(8, 32)
        self.k_linear2 = nn.Linear(32, 8*(self.grid.shape[1]-self.k-1))
        self.relu = nn.ReLU()
        self.base_fun = nn.SiLU()
        self.scale_base = torch.nn.Parameter(0.0 * 1 / np.sqrt(1) + \
                         1.0 * (torch.rand(1, 1)*2-1) * 1/np.sqrt(1)).requires_grad_(True)
        self.scale_sp = torch.nn.Parameter(torch.ones(self.input_dim, 1) * 1.0 * 1 / np.sqrt(self.input_dim)).requires_grad_(True)  # make scale trainable


    def forward(self, x, budget):
        emb = self.id_embedding(x.long())
        coef = self.relu(self.k_linear(emb))
        coef = self.k_linear2(coef)
        coef = coef.view(x.shape[0],self.input_dim, 1, self.grid.shape[1]-self.k-1)
        B = B_batch(budget,self.grid,self.k)
        base = self.base_fun(budget)
        # print(B.shape,self.grid.shape)
        # print(coef.shape)
        y = torch.einsum('ijk,ijlk->ijl', B, coef)
        y = self.scale_base[None,:,:] * base[:,:,None] + self.scale_sp[None,:,:] * y
        y_pred = torch.sum(y, dim=1)
        return y_pred
class BC_SPLINE(nn.Module):
    """
        Usage:
        bc = BC(dim_obs=16)
        bc.load_net(load_path="path_to_saved_model")
        actions = bc.take_actions(states)
    """

    def __init__(self, dim_obs, actor_lr=0.001, total_training_steps = 20000,network_random_seed=1, actor_train_iter=1,hidden_size=128,isTanh=False):
        super().__init__()
        self.dim_obs = dim_obs
        self.actor_lr = actor_lr
        self.network_random_seed = network_random_seed
        torch.manual_seed(self.network_random_seed)
        self.isTanh = isTanh
        # self.actor = ActorSpline(self.dim_obs, hidden_size=hidden_size,isTanh=isTanh)
        self.actor = SimpleNet(k=3, grid_size=3, num=3)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.scheduler = LinearLR(
            self.actor_optimizer,
            start_factor=1.0,
            end_factor=0.0,
            total_iters=total_training_steps
        )
        self.actor_train_iter = actor_train_iter
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda:0" if self.use_cuda else "cpu")
        self.actor.to(self.device)
        self.train_episode = 0

    def step(self, states, budget, actions):
        states = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions = torch.tensor(actions, dtype=torch.float32, device=self.device)
        budget = torch.tensor(budget, dtype=torch.float32, device=self.device)
        loss_list = []
        for _ in range(self.actor_train_iter):
            predicted_actions = self.actor(states,budget = budget)
            # mask = []
            # for i in range(actions.shape[0]):
            #     temp_mask = torch.zeros_like(actions[i])
            #     for j in range(actions.shape[1]):
            #         if actions[i][j] > 0 and actions[i][j] < 300:
            #             temp_mask[j] = 1
            #         else:
            #             break
            #     mask.append(temp_mask)
            # mask = torch.stack(mask, dim=0).to(self.device)
            predicted_actions = predicted_actions.squeeze()
            loss = nn.MSELoss()(predicted_actions, actions)
            self.actor_optimizer.zero_grad()
            self.train_episode += 1
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.7)
            self.actor_optimizer.step()
            loss_list.append(loss.item())
        # self.scheduler.step()
        return np.array(loss_list)

    def take_actions(self, states):
        if not isinstance(states, torch.Tensor):
            states = torch.tensor(states, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            actions = self.actor(states)
        if not self.isTanh:
            actions = actions.clamp(min=0).cpu().numpy()
        else:
            actions = actions.cpu().numpy()
        return actions

    def save_net_pkl(self, save_path):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        file_path = os.path.join(save_path, "bc.pkl")
        torch.save(self.actor, file_path)

    def save_jit(self, save_path):
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        jit_model = torch.jit.script(self.cpu())
        torch.jit.save(jit_model, f'{save_path}/bc_model.pth')

    def forward(self, states):
        with torch.no_grad():
            actions = self.actor(states)
        if not self.isTanh:
            actions = torch.clamp(actions, min=0)
        return actions

    def save_net(self, save_path):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        file_path = os.path.join(save_path, "bc.pt")
        torch.save(self.actor.state_dict(), file_path)

    def load_net(self, load_path="saved_model/fixed_initial_budget", device='cpu'):
        file_path = os.path.join(load_path, "bc.pt")
        self.actor.load_state_dict(torch.load(file_path, map_location=device))
        self.actor.to(self.device)
        print(f"Model loaded from {self.device}.")

    def load_net_pkl(self, load_path="saved_model/fixed_initial_budget"):
        file_path = os.path.join(load_path, "bc.pkl")
        self.actor = torch.load(file_path, map_location=self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.actor.to(self.device)
        print(f"Model loaded from {self.device}.")
