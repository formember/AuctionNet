import torch
import torch.nn as nn
import os

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
class BC_SPLINE(nn.Module):
    def __init__(self, k=3, grid_size=3, num = 3, hidden_size = 128):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        super(BC_SPLINE, self).__init__()
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

        emb = torch.cat([id_emb, period_emb, time_emb,x[:,3:]], dim=1)
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