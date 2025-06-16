import numpy as np
import torch
import pandas as pd
from bidding_train_env.common.utils import save_normalize_dict
from bidding_train_env.baseline.iql.replay_buffer import ReplayBuffer
from bidding_train_env.baseline.bc.behavior_clone import BC_SPLINE
import logging
import ast
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, TensorDataset

np.set_printoptions(suppress=True, precision=4)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(name)s] [%(filename)s(%(lineno)d)] [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def run_bc():
    """
    Run bc model training and evaluation.
    """
    train_model()
    # load_model()



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


def train_model():
    """
    train BC model
    """

    train_data_path = "./data/bspline_data/training_data_bspline_50.csv"
    training_data = pd.read_csv(train_data_path)
    training_data = training_data[(training_data["period"] == 7) & (training_data["timeStepIndex"] == 0)]
    
    def safe_literal_eval(val):
        if pd.isna(val):
            return val  # 如果是NaN，返回NaN
        try:
            return ast.literal_eval(val)
        except (ValueError, SyntaxError):
            print(val)
            return val  # 如果解析出错，返回原值

    
    training_data["state"] = training_data["state"].apply(safe_literal_eval)
    training_data["budget_coef"] = training_data["budget_coef"].apply(
        safe_literal_eval
    )
        
    # training_data["next_state"] = training_data["next_state"].apply(safe_literal_eval)

    state_dim = 46
    normalize_indices = [42,43,44,45]
    is_normalize = True

    normalize_dic = normalize_state(training_data, state_dim, normalize_indices)
    # normalize_reward(training_data, "reward_continuous")
    save_normalize_dict(normalize_dic, "saved_model/bc_bspline")

    replay_buffer = ReplayBuffer()
    add_to_replay_buffer_v2(replay_buffer, training_data, is_normalize)
    print(len(replay_buffer.memory))
    writer = SummaryWriter(log_dir="./log/bc_bspline")
    logger.info(f"Replay buffer size: {len(replay_buffer.memory)}")
    step_num = 10000
    model = BC_SPLINE(dim_obs=state_dim,hidden_size=128,total_training_steps =step_num,actor_lr=0.001)
    
    batch_size = 128
    for i in range(step_num): 
        states, budgets, coefs, _, _ = replay_buffer.sample(batch_size)
        
        a_loss = model.step(states, budgets, coefs)
        writer.add_scalar("Loss/Action", np.mean(a_loss), i)
        logger.info(f"Step: {i} Action loss: {np.mean(a_loss)}")

    # model.save_net("saved_model/BCtest")
    model.save_jit("saved_model/bc_bspline")
    test_trained_model(model, replay_buffer)


def load_model():
    """
    load model
    """
    model = BC_SPLINE(dim_obs=16)
    model.load_net("saved_model/bc_bspline")
    test_state = np.ones(16, dtype=np.float32)
    test_state_tensor = torch.tensor(test_state, dtype=torch.float)
    logger.info(f"Test action: {model.take_actions(test_state_tensor)}")


def add_to_replay_buffer(replay_buffer, training_data, is_normalize):
    for row in training_data.itertuples():
        state, budget_coef = row.state if not is_normalize else row.normalize_state, row.budget_coef
        for item in budget_coef:
            replay_buffer.push(np.array(state[1]), np.array([item[0] / 20000.0]), np.array([item[1] / 300.0]), np.zeros_like(state),
                               np.array([0]))
        # replay_buffer.push(np.array(state), np.array([item[0] / 20000.0 for item in budget_coef]), np.array([item[1] / 300.0 if not isinstance(item[1],str) else 0 for item in budget_coef]), np.zeros_like(state),
        #                        np.array([0]))

def return_dataloder(training_data, batch_size=128):
    """
    Data loader for training data.
    Args:
        training_data: A DataFrame containing the training data.
        batch_size: The size of each batch.

    Returns:
        A generator that yields batches of training data.
    """
    data = pd.read_csv("./linear_solution/alpha_cpa.csv")
    train_data = data[(data["budget"] % 1000 ==0)]
    input_data = torch.tensor(train_data["advertiserNumber"].values, dtype=torch.float32).view(-1, 1)
    budget = torch.tensor(train_data["budget"].values, dtype=torch.float32).view(-1, 1)
    budget = budget / 20000.0  # Normalize budget to [0, 1]

    labels = torch.tensor(train_data["alpha"].values, dtype=torch.float32).view(-1, 1)
    labels = labels / 300.0  # Normalize labels to [0, 1]
    train_dataset = TensorDataset(input_data, budget, labels)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    return train_loader
    

def add_to_replay_buffer_v2(replay_buffer, training_data, is_normalize):
    data = pd.read_csv("./linear_solution/alpha_cpa.csv")
    train_data = data[(data["budget"] % 1000 ==0)]
    for row in train_data.itertuples():
        state, budget , alpha  = row.advertiserNumber, row.budget, row.alpha
        replay_buffer.push(np.array(state), np.array([budget / 20000.0]), np.array([alpha / 300.0]), np.zeros_like(state),
                            np.array([0]))
        # replay_buffer.push(np.array(state), np.array([item[0] / 20000.0 for item in budget_coef]), np.array([item[1] / 300.0 if not isinstance(item[1],str) else 0 for item in budget_coef]), np.zeros_like(state),
        #                        np.array([0]))



def test_trained_model(model, replay_buffer):
    states, actions, rewards, next_states, terminals = replay_buffer.sample(100)
    pred_actions = model.take_actions(states)
    actions = actions.cpu().detach().numpy()
    tem = np.concatenate((actions, pred_actions), axis=1)
    print("action VS pred_action:", tem)


if __name__ == "__main__":
    run_bc()