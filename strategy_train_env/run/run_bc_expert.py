import numpy as np
import torch
import pandas as pd
from bidding_train_env.common.utils import normalize_state, normalize_reward, save_normalize_dict
from bidding_train_env.baseline.iql.replay_buffer import ReplayBuffer
from bidding_train_env.baseline.bc.behavior_clone import BC
import logging
import ast

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


def train_model():
    """
    train BC model
    """

    train_data_path = "./data/subopt_rl_data/period-7-rlData_fill_action.csv"
    training_data = pd.read_csv(train_data_path)

    def safe_literal_eval(val):
        if pd.isna(val):
            return val  # 如果是NaN，返回NaN
        try:
            return list(ast.literal_eval(val))
        except (ValueError, SyntaxError):
            val = val.replace(" ", ",")  # 将"nan"替换为"0"
            return list(ast.literal_eval(val))
        
    def add_feature(row):
        if not isinstance(row["next_state"], list):
            return row["next_state"]
        else:
            # print(row["next_state"])
            return [row["CPAConstraint"]/130.0, row["budget"]/4850.0] + row["next_state"][:]
    # 使用apply方法应用上述函数
    training_data["state"] = training_data["state"].apply(safe_literal_eval)
    training_data["next_state"] = training_data["next_state"].apply(safe_literal_eval)
    # Insert CPAConstraint and budget into the state
    training_data["state"] = training_data.apply(
        lambda row: [row["CPAConstraint"]/130.0, row["budget"]/4850.0] + row["state"][:], axis=1
    )
    # print(training_data["next_state"].iloc[0])
    training_data["next_state"] = training_data.apply(add_feature, axis=1)
    STATE_DIM = len(training_data['state'].iloc[0])
    is_normalize = True
    if is_normalize:
        normalize_dic = normalize_state(training_data, STATE_DIM, normalize_indices=[15, 16, 17])
        normalize_reward(training_data, "reward_continuous")
        save_normalize_dict(normalize_dic, "saved_model/BCExperttest")

    replay_buffer = ReplayBuffer()
    add_to_replay_buffer(replay_buffer, training_data, is_normalize)
    print(len(replay_buffer.memory))

    logger.info(f"Replay buffer size: {len(replay_buffer.memory)}")

    model = BC(dim_obs=STATE_DIM)
    step_num = 20000
    batch_size = 100
    for i in range(step_num):
        states, actions, _, _, _ = replay_buffer.sample(batch_size)
        a_loss = model.step(states, actions)
        logger.info(f"Step: {i} Action loss: {np.mean(a_loss)}")

    # model.save_net("saved_model/BCExperttest")
    model.save_jit("saved_model/BCExperttest")
    test_trained_model(model, replay_buffer)


def load_model():
    """
    load model
    """
    model = BC(dim_obs=16)
    model.load_net("saved_model/BCExperttest")
    test_state = np.ones(16, dtype=np.float32)
    test_state_tensor = torch.tensor(test_state, dtype=torch.float)
    logger.info(f"Test action: {model.take_actions(test_state_tensor)}")


def add_to_replay_buffer(replay_buffer, training_data, is_normalize):
    for row in training_data.itertuples():
        state, action, reward, next_state, done = row.state if not is_normalize else row.normalize_state, row.action, row.reward if not is_normalize else row.normalize_reward, row.next_state if not is_normalize else row.normalize_nextstate, row.done
        # ! 去掉了所有的done==1的数据
        if done != 1:
            replay_buffer.push(np.array(state), np.array([action]), np.array([reward]), np.array(next_state),
                               np.array([done]))
        else:
            replay_buffer.push(np.array(state), np.array([action]), np.array([reward]), np.zeros_like(state),
                               np.array([done]))


def test_trained_model(model, replay_buffer):
    states, actions, rewards, next_states, terminals = replay_buffer.sample(100)
    pred_actions = model.take_actions(states)
    actions = actions.cpu().detach().numpy()
    tem = np.concatenate((actions, pred_actions), axis=1)
    print("action VS pred_action:", tem)


if __name__ == "__main__":
    run_bc()
