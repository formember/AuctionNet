import numpy as np
from bidding_train_env.common.utils import normalize_state, normalize_reward, save_normalize_dict
from bidding_train_env.baseline.dt.utils import EpisodeReplayBuffer,EpisodeConstraintReplayBuffer
from bidding_train_env.baseline.dt.dt import DecisionTransformer,DecisionTransformerConstraint,DecisionTransformerV2
from torch.utils.data import DataLoader, WeightedRandomSampler
import logging
import pickle

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(name)s] [%(filename)s(%(lineno)d)] [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def run_dt():
    train_model()


def train_model():
    state_dim = 16
    k  = 10
    # replay_buffer = EpisodeReplayBuffer(16, 1, "./data/trajectory/trajectory_data.csv")
    replay_buffer = EpisodeConstraintReplayBuffer(16, 1, "./data/adjust_best_alpha/training_data_dt_origin_feature_without7.csv",K=k)
    save_normalize_dict({"state_mean": replay_buffer.state_mean, "state_std": replay_buffer.state_std},
                        "saved_model/DTaa_origin_feature_without7_test")
    logger.info(f"Replay buffer size: {len(replay_buffer.trajectories)}")
    step_num = 4000
    batch_size = 128
    model = DecisionTransformerV2(state_dim=state_dim, act_dim=1, state_mean=replay_buffer.state_mean,
                                state_std=replay_buffer.state_std,K=k,total_steps=step_num)
    
    sampler = WeightedRandomSampler(replay_buffer.p_sample, num_samples=step_num * batch_size, replacement=True)
    dataloader = DataLoader(replay_buffer, sampler=sampler, batch_size=batch_size)

    model.train()
    i = 0
    min_loss = 1e10
    for states, actions, rewards, dones, rtg, c_b, timesteps, attention_mask in dataloader:
        # print(c_b.shape)
        train_loss = model.step(states, actions, rewards, dones, c_b, timesteps, attention_mask)
        i += 1
        # if np.mean(train_loss) < min_loss:
        #     min_loss = np.mean(train_loss)
        #     model.save_net("saved_model/DTaa_origin_feature_without7_test")
        logger.info(f"Step: {i} Action loss: {np.mean(train_loss)}")
        model.scheduler.step()

    model.save_net("saved_model/DTaa_origin_feature_without7_test")
    test_state = np.ones(state_dim, dtype=np.float32)
    logger.info(f"Test action: {model.take_actions(test_state,target_return=np.array([0.5,0.5]))}")
    logger.info(f"Test action: {model.take_actions(test_state,target_return=np.array([0.5,0.5]),pre_reward=0)}")
    logger.info(f"Test action: {model.take_actions(test_state,target_return=np.array([0.5,0.5]),pre_reward=0)}")




def load_model():
    """
    加载模型。
    """
    with open('./Model/DT/DTaa_origin_feature_without7_test/normalize_dict.pkl', 'rb') as f:
        normalize_dict = pickle.load(f)
    model = DecisionTransformer(state_dim=16, act_dim=1, state_mean=normalize_dict["state_mean"],
                                state_std=normalize_dict["state_std"])
    model.load_net("Model/DT/DTaa_origin_feature_without7_test")
    test_state = np.ones(16, dtype=np.float32)
    logger.info(f"Test action: {model.take_actions(test_state)}")


if __name__ == "__main__":
    run_dt()
