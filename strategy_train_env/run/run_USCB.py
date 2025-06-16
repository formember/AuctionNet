# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/td3/#td3_continuous_actionpy
import os
import random
import time
from dataclasses import dataclass
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
import pandas as pd
import glob
import copy
from gymnasium import spaces
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bidding_train_env.offline_eval.online_env import OnlineEnv

@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = False
    """whether to save model into the `runs/{run_name}` folder"""
    upload_model: bool = False
    """whether to upload the saved model to huggingface"""
    hf_entity: str = ""
    """the user or org name of the model repository from the Hugging Face Hub"""

    # Algorithm specific arguments
    env_id: str = "online"
    """the id of the environment"""
    total_timesteps: int = 2000000  
    """total timesteps of the experiments"""
    learning_rate: float = 1e-5
    """the learning rate of the optimizer"""
    buffer_size: int = int(1e6)
    """the replay memory buffer size"""
    gamma: float = 1.0
    """the discount factor gamma"""
    tau: float = 0.005
    """target smoothing coefficient (default: 0.005)"""
    batch_size: int = 256
    """the batch size of sample from the reply memory"""
    policy_noise: float = 0.2
    """the scale of policy noise"""
    exploration_noise: float = 0.1
    """the scale of exploration noise"""
    learning_starts: int = 400
    """timestep to start learning"""
    policy_frequency: int = 2
    """the frequency of training policy (delayed)"""
    noise_clip: float = 0.5
    """noise clip parameter of the Target Policy Smoothing Regularization"""
    load_bc = False

STATE_DIM = 5
ACTION_HIGH = 150
ACTION_LOW = 40
# ALGO LOGIC: initialize agent here:
class QNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(np.array(STATE_DIM + 1), 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(STATE_DIM, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc_mu = nn.Linear(16, 1)
        # action rescaling
        self.register_buffer(
            "action_scale", torch.tensor((ACTION_HIGH - ACTION_LOW) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.tensor((ACTION_HIGH + ACTION_LOW) / 2.0, dtype=torch.float32)
        )

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc_mu(x))
        return x * self.action_scale + self.action_bias


if __name__ == "__main__":

    args = tyro.cli(Args)
    run_name = f"{args.env_id}__{args.exp_name}__{int(time.time())}"
    writer = SummaryWriter(f"./saved_model/final/{run_name}_not_full/log")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    data_path = "./online_data_part/"
    csv_files = glob.glob(os.path.join(data_path, '*.csv'))
    
    
    for adnum in range(0,48):
        # env setup
        env = OnlineEnv()

        actor = Actor().to(device)
        qf1 = QNetwork().to(device)
        qf2 = QNetwork().to(device)
        qf1_target = QNetwork().to(device)
        qf2_target = QNetwork().to(device)
        target_actor = Actor().to(device)
        target_actor.load_state_dict(actor.state_dict())
        qf1_target.load_state_dict(qf1.state_dict())
        qf2_target.load_state_dict(qf2.state_dict())

        q_optimizer = optim.Adam(list(qf1.parameters()), lr=args.learning_rate)
        actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.learning_rate)

        rb = ReplayBuffer(
            args.buffer_size,
            observation_space=spaces.Box(low=0, high=1, shape=(STATE_DIM,), dtype=np.float32),
            action_space=spaces.Box(low=ACTION_LOW, high=ACTION_HIGH, shape=(1,), dtype=np.float32),
            device=device,
            handle_timeout_termination=False,
        )
        start_time = time.time()

        # TRY NOT TO MODIFY: start the game

        if args.load_bc:

            pretrained_state_dict = torch.load('./saved_model/final/online__run_td3__1727341821_not_full/run_td3_500_1.cleanrl_model',map_location='cpu')
            actor.load_state_dict(pretrained_state_dict[0])
            target_actor.load_state_dict(actor.state_dict())

        dataset_index = 0
        dataset = pd.read_csv(csv_files[dataset_index % 21])
        reset_count = 0
        obs = env.reset(dataset,adnum)
        total_action = 0
        R = 0
        learning_start = False
        total_training_steps = 0
        for global_step in range(args.total_timesteps):
            # ALGO LOGIC: put action logic here
            if not learning_start:
                actions = np.array([np.random.uniform(40, 150)])
                # with torch.no_grad():
                #     actions = actor(torch.Tensor(obs).to(device))
                #     actions += torch.normal(0, actor.action_scale * args.exploration_noise)
                #     actions = actions.cpu().numpy().clip(ACTION_LOW,ACTION_HIGH)
            else:
                with torch.no_grad():
                    actions = actor(torch.Tensor(obs).to(device))
                    actions += torch.normal(0, actor.action_scale * args.exploration_noise)
                    actions = actions.cpu().numpy().clip(ACTION_LOW,ACTION_HIGH)
            # print(obs)
            # print(actions)
            total_action+=actions.item()
            if global_step % 10 ==0:
                print(total_action/10)
                total_action=0
            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, rewards, terminations, truncations,info = env.step(actions)

            if not truncations and not terminations:
                inner_trun, inner_term = False, False
                inner_env = copy.deepcopy(env)
                inner_obs = copy.deepcopy(next_obs)
                while not inner_trun and not inner_term:
                #     if global_step < args.learning_starts:
                #         # inner_actions = np.array([np.random.random()*200])
                #         with torch.no_grad():
                #             inner_actions = actor(torch.Tensor(obs).to(device))
                #             inner_actions += torch.normal(0, actor.action_scale * args.exploration_noise)
                #             inner_actions = inner_actions.cpu().numpy().clip(ACTION_LOW,ACTION_HIGH)
                #     else:
                #         with torch.no_grad():
                #             inner_actions = actor(torch.Tensor(inner_obs).to(device))
                #             inner_actions += torch.normal(0, actor.action_scale * args.exploration_noise)
                #             inner_actions = inner_actions.cpu().numpy().clip(ACTION_LOW,ACTION_HIGH)
                    inner_obs, inner_rewards, inner_term, inner_trun, inner_info = inner_env.step(actions)
                V_inner = inner_info['episodic_score']
                final_reward = V_inner

            # TRY NOT TO MODIFY: record rewards for plotting purposes
            if truncations or terminations:
                print(f"global_step={global_step}, episodic_length={info['episodic_length']}, episodic_return={info['episodic_return']}, episodic_score={info['episodic_score']}")
                writer.add_scalar("charts/episodic_return", info['episodic_return'], global_step)
                writer.add_scalar("charts/episodic_length", info['episodic_length'], global_step)
                writer.add_scalar("charts/episodic_score", info['episodic_score'], global_step)
                final_reward = info['episodic_score']

            # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
            rb.add(obs, next_obs, actions, min(final_reward/10.0, 1.0), terminations or truncations, info)
            obs = next_obs
            # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
            if truncations or terminations:
                if (reset_count + 1) % 5 == 0:
                    dataset_index += 1
                    if dataset_index == 21:
                        learning_start = True
                    dataset = pd.read_csv(csv_files[dataset_index%21])
                reset_count += 1
                print(reset_count)
                obs = env.reset(dataset,adnum)
            # ALGO LOGIC: training.
            if learning_start:
                print("starting training")
                data = rb.sample(args.batch_size)
    #             with torch.no_grad():
    #                 clipped_noise = (torch.randn_like(data.actions, device=device) * args.policy_noise).clamp(
    #                     -args.noise_clip, args.noise_clip
    #                 ) * target_actor.action_scale

    #                 next_state_actions = (target_actor(data.next_observations) + clipped_noise).clamp(
    #                     ACTION_LOW, ACTION_HIGH
    #                 )
    #                 qf1_next_target = qf1_target(data.next_observations, next_state_actions)
    #                 qf2_next_target = qf2_target(data.next_observations, next_state_actions)
    #                 min_qf_next_target = torch.min(qf1_next_target, qf2_next_target)
    #                 next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * args.gamma * (min_qf_next_target).view(-1)

                qf1_a_values = qf1(data.observations, data.actions).view(-1)
                qf1_loss = F.mse_loss(qf1_a_values, data.rewards.flatten())
                qf_loss = qf1_loss

                # optimize the model
                q_optimizer.zero_grad()
                qf_loss.backward()
                q_optimizer.step()

                if global_step % args.policy_frequency == 0:
                    actor_loss = -qf1(data.observations, actor(data.observations)).mean()
                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    actor_optimizer.step()

                    # update the target network
                    for param, target_param in zip(actor.parameters(), target_actor.parameters()):
                        target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                    # for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                    #     target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                    # for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                        # target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                total_training_steps += 1
                if global_step % 100 == 0:
                    writer.add_scalar("losses/qf1_values", qf1_a_values.mean().item(), global_step)
                    writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
                    writer.add_scalar("losses/qf_loss", qf_loss.item() / 2.0, global_step)
                    writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                    print("SPS:", int(global_step / (time.time() - start_time)))
                    writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
                if total_training_steps == 400:
                    model_path = f"./saved_model/final/{run_name}_not_full/{args.exp_name}_{global_step}_{adnum}.cleanrl_model"
                    torch.save((actor.state_dict(), qf1.state_dict(), qf2.state_dict()), model_path)
                    print(f"model saved to {model_path}")
                    break

        writer.close()