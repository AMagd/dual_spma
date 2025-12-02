import math
import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
from gymnasium.envs.toy_text.frozen_lake import generate_random_map


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    track: bool = False
    wandb_project_name: str = "cleanRL"
    wandb_entity: str = None
    capture_video: bool = True

    # Algorithm specific arguments
    env_id: str = "FrozenLake-v1"
    total_timesteps: int = 2000000
    learning_rate: float = 1e-3
    num_envs: int = 8
    num_steps: int = 128
    anneal_lr: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 4
    update_epochs: int = 4
    norm_adv: bool = True
    clip_coef: float = 0.2
    clip_vloss: bool = True
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float = None
    map_size: int = 10
    
    # Occupancy Maximization Args
    occ_ent_coef: float = 10000000000000000.0

    # to be filled in runtime
    batch_size: int = 0
    minibatch_size: int = 0
    num_iterations: int = 0


class OneHotWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.n = env.observation_space.n
        self.observation_space = gym.spaces.Box(0, 1, (self.n,), dtype=np.float32)

    def observation(self, obs):
        one_hot = np.zeros(self.n, dtype=np.float32)
        one_hot[obs] = 1.0
        return one_hot


def make_env(env_id, idx, capture_video, run_name, map_desc):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array", desc=map_desc, is_slippery=False)
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id, desc=map_desc, is_slippery=False)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = OneHotWrapper(env)
        return env
    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def safe_occ_value(value: float) -> float:
    return value if math.isfinite(value) else 0.0


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        input_dim = np.array(envs.single_observation_space.shape).prod()
        self.network = nn.Sequential(
            layer_init(nn.Linear(input_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
        )
        self.actor = layer_init(nn.Linear(64, envs.single_action_space.n), std=0.01)
        self.critic = layer_init(nn.Linear(64, 1), std=1)

    def get_value(self, x):
        return self.critic(self.network(x))

    def get_action_and_value(self, x, action=None):
        hidden = self.network(x)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)

# -------------------------------------------------------------------------
# OCCUPANCY FUNCTIONS
# -------------------------------------------------------------------------

def get_transition_tensor(env, map_size, device):
    ns = map_size * map_size
    na = env.action_space.n
    P_tensor = torch.zeros((ns, na, ns), device=device)
    P_dict = env.unwrapped.P
    for s in range(ns):
        for a in range(na):
            if s in P_dict and a in P_dict[s]:
                transitions = P_dict[s][a]
                for prob, next_s, _, _ in transitions:
                    P_tensor[s, a, next_s] += prob
    
    row_sums = P_tensor.sum(dim=2)
    mask = (row_sums == 0)
    if mask.any():
        for s in range(ns):
            for a in range(na):
                if mask[s, a]:
                    P_tensor[s, a, s] = 1.0
    return P_tensor

def get_occupancy(agent, P_tensor, device, map_size, gamma=0.99):
    num_states = map_size * map_size
    
    # 1. Get Policy Matrix pi(a|s)
    all_states_idx = torch.arange(num_states, device=device).long()
    all_states_onehot = torch.nn.functional.one_hot(all_states_idx, num_classes=num_states).float()
    
    hidden = agent.network(all_states_onehot)
    logits = agent.actor(hidden)
    pi_probs = torch.softmax(logits, dim=1) 
    
    # 2. Compute Transition Matrix P_pi
    P_pi = torch.einsum('sa,san->sn', pi_probs, P_tensor)
    
    # 3. Solve linear system for stationary distribution d(s)
    I = torch.eye(num_states, device=device)
    # Adding small jitter to diagonal for numerical stability
    A_mat = (I - gamma * P_pi).T + 1e-6 * I
    
    rho = torch.zeros(num_states, device=device)
    rho[0] = 1.0
    b = (1 - gamma) * rho
    
    try:
        d_s = torch.linalg.solve(A_mat, b)
    except RuntimeError:
        d_s = torch.ones(num_states, device=device) / num_states

    # 4. Joint State-Action Distribution
    d_s = torch.clamp(d_s, min=1e-12)
    d_s = d_s / d_s.sum() 
    d_sa = d_s.unsqueeze(1) * pi_probs 
    
    return d_sa

# -------------------------------------------------------------------------

if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    run_name = f"{args.env_id}_{args.map_size}x{args.map_size}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}", flush_secs=10)
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    random_map = generate_random_map(size=args.map_size, p=0.8)
    
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, run_name, random_map) for i in range(args.num_envs)],
    )
    ref_env = gym.make(args.env_id, desc=random_map, is_slippery=False)
    P_tensor = get_transition_tensor(ref_env, args.map_size, device)
    
    agent = Agent(envs).to(device)
    
    # ----------------------------------------------------------------------
    # FTL (Follow the Leader) Setup
    # ----------------------------------------------------------------------
    # We do NOT add lambda to the optimizer. Lambda is determined explicitly 
    # by the history of occupancies.
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    
    # Storage for FTL history (Cumulative sum of d_sa)
    # Shape: (Num States, Num Actions)
    num_states = args.map_size * args.map_size
    num_actions = envs.single_action_space.n
    cumulative_d_sa = torch.zeros(num_states, num_actions, device=device)
    ftl_update_count = 0

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    for iteration in range(1, args.num_iterations + 1):
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                        writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                        writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

        # bootstrap value
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        # ---------------------------------------------------------------------
        # FTL (Follow the Leader) UPDATE STEP
        # ---------------------------------------------------------------------
        
        optimizer.zero_grad()
        
        # 1. Get current occupancy d_pi (using lower gamma for stability/diversity as discussed)
        d_sa = get_occupancy(agent, P_tensor, device, args.map_size, gamma=0.99)
        
        # 2. Update FTL History (Avg Occupancy)
        # We need the average of past occupancies: d_bar = (1/k) * sum(d_j)
        with torch.no_grad():
            cumulative_d_sa += d_sa.detach()
            ftl_update_count += 1
            avg_d_sa = cumulative_d_sa / ftl_update_count
            
            # 3. Calculate Lambda using FTL rule: lambda^k = grad f(d_bar)
            # f(d) = d log d  =>  grad f(d) = 1 + log(d)
            # We add epsilon to prevent log(0)
            lambda_ftl = 1.0 + torch.log(avg_d_sa + 1e-12)

        # 4. Best Response Policy Update
        # The agent minimizes: d_pi * lambda^k
        # (This discourages visiting states that have been visited often in the past)
        # We assume lambda_ftl is fixed (constant) for this gradient step.
        
        loss_ftl = (d_sa * lambda_ftl).sum() * args.occ_ent_coef
        
        loss_ftl.backward()
        nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
        optimizer.step()
        
        # ---------------------------------------------------------------------

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
        
        # Logging
        with torch.no_grad():
            valid_mask = d_sa > 1e-12
            exact_d_log_d = (d_sa[valid_mask] * torch.log(d_sa[valid_mask])).sum()
            exact_occ_entropy = safe_occ_value(-exact_d_log_d.item())

        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/grad_norm", grad_norm.item(), global_step)
        writer.add_scalar("charts/exact_occupancy_entropy", exact_occ_entropy, global_step)
        writer.add_scalar("charts/ftl_loss_val", loss_ftl.item(), global_step)

        print(f"SPS: {int(global_step / (time.time() - start_time))}, OccEntropy: {exact_occ_entropy:.4f}")
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
        
        writer.flush()

    envs.close()
    ref_env.close()
    writer.close()