# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppopy
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
from gymnasium.spaces import Box
import minigrid
from minigrid.wrappers import FullyObsWrapper, ImgObsWrapper
from minigrid.core.world_object import Goal


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

    # Algorithm specific arguments
    env_id: str = "FrozenLake-v1"
    """the id of the environment"""
    total_timesteps: int = 500000
    """total timesteps of the experiments"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    num_envs: int = 4
    """the number of parallel game environments"""
    num_steps: int = 128
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 4
    """the number of mini-batches"""
    update_epochs: int = 10
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.1
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""

    # === Convex MDP specific arguments ===
    beta: float = 0.8
    """beta in f(d_pi) = - beta <d_pi, r> + (1-beta)*entropy(d_pi)"""
    d_bar_log_epsilon: float = 1e-8
    """small epsilon to avoid log(0) in log(d_pi)"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""


def make_env(env_id, idx, capture_video, run_name):
    def thunk():
        # --- create env ---
        if capture_video and idx == 0:
            env = gym.make(
                "FrozenLake-v1",
                desc=None,
                map_name="4x4",
                is_slippery=False,
                render_mode="rgb_array",
            )
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            if env_id == "FrozenLake-v1":
                env = gym.make(env_id, is_slippery=False)
            else:
                env = gym.make(env_id)

        env = gym.wrappers.RecordEpisodeStatistics(env)

        # --- if it's a Discrete observation (e.g. FrozenLake), make it one-hot ---
        if isinstance(env.observation_space, gym.spaces.Discrete):
            n_states = env.observation_space.n

            def obs_to_one_hot(s):
                # s is an int in [0, n_states-1]
                one_hot = np.zeros(n_states, dtype=np.float32)
                one_hot[s] = 1.0
                return one_hot

            env = gym.wrappers.TransformObservation(env, obs_to_one_hot)

            env.observation_space = Box(
                low=0.0,
                high=1.0,
                shape=(n_states,),
                dtype=np.float32,
            )

        return env

    return thunk


def ema_return(previous_ema, new_return, alpha=0.9):
    return alpha * previous_ema + (1 - alpha) * new_return


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        obs_dim = int(np.prod(envs.single_observation_space.shape))

        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, envs.single_action_space.n), std=0.01),
        )

    def get_value(self, x):
        x = x.view(x.shape[0], -1)
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        x = x.view(x.shape[0], -1)
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)


# === Convex MDP helpers ===

def estimate_d_pi_from_obs_action(
    obs_buffer: torch.Tensor,
    action_buffer: torch.Tensor,
    n_actions: int,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Monte Carlo estimate of state-action occupancy d_pi(s, a) from one-hot observations
    and discrete actions.

    Args:
        obs_buffer:    Tensor of shape [T, N, n_states], one-hot encoding of states.
        action_buffer: Tensor of shape [T, N], integer actions in [0, n_actions-1].
        eps:           Small constant to avoid division by zero.

    Returns:
        d_sa: Tensor of shape [n_states, n_actions] such that
              d_sa[s, a] ≈ visitation rate of (s, a), and d_sa.sum() ≈ 1.
    """
    # Shapes
    T, N, n_states = obs_buffer.shape
    device = obs_buffer.device

    # State indices from one-hot
    state_idx = obs_buffer.argmax(dim=-1).long()   # [T, N]
    action_idx = action_buffer.long().view(T, N)   # [T, N]

    # Flatten to 1D lists of (s, a)
    flat_s = state_idx.view(-1)                    # [T*N]
    flat_a = action_idx.view(-1)                   # [T*N]

    # Count visits to each (s, a)
    counts = torch.zeros(n_states, n_actions, device=device)
    lin_idx = flat_s * n_actions + flat_a          # [T*N] linear indices
    counts.view(-1).index_add_(
        0,
        lin_idx,
        torch.ones_like(lin_idx, dtype=torch.float32, device=device),
    )

    # Normalize to get an empirical occupancy measure (visitation rate)
    d_sa = counts / (counts.sum() + eps)
    return d_sa


def update_running_average(old_avg: torch.Tensor, new_value: torch.Tensor, k: int) -> torch.Tensor:
    """
    Running average d_bar^{(k)} = ( (k-1)/k ) d_bar^{(k-1)} + (1/k) d^{(k)}.
    """
    if k == 1:
        return new_value
    return (old_avg * (k - 1) + new_value) / k

def lambda_reward_reshape(r, d_bar, beta, eps):
    """
    modify the reward based on the convex MDP objective
    
    """
    if not isinstance(r, torch.Tensor):
        r = torch.tensor(r)
    # if not isinstance(d_bar, torch.Tensor):
    #     d_bar = torch.tensor(d_bar)
    # if not isinstance(beta, torch.Tensor):
    #     beta = torch.tensor(beta)
    # if not isinstance(eps, torch.Tensor):
    #     eps = torch.tensor(eps)
    r_new = - beta * r + (1.0 - beta) * (1.0 + torch.log(d_bar + eps))
    return r_new


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
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
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, run_name) for i in range(args.num_envs)],
    )
    assert isinstance(
        envs.single_action_space, gym.spaces.Discrete
    ), "only discrete action space is supported"

    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # --- Convex MDP: d_pi is a function of (s,a) and r is a function of (s,a) ---
    n_states = envs.single_observation_space.shape[0]
    action_space = envs.single_action_space.n

    # initialize d_bar uniformly and lambda_1 from it
    d_bar = torch.ones(n_states, action_space, device=device) / (n_states * action_space)
    
    # initialize d_bar randomly
    # d_bar = torch.rand(n_states, action_space, device=device)
    # d_bar = d_bar / d_bar.sum(dim=-1, keepdim=True)

    # ALGO Logic: Storage setup
    obs = torch.zeros(
        (args.num_steps, args.num_envs) + envs.single_observation_space.shape
    ).to(device)
    actions = torch.zeros(
        (args.num_steps, args.num_envs) + envs.single_action_space.shape
    ).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)  # will store shaped reward -lambda
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    ema_return_val = None

    for iteration in range(1, args.num_iterations + 1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        # === Rollout with fixed lambda_k (cost player FTL; lambda_k is fixed this iteration) ===
        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # Execute the game
            next_obs_np, env_reward, terminations, truncations, infos = envs.step(
                action.cpu().numpy()
            )
            # We keep env_reward only for logging; PPO optimization uses shaped reward.
            next_done = np.logical_or(terminations, truncations)

            # === Convex MDP reward shaping: r'_t = -lambda_k(s_t) ===
            # obs[step] is one-hot over states; use argmax to get state index.
            state_indices = obs[step].argmax(dim=-1).long()  # [num_envs]
            d_bar_value = d_bar[state_indices, action]
            shaped_reward = -lambda_reward_reshape(env_reward, d_bar_value, args.beta, args.d_bar_log_epsilon)
            rewards[step] = shaped_reward

            # Convert next_obs and done to tensors
            next_obs, next_done = (
                torch.Tensor(next_obs_np).to(device),
                torch.Tensor(next_done).to(device),
            )

            # Logging original environment episodic return (not the convex objective)
            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        writer.add_scalar(
                            "charts/episodic_return",
                            info["episode"]["r"],
                            global_step,
                        )
                        writer.add_scalar(
                            "charts/episodic_length",
                            info["episode"]["l"],
                            global_step,
                        )
                        # update EMA return
                        if ema_return_val is None:
                            ema_return_val = info["episode"]["r"]
                        else:
                            ema_return_val = ema_return(
                                ema_return_val, info["episode"]["r"]
                            )
                        writer.add_scalar(
                            "charts/ema_episodic_return",
                            ema_return_val,
                            global_step,
                        )
                        print(
                            f"global_step={global_step}, ema_episodic_return={ema_return_val}"
                        )

        # === GAE and returns use shaped reward r' ===
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
                delta = (
                    rewards[t]
                    + args.gamma * nextvalues * nextnonterminal
                    - values[t]
                )
                advantages[t] = lastgaelam = (
                    delta
                    + args.gamma
                    * args.gae_lambda
                    * nextnonterminal
                    * lastgaelam
                )
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb_inds], b_actions.long()[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [
                        ((ratio - 1.0).abs() > args.clip_coef)
                        .float()
                        .mean()
                        .item()
                    ]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                        mb_advantages.std() + 1e-8
                    )

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratio, 1 - args.clip_coef, 1 + args.clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
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
                    v_loss = 0.5 * (
                        (newvalue - b_returns[mb_inds]) ** 2
                    ).mean()

                entropy_loss = entropy.mean()
                loss = (
                    pg_loss
                    - args.ent_coef * entropy_loss
                    + v_loss * args.vf_coef
                )

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = (
            np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
        )

        # === After PPO update: update d_pi, d_bar (FTL), and lambda for next iteration ===
        with torch.no_grad():
            d_k = estimate_d_pi_from_obs_action(
                obs, actions, action_space, eps=args.d_bar_log_epsilon
            )
            d_bar = update_running_average(d_bar, d_k, iteration)

            # Optional logging of convex objective components
            entropy_d = -(d_bar * torch.log(d_bar + args.d_bar_log_epsilon)).sum()

            writer.add_scalar(
                "convex_mdp/entropy_d_bar", entropy_d.item(), global_step
            )

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar(
            "charts/learning_rate", optimizer.param_groups[0]["lr"], global_step
        )
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        writer.add_scalar(
            "charts/SPS", int(global_step / (time.time() - start_time)), global_step
        )

    envs.close()
    writer.close()
