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
import numpy as np
import minigrid
from minigrid.wrappers import FullyObsWrapper, ImgObsWrapper

########################################################################
############################ Begin Args ################################
########################################################################

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
    env_id: str = "CartPole-v1"
    """the id of the environment"""
    total_timesteps: int = 500000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 8
    """the number of parallel game environments"""
    num_steps: int = 128
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = False
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 4
    """the number of mini-batches"""
    update_epochs: int = 5
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.01
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""

    # SPMA specific arguments
    spma_eta: float = 0.7
    """SPMA mirror step size η used in π_t+1/2 = π_t (1 + η A)"""
    armijo_init_stepsize: float = 1.0
    armijo_beta: float = 0.5
    armijo_c: float = 1e-4
    armijo_max_backtracks: int = 10

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""

########################################################################
############################ End Args ##################################
########################################################################


########################################################################
############################ Begin Env #################################
########################################################################

def make_env(env_id, idx, capture_video, run_name):
    def thunk():
        # --- create env ---
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = FullyObsWrapper(env)
            env = ImgObsWrapper(env) 
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
            env = FullyObsWrapper(env)
            env = ImgObsWrapper(env) 

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

########################################################################
############################## End Env #################################
########################################################################

########################################################################
############################ Begin Agent ###############################
########################################################################

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
        x = x.view(x.shape[0], -1)   # FLATTEN
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        x = x.view(x.shape[0], -1)   # FLATTEN
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, logits, probs.log_prob(action), probs.entropy(), self.critic(x)
    
    def ema_return(self, previous_ema, new_return, alpha=0.9):
        return alpha * previous_ema + (1 - alpha) * new_return

########################################################################
############################## End Agent ###############################
########################################################################

if __name__ == "__main__":

    ########################################################################
    ############################ Begin Setup ###############################
    ########################################################################
    
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
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
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
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    ### SPMA: store logits of π_t for each (t, env)
    num_actions = envs.single_action_space.n
    policy_logits = torch.zeros((args.num_steps, args.num_envs, num_actions)).to(device)


    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    ########################################################################
    ############################## End Setup ###############################
    ########################################################################

    ########################################################################
    ########################### Begin Algorithm ############################
    ########################################################################

    ema_return_val = None

    for iteration in range(1, args.num_iterations + 1): # t <- 0 to T-1 in Alg. 1
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        ##########################################################################################################
        #### Begin 1. Interact with the environment using πt and form the surrogate function ℓt(θ) in Eq. (5) ####
        ##########################################################################################################

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                ### SPMA: explicit actor/critic forward to capture logits
                action, logits, logprob, _, value = agent.get_action_and_value(next_obs)

                values[step] = value.flatten()
                logprobs[step] = logprob
                policy_logits[step] = logits                 # SPMA: store π_t logits

            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                        writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                        # update EMA return
                        if ema_return_val is None:
                            ema_return_val = info["episode"]["r"]
                        else:
                            ema_return_val = agent.ema_return(ema_return_val, info["episode"]["r"])
                        writer.add_scalar("charts/ema_episodic_return", ema_return_val, global_step)
                        print(f"global_step={global_step}, ema_episodic_return={ema_return_val}")

        # bootstrap value if not done
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

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        ### SPMA: flatten logits from π_t
        b_policy_logits = policy_logits.reshape(args.batch_size, num_actions)  # [B, A]

        ##########################################################################################################
        ##### End 1. Interact with the environment using πt and form the surrogate function ℓt(θ) in Eq. (5) #####
        ##########################################################################################################

        ##########################################################################################################
        ##################### Begin SPMA: Precompute π_{t+1/2} soft targets (fixed during inner loop) ############
        ##########################################################################################################
        with torch.no_grad():
            # π_t(a|s) from stored logits
            b_old_probs = torch.softmax(b_policy_logits, dim=-1)  # [B, A]

            # action one-hot: shape [B, A]
            b_actions_flat = b_actions.long().view(-1)             # [B] doing nothing at the moment
            action_one_hot = torch.nn.functional.one_hot(
                b_actions_flat,
                num_classes=num_actions,
            ).float()

            # advantages for taken actions: shape [B]
            # NOTE: we use raw advantages (GAE), not normalized, to match A^{π_t} as much as possible
            eta = args.spma_eta

            # weights = 1 + η * A(s,a_t) * 1{a == a_t}
            # shape [B, A]
            weights = 1.0 + eta * b_advantages.unsqueeze(-1) * action_one_hot
            # ensure positivity (theory wants 1 + η A ≥ 0); clamp is a practical safeguard
            weights = torch.clamp(weights, min=1e-8) # section 3.2 MDP setting

            # unnormalized π_{t+1/2} ∝ π_t * weights
            unnormalized_targets = b_old_probs * weights           # [B, A]
            target_sums = unnormalized_targets.sum(dim=-1, keepdim=True)
            b_spma_targets = unnormalized_targets / target_sums    # π_{t+1/2} [B, A], soft labels for SPMA surrogate

        ##########################################################################################################
        ####################### End SPMA: Precompute π_{t+1/2} soft targets (fixed during inner loop) ############
        ##########################################################################################################

        ##########################################################################################################
        ################################# Begin 2. Initialize inner-loop: ω0 = θt ################################
        ##########################################################################################################

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        params = list(agent.parameters())
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]
                mb_target_probs = b_spma_targets[mb_inds]

                # ---- 1. Compute loss and gradient at current ω_k ----
                # Forward
                _, newlogits, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb_inds], b_actions.long()[mb_inds]
                )
                log_probs_all = torch.log_softmax(newlogits, dim=-1)

                policy_loss = -(mb_target_probs * log_probs_all).sum(dim=-1).mean()

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

                entropy_loss = 0.0
                loss = policy_loss - args.ent_coef * entropy_loss + args.vf_coef * v_loss

                optimizer.zero_grad()
                loss.backward()

                # Optional: gradient clipping BEFORE line search
                nn.utils.clip_grad_norm_(params, args.max_grad_norm)

                # Flatten gradient norm^2 (for Armijo RHS)
                with torch.no_grad():
                    grad_norm_sq = 0.0
                    for p in params:
                        if p.grad is not None:
                            grad_norm_sq += p.grad.pow(2).sum().item()

                L_curr = loss.item()
                if grad_norm_sq == 0.0:
                    # no gradient, just skip update
                    continue

                # ---- 2. Armijo backtracking line search for ζ ----
                with torch.no_grad():
                    # save current parameters
                    saved_params = [p.clone() for p in params]

                    step = args.armijo_init_stepsize
                    found = False

                    for _ in range(args.armijo_max_backtracks):
                        # trial step: ω_trial = ω - step * g
                        for p in params:
                            if p.grad is not None:
                                p.add_(p.grad, alpha=-step)

                        # recompute loss at trial point (no grad)
                        _, logits_trial, _, _, value_trial = agent.get_action_and_value(
                            b_obs[mb_inds], b_actions.long()[mb_inds]
                        )
                        log_probs_trial = torch.log_softmax(logits_trial, dim=-1)
                        policy_loss_trial = -(mb_target_probs * log_probs_trial).sum(dim=-1).mean()

                        value_trial = value_trial.view(-1)
                        if args.clip_vloss:
                            v_loss_unclipped_trial = (value_trial - b_returns[mb_inds]) ** 2
                            v_clipped_trial = b_values[mb_inds] + torch.clamp(
                                value_trial - b_values[mb_inds],
                                -args.clip_coef,
                                args.clip_coef,
                            )
                            v_loss_clipped_trial = (v_clipped_trial - b_returns[mb_inds]) ** 2
                            v_loss_max_trial = torch.max(v_loss_unclipped_trial, v_loss_clipped_trial)
                            v_loss_trial = 0.5 * v_loss_max_trial.mean()
                        else:
                            v_loss_trial = 0.5 * ((value_trial - b_returns[mb_inds]) ** 2).mean()

                        loss_trial = (
                            policy_loss_trial
                            - args.ent_coef * 0.0
                            + args.vf_coef * v_loss_trial
                        )

                        L_trial = loss_trial.item()

                        # Armijo condition:
                        # L(ω - step*g) <= L(ω) - c * step * ||g||^2
                        if L_trial <= L_curr - args.armijo_c * step * grad_norm_sq:
                            found = True
                            break

                        # Otherwise, revert params and shrink step
                        for p, p_saved in zip(params, saved_params):
                            p.copy_(p_saved)
                        step *= args.armijo_beta

                    if not found:
                        # if nothing passed, revert to original (no update)
                        for p, p_saved in zip(params, saved_params):
                            p.copy_(p_saved)



        ##########################################################################################################
        ################################## End 2. Initialize inner-loop: ω0 = θt #################################
        ##########################################################################################################

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", policy_loss.item(), global_step)
        # writer.add_scalar("losses/entropy_loss", entropy_loss.item(), global_step)
        # if args.target_kl is not None:
        #     writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    envs.close()
    writer.close()
