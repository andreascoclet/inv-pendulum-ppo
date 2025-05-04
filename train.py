# train.py
from __future__ import annotations
import os
import time
from dotenv import load_dotenv
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.distributions import Normal
from tqdm import trange
import wandb
from dataclasses import asdict, dataclass
from typing import Tuple

from env import InvertedPendulumEnv
from model import ActorCritic

# ——— PPO configuration —————————————————————————————————
@dataclass
class PPOConfig:
    total_steps: int = 64_000_000
    horizon: int = 25600
    num_updates: int = total_steps // horizon
    batch_size: int = 4096
    max_env_steps: int = 1000
    epochs: int = 10
    gamma: float = 0.95
    lam: float = 0.95
    clip_eps: float = 0.1
    vf_coef: float = 0.5
    ent_coef: float = 0.0005
    ent_coef_init: float = 1e-2
    ent_coef_final: float = 1e-4
    ent_warmup_updates: int = 64
    lr: float = 3e-4
    lr_warmup_updates: int = 500
    max_grad_norm: float = 0.5
    seed: int = 42
    save_path: str = "checkpoints/ppo_ipendulum_upd_complex_reward_centered_to_ball_"
    wandb_project: str = "InvertedPendulum-PPO"
    wandb_entity: str | None = None
    wandb_api_key: str | None = None
    hidden_sizes: Tuple[int, int] = (64, 64)


# ——— Replay buffer —————————————————————————————————————
class RolloutBuffer:
    def __init__(self, obs_dim: int, act_dim: int, size: int, device: torch.device):
        self.obs     = torch.zeros((size, obs_dim),   dtype=torch.float32, device=device)
        self.actions = torch.zeros((size, act_dim),   dtype=torch.float32, device=device)
        self.logps   = torch.zeros(size, dtype=torch.float32, device=device)
        self.rewards = torch.zeros(size, dtype=torch.float32, device=device)
        self.dones   = torch.zeros(size, dtype=torch.float32, device=device)
        self.values  = torch.zeros(size, dtype=torch.float32, device=device)
        self.ptr, self.max_size = 0, size

    def store(self, obs, action, logp, reward, done, value):
        i = self.ptr
        self.obs[i]     = obs
        self.actions[i] = action
        self.logps[i]   = logp
        self.rewards[i] = reward
        self.dones[i]   = done
        self.values[i]  = value
        self.ptr += 1

    def is_full(self):
        return self.ptr == self.max_size

    def finish_path(self, last_val: torch.Tensor, gamma: float, lam: float):
        adv = torch.zeros_like(self.rewards)
        gae = 0.0
        for t in reversed(range(self.max_size)):
            next_val = last_val if t == self.max_size - 1 else self.values[t + 1]
            delta = self.rewards[t] + gamma * next_val * (1 - self.dones[t]) - self.values[t]
            gae = delta + gamma * lam * (1 - self.dones[t]) * gae
            adv[t] = gae
        self.returns = adv + self.values
        self.adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    def batches(self, batch_size: int):
        idxs = np.random.permutation(self.max_size)
        for start in range(0, self.max_size, batch_size):
            mb = idxs[start : start + batch_size]
            yield (
                self.obs[mb],
                self.actions[mb],
                self.logps[mb],
                self.adv[mb],
                self.returns[mb],
                self.values[mb],
            )


# ——— PPO agent ——————————————————————————————————————
class PPOAgent:
    def __init__(self, obs_dim: int, act_dim: int, cfg: PPOConfig, device: torch.device):
        self.net     = ActorCritic(obs_dim, act_dim, cfg.hidden_sizes).to(device)
        self.opt     = optim.Adam(self.net.parameters(), lr=cfg.lr)
        self.cfg     = cfg
        self.device  = device

    @torch.no_grad()
    def act(self, obs: np.ndarray) -> Tuple[np.ndarray, torch.Tensor, torch.Tensor]:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        mu, std, val = self.net(obs_t)
        dist = Normal(mu, std)
        action = dist.sample()
        logp   = dist.log_prob(action).sum(-1)
        return action.cpu().numpy(), logp, val

    def update(self, buf: RolloutBuffer, ent_coef: float | None = None) -> dict:
        if ent_coef is None:
            ent_coef = self.cfg.ent_coef

        policy_losses, value_losses, entropies, total_losses = [], [], [], []
        for _ in range(self.cfg.epochs):
            for obs_b, act_b, old_logp_b, adv_b, ret_b, val_b in buf.batches(self.cfg.batch_size):
                mu, std, val = self.net(obs_b)
                dist = Normal(mu, std)
                logp = dist.log_prob(act_b).sum(-1)
                ratio = torch.exp(logp - old_logp_b)

                # clipped surrogate objective
                surr1 = ratio * adv_b
                surr2 = torch.clamp(ratio, 1 - self.cfg.clip_eps, 1 + self.cfg.clip_eps) * adv_b
                policy_loss = -torch.min(surr1, surr2).mean()

                # value function loss (clipped)
                val_clipped = val_b + (val - val_b).clamp(-self.cfg.clip_eps, self.cfg.clip_eps)
                vf_loss1 = (val - ret_b).pow(2)
                vf_loss2 = (val_clipped - ret_b).pow(2)
                value_loss = 0.5 * torch.max(vf_loss1, vf_loss2).mean()

                entropy = dist.entropy().mean()
                loss = policy_loss + self.cfg.vf_coef * value_loss - ent_coef * entropy

                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), self.cfg.max_grad_norm)
                self.opt.step()

                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropies.append(entropy.item())
                total_losses.append(loss.item())

        return {
            "loss/policy":  float(np.mean(policy_losses)),
            "loss/value":   float(np.mean(value_losses)),
            "loss/entropy": float(np.mean(entropies)),
            "loss/total":   float(np.mean(total_losses)),
        }


# ——— Learning‐rate & entropy schedules —————————————————————
def warmup_const_cosine(update_step: int, cfg: PPOConfig) -> float:
    warmup = cfg.lr_warmup_updates
    if update_step < warmup:
        return 1.0
    prog = (update_step - warmup) / (cfg.num_updates - warmup)
    return 0.5 * (1 + math.cos(math.pi * prog))

def entropy_schedule(update_step: int, cfg: PPOConfig) -> float:
    if update_step < cfg.ent_warmup_updates:
        return cfg.ent_coef_init
    prog = (update_step - cfg.ent_warmup_updates) / (cfg.num_updates - cfg.ent_warmup_updates)
    cosv = 0.5 * (1 + math.cos(math.pi * prog))
    return cfg.ent_coef_final + (cfg.ent_coef_init - cfg.ent_coef_final) * cosv


# ——— WandB initialization —————————————————————————————————
def init_wandb(cfg: PPOConfig, resume_step: int | None = None, ckpt_path: str | None = None):
    load_dotenv()
    if cfg.wandb_api_key:
        os.environ.setdefault("WANDB_API_KEY", cfg.wandb_api_key)
    wandb_name = time.strftime("run_%Y%m%d_%H%M%S")
    if ckpt_path and resume_step is not None:
        return wandb.init(
            project=cfg.wandb_project,
            entity=cfg.wandb_entity,
            config=asdict(cfg),
            resume="allow",
            name=wandb_name,
            save_code=True,
        )
    return wandb.init(
        project=cfg.wandb_project,
        entity=cfg.wandb_entity,
        config=asdict(cfg),
        name=wandb_name,
        save_code=True,
    )


# ——— Checkpointing & evaluation —————————————————————————————
def save_checkpoint(agent: PPOAgent, step: int, path: str) -> None:
    torch.save({"model_state_dict": agent.net.state_dict(), "step": step}, path)

@torch.no_grad()
def evaluate(agent: PPOAgent, env: InvertedPendulumEnv, steps: int = 1000, render: bool = False) -> float:
    obs = env.reset_model()
    total_reward = 0.0
    for _ in range(steps):
        action, _, _ = agent.act(obs)
        obs, reward, done = env.step(3 * np.clip(action, -1, 1))
        if render:
            time.sleep(0.02)
        total_reward += reward
        if done:
            break
    return total_reward


# ——— Main training loop ————————————————————————————————————
def train(seed: int = 42, ckpt_path: str | None = None, resume_step: int | None = None) -> None:
    cfg = PPOConfig()
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    device = (
        torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("mps")
        if torch.backends.mps.is_available()
        else torch.device("cpu")
    )

    
    env = InvertedPendulumEnv(use_viewer=False, max_steps=cfg.max_env_steps)
    obs_dim = env.obs().shape[0]
    act_dim = 1

    wb = init_wandb(cfg, resume_step, ckpt_path)
    agent = PPOAgent(obs_dim, act_dim, cfg, device)
    buf   = RolloutBuffer(obs_dim, act_dim, cfg.horizon, device)
    scheduler = LambdaLR(agent.opt, lr_lambda=lambda step: warmup_const_cosine(step, cfg))

    global_step = resume_step or 0
    if ckpt_path and resume_step is not None:
        ckpt = torch.load(ckpt_path, map_location=device)
        agent.net.load_state_dict(ckpt["model_state_dict"])
        print(f"Resumed training from step {global_step}")

    obs = env.reset_model()
    pbar = trange(cfg.num_updates, desc="training", initial=global_step, dynamic_ncols=True)

    for _ in pbar:
        obs = env.reset_model()
        buf.ptr = 0

        # collect rollout
        for _ in range(cfg.horizon):
            action, logp, value = agent.act(obs)
            nxt_obs, reward, done = env.step(3 * np.clip(action, -1, 1))
            buf.store(torch.as_tensor(obs, dtype=torch.float32, device=device),
                    torch.as_tensor(action, dtype=torch.float32, device=device),
                    torch.as_tensor(logp, dtype=torch.float32, device=device),                    
                    torch.as_tensor(reward, dtype=torch.float32, device=device),
                    torch.as_tensor(done, dtype=torch.bool, device=device),
                    torch.as_tensor(value, device=device))
            obs = nxt_obs

            if done:
                obs = env.reset_model()

        # finish path & update
        _, _, last_val = agent.act(obs)
        buf.finish_path(last_val, cfg.gamma, cfg.lam)
        metrics = agent.update(buf)#, ent_coef=entropy_schedule(global_step, cfg))
        scheduler.step()

        # evaluation
        eval_ret = evaluate(agent, env, steps=cfg.horizon)
        metrics.update({
            "eval_return": eval_ret,
            "lr": scheduler.get_last_lr()[0],
            "step_update": global_step,
        })

        wb.log(metrics, step=global_step)
        pbar.set_postfix(eval_return=eval_ret)

        if (global_step + 1) % 1 == 0:
            ckpt_file = f"{cfg.save_path}{global_step+1}.pt"
            print(f"saving weight to ... {ckpt_file}")
            save_checkpoint(agent, global_step + 1, ckpt_file)

        global_step += 1

    # final checkpoint
    save_checkpoint(agent, cfg.num_updates, f"{cfg.save_path}final.pt")
    wb.finish()


if __name__ == "__main__":
    train()