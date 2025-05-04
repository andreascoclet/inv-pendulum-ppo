# testing.py
from __future__ import annotations
import time
import numpy as np
import torch
from torch.distributions import Normal


from env import InvertedPendulumEnv
from train import PPOAgent, PPOConfig

@torch.no_grad()
def run_optimal(
    path: str,
    episodes: int = 5,
    max_steps: int = 6400,
    render: bool = False
) -> None:
    device = (
        torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    
    cfg   = PPOConfig()
    env = InvertedPendulumEnv(use_viewer=render, max_steps=max_steps)
    obs_dim = env.obs().shape[0]
    act_dim = 1

    
    agent = PPOAgent(obs_dim, act_dim, cfg, device)

    # load checkpoint
    ckpt = torch.load(path, map_location=device)
    agent.net.load_state_dict(ckpt["model_state_dict"])
    agent.net.eval()

    for ep in range(episodes):
        obs = env.reset_model()
        done = False
        total_reward = 0.0
        while not done:
            action, _, _ = agent.act(obs)
            obs, reward, done = env.step(3 * np.clip(action, -1, 1))

            if render:
                time.sleep(0.02)
            total_reward += reward

        print(f"Episode {ep+1}: return = {total_reward:.2f}")


if __name__ == "__main__":
    run_optimal(
         path="checkpoints/ppo_ipendulum_upd_complex_reward_centered_to_ball_200.pt", 
        episodes=1, 
        max_steps=6400, 
        render=True)
    