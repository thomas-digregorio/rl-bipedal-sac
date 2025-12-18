import argparse
import gymnasium as gym
import numpy as np
import torch
import wandb
import os
import time
from src.agent import SAC
from src.buffers import ReplayBuffer

def parse_args():
    parser = argparse.ArgumentParser(description="Train Custom SAC Agent")
    # Sweep parameters (must match sweep.yaml)
    parser.add_argument("--env_id", type=str, default="BipedalWalker-v3")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--total_timesteps", type=int, default=1000000)
    parser.add_argument("--policy_lr", type=float, default=3e-4)
    parser.add_argument("--q_lr", type=float, default=1e-3)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--buffer_size", type=int, default=1000000)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--target_entropy", type=str, default="auto") # "auto" or float string
    parser.add_argument("--gradient_steps", type=int, default=1)
    parser.add_argument("--train_freq", type=int, default=1)
    parser.add_argument("--use_wandb", type=bool, default=True)
    parser.add_argument("--wandb_project", type=str, default="bipedal-sac")
    return parser.parse_args()

def make_env(env_id, seed, run_name):
    env = gym.make(env_id, render_mode="rgb_array")
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = gym.wrappers.ClipAction(env)
    # Record videos
    trigger = lambda t: t % 50000 == 0
    env = gym.wrappers.RecordVideo(env, f"videos/{run_name}", episode_trigger=trigger, disable_logger=True)
    
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    return env

def train():
    args = parse_args()
    run_name = f"custom_sac_{args.env_id}_{args.seed}_{int(time.time())}"

    if args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )

    # Setup Checkpointing
    ckpt_dir = "checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, "agent.pt")

    # Environment
    env = make_env(args.env_id, args.seed, run_name)
    
    # Sync videos to W&B
    if args.use_wandb:
        # Use policy="live" to sync files as they are written
        wandb.save(f"videos/{run_name}/*.mp4", base_path=f"videos/{run_name}", policy="live")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Agent & Buffer
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    buffer = ReplayBuffer(args.buffer_size, state_dim, action_dim, device)
    
    target_ent = None
    if args.target_entropy != "auto":
        target_ent = float(args.target_entropy)

    agent = SAC(
        state_dim, action_dim, device, buffer,
        hidden_dim=args.hidden_dim,
        policy_lr=args.policy_lr,
        q_lr=args.q_lr,
        gamma=args.gamma,
        tau=args.tau,
        target_entropy=target_ent
    )

    if args.use_wandb:
        wandb.watch(agent.actor, log="all", log_freq=1000)

    # Training Loop
    obs, _ = env.reset(seed=args.seed)
    start_time = time.time()
    
    for global_step in range(args.total_timesteps):
        # 1. ALGO LOGIC: Action selection
        if global_step < 10000: # Warmup
            action = env.action_space.sample()
        else:
            action = agent.select_action(obs)

        # 2. Execute step
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Add to buffer
        # Note: 'done' for bootstrap handling:
        # If truncated (time limit), real_done is False.
        real_done = terminated
        buffer.add(obs, action, reward, next_obs, real_done)

        obs = next_obs

        # 3. Train
        if global_step >= 10000:
            if global_step % args.train_freq == 0:
                for _ in range(args.gradient_steps):
                    metrics = agent.update(args.batch_size)
                    if args.use_wandb and metrics:
                         wandb.log(metrics, commit=False)

        # Logging (Episode End)
        if done:
            if "episode" in info:
                print(f"Step: {global_step}, Reward: {info['episode']['r']}, Length: {info['episode']['l']}")
                if args.use_wandb:
                    wandb.log({
                        "charts/episodic_return": info["episode"]["r"],
                        "charts/episodic_length": info["episode"]["l"],
                        "charts/SPS": int(global_step / (time.time() - start_time))
                    })
            obs, _ = env.reset()

        # Checkpoint
        if global_step % 50000 == 0:
            agent.save_checkpoint(ckpt_path)
            if args.use_wandb:
                wandb.save(ckpt_path)

    env.close()
    if args.use_wandb:
        wandb.finish()

if __name__ == "__main__":
    train()
