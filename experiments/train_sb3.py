import gymnasium as gym
import torch as th
import wandb
import os
import argparse
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
from stable_baselines3.common.torch_layers import FlattenExtractor
from wandb.integration.sb3 import WandbCallback

def make_env(env_id, seed, video_folder):
    def _init():
        env = gym.make(env_id, render_mode="rgb_array")
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env
    return _init

def train_sb3():
    parser = argparse.ArgumentParser()
    parser.add_argument("--total_timesteps", type=int, default=1000000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb_project", type=str, default="bipedal-sac")
    parser.add_argument("--wandb_entity", type=str, default=None)
    args = parser.parse_args()

    # W&B Init
    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        config=vars(args),
        sync_tensorboard=True, # Auto-upload sb3 tensorboard metrics
        name=f"sb3_sac_{args.seed}",
        monitor_gym=True,
        save_code=True,
    )

    # Environment
    env = DummyVecEnv([make_env("BipedalWalker-v3", args.seed, "videos/sb3")])
    env = VecVideoRecorder(
        env,
        f"videos/sb3/{run.id}",
        record_video_trigger=lambda x: x % 50000 == 0,
        video_length=2000
    )
    
    # Sync videos
    wandb.save(f"videos/sb3/{run.id}/*.mp4", base_path=f"videos/sb3/{run.id}", policy="live")

    # Architecture matching Custom Agent
    policy_kwargs = dict(
        activation_fn=th.nn.SiLU,
        net_arch=[256, 256],
        features_extractor_class=FlattenExtractor
    )

    model = SAC(
        "MlpPolicy",
        env,
        verbose=1,
        seed=args.seed,
        device="cuda" if th.cuda.is_available() else "cpu",
        policy_kwargs=policy_kwargs,
        tensorboard_log=f"runs/{run.id}"
    )

    model.learn(
        total_timesteps=args.total_timesteps,
        callback=WandbCallback(
            gradient_save_freq=100,
            model_save_path=f"models/{run.id}",
            verbose=2,
        )
    )

    model.save("final_sb3_model")
    env.close()
    wandb.finish()

if __name__ == "__main__":
    train_sb3()
