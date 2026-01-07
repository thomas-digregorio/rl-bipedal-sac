import gymnasium as gym
import torch as th
import wandb
import os
import argparse
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder, VecNormalize
from stable_baselines3.common.torch_layers import FlattenExtractor
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

class StopTrainingCallback(BaseCallback):
    """
    Stop the training once a threshold in episodic reward is reached.
    """
    def __init__(self, reward_threshold: float = 300, verbose: int = 0):
        super().__init__(verbose)
        self.reward_threshold = reward_threshold

    def _on_step(self) -> bool:
        # Check every 1000 steps
        if self.n_calls % 1000 == 0:
            if len(self.model.ep_info_buffer) > 0:
                mean_reward = sum(ep['r'] for ep in self.model.ep_info_buffer) / len(self.model.ep_info_buffer)
                if mean_reward > self.reward_threshold:
                    if self.verbose > 0:
                        print(f"Stopping training because reward {mean_reward:.2f} > {self.reward_threshold}")
                    return False
        return True

class WandbUploadCallback(BaseCallback):
    """
    Explicitly upload model checkpoints as W&B Artifacts.
    Avoids reliance on flaky 'live' filesystem watchers.
    """
    def __init__(self, model_dir, run_id, save_freq=12500, verbose=0):
        super().__init__(verbose)
        self.model_dir = model_dir
        self.run_id = run_id
        self.save_freq = save_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            # Wait for file system flush
            import time
            time.sleep(2.0)
            
            try:
                import glob
                # Find latest zip (Use absolute path to be safe)
                abs_model_dir = os.path.abspath(self.model_dir)
                zips = glob.glob(os.path.join(abs_model_dir, "*.zip"))
                if zips:
                    latest_zip = max(zips, key=os.path.getmtime)
                    # Create artifact
                    step_count = self.num_timesteps
                    artifact = wandb.Artifact(name=f"model_step_{step_count}", type="model")
                    artifact.add_file(latest_zip)
                    
                    # Find matching pkl
                    latest_pkl = latest_zip.replace("rl_model_", "rl_model_vecnormalize_").replace(".zip", ".pkl")
                    if os.path.exists(latest_pkl):
                         artifact.add_file(latest_pkl)
                    
                    wandb.log_artifact(artifact)
                    if self.verbose > 0:
                        print(f"Uploaded W&B Artifact: {latest_zip}")
                elif self.verbose > 0:
                    print(f"WandbUploadCallback: No zips found in {abs_model_dir}")
            except Exception as e:
                print(f"Failed to upload artifact: {e}")
        return True

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
    parser.add_argument("--total_timesteps", type=int, default=2000000) # Optimization: Run longer
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb_project", type=str, default="bipedal-sac")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--load_model", type=str, default=None, help="Path to model.zip to resume training")
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
    # Environment (Parallel)
    vec_env = make_vec_env(
        "BipedalWalker-v3",
        n_envs=4,
        seed=args.seed,
        vec_env_cls=SubprocVecEnv
    )

    # Optimization: Normalize Observations and Rewards ("Clean Glasses")
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    # Record video on the first environment only
    env = VecVideoRecorder(
        vec_env,
        f"videos/sb3/{run.id}",
        record_video_trigger=lambda x: x % 10000 == 0,
        video_length=2000
    )
    
    # Sync videos
    # Critical Fix: Create directories BEFORE starting the watcher or it fails silently
    os.makedirs(f"videos/sb3/{run.id}", exist_ok=True)
    os.makedirs(f"models/{run.id}", exist_ok=True)

    wandb.save(f"videos/sb3/{run.id}/*.mp4", base_path=f"videos/sb3/{run.id}", policy="live")
    # Restore Watcher as Backup Layer
    wandb.save(f"models/{run.id}/*.zip", base_path=f"models/{run.id}", policy="live") 
    wandb.save(f"models/{run.id}/*.pkl", base_path=f"models/{run.id}", policy="live")

    # Architecture matching Custom Agent
    policy_kwargs = dict(
        activation_fn=th.nn.SiLU,
        net_arch=[256, 256],
        features_extractor_class=FlattenExtractor
    )

    if args.load_model:
        # Resume training
        print(f"Resuming training from: {args.load_model}")
        model = SAC.load(
            args.load_model,
            env=env,
            device="cuda" if th.cuda.is_available() else "cpu",
            tensorboard_log=f"runs/{run.id}"
            # policy_kwargs are loaded from the file
            # ent_coef is loaded from the file
        )
        
        # Try to load VecNormalize stats
        # Assumption: CheckpointCallback saves as 'rl_model_X_steps.zip' and 'rl_model_vecnormalize_X_steps.pkl'
        # We need to construct the pkl path from the zip path
        stats_path = args.load_model.replace(".zip", "")
        # Handle the specific naming pattern of CheckpointCallback (inserting _vecnormalize before steps)
        # Standard pattern: prefix_steps.zip -> prefix_vecnormalize_steps.pkl
        # Heuristic: Replace "rl_model_" with "rl_model_vecnormalize_"? 
        # Or just try the direct substitution if manual save:
        
        # Approach 1: Try adjacent .pkl file with same basename (Manual save)
        pk_path_manual = args.load_model.replace(".zip", ".pkl")
        
        # Approach 2: CheckpointCallback pattern
        # "models/runid/rl_model_50000_steps.zip" -> "models/runid/rl_model_vecnormalize_50000_steps.pkl"
        path_parts = args.load_model.split(os.sep)
        filename = path_parts[-1]
        if "rl_model_" in filename:
            new_filename = filename.replace("rl_model_", "rl_model_vecnormalize_").replace(".zip", ".pkl")
            pk_path_callback = os.path.join(*path_parts[:-1], new_filename)
            if path_parts[0] == "": # Fix absolute path join issue
                 pk_path_callback = "/" + pk_path_callback
        else:
            pk_path_callback = "impossible_path"

        if os.path.exists(pk_path_manual):
            print(f"Loading VecNormalize stats from {pk_path_manual}")
            vec_env = VecNormalize.load(pk_path_manual, vec_env)
            # Re-wrap the env for the model
            model.set_env(env)
        elif os.path.exists(pk_path_callback):
            print(f"Loading VecNormalize stats from {pk_path_callback}")
            vec_env = VecNormalize.load(pk_path_callback, vec_env)
            model.set_env(env)
        else:
            print("WARNING: Could not find VecNormalize stats (.pkl). Retraining with fresh stats (Suboptimal!)")

    else:
        # Start fresh
        model = SAC(
            "MlpPolicy",
            env,
            verbose=1,
            seed=args.seed,
            device="cuda" if th.cuda.is_available() else "cpu",
            policy_kwargs=policy_kwargs,
            batch_size=256,
            gradient_steps=4,       # Optimize: Match custom agent (1 update per step per env)
            ent_coef="auto",
            tensorboard_log=f"runs/{run.id}"
        )

    # Create Checkpoint Callback (Saves to disk every 50k steps)
    checkpoint_callback = CheckpointCallback(
        save_freq=12500, # 50,000 steps / 4 envs = 12,500
        save_path=f"models/{run.id}",
        name_prefix="rl_model",
        save_replay_buffer=False,
        save_vecnormalize=True,
    )

    # W&B Callback (Logging only)
    wandb_callback = WandbCallback(
        verbose=2,
    )

    # Stop Callback (Stop if > 300)
    stop_callback = StopTrainingCallback(reward_threshold=300, verbose=1)

    # Upload Callback (Explicit Artifact Upload)
    upload_callback = WandbUploadCallback(
        model_dir=f"models/{run.id}",
        run_id=run.id,
        save_freq=12500,
        verbose=1
    )

    model.learn(
        total_timesteps=args.total_timesteps,
        callback=CallbackList([checkpoint_callback, wandb_callback, stop_callback, upload_callback])
    )

    model.save("final_sb3_model")
    vec_env.save("vec_normalize.pkl")
    env.close()
    wandb.finish()

if __name__ == "__main__":
    train_sb3()
