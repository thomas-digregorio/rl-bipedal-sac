import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder, VecNormalize
import os

def record_video(model_path="final_sb3_model.zip", stats_path="vec_normalize.pkl", video_folder="videos/final_run", length=2000):
    # 1. Create the environment
    env = DummyVecEnv([lambda: gym.make("BipedalWalker-v3", render_mode="rgb_array")])

    # 2. Load the normalization statistics (CRITICAL for correct performance)
    # The agent was trained with "glasses" (normalized obs), so we need to put them back on.
    if os.path.exists(stats_path):
        print(f"Loading normalization stats from {stats_path}...")
        env = VecNormalize.load(stats_path, env)
        # We don't need to update the stats during evaluation, only use them
        env.training = False
        env.norm_reward = False 
    else:
        print("Warning: No normalization stats found. The agent might perform poorly!")

    # 3. Wrap with Video Recorder
    obs = env.reset()
    env = VecVideoRecorder(
        env,
        video_folder,
        record_video_trigger=lambda x: x == 0,
        video_length=length,
        name_prefix="final_agent_demonstration"
    )

    # 4. Load the Agent
    print(f"Loading model from {model_path}...")
    model = SAC.load(model_path, env=env)

    # 5. Run the Agent
    print("Recording video...")
    obs = env.reset()
    for _ in range(length + 1):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, _, _ = env.step(action)

    env.close()
    print(f"Video saved to {video_folder}")

if __name__ == "__main__":
    record_video()
