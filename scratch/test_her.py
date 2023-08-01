from scratch.custom_her_env import BitFlippingEnv
from stable_baselines3 import HerReplayBuffer, DDPG


if __name__ == "__main__":
    env = BitFlippingEnv(continuous=True)

    model = DDPG(
        "MultiInputPolicy",
        env,
        replay_buffer_class=HerReplayBuffer,
        learning_starts=1000,
        verbose=1,
    )

    model.learn(3000, progress_bar=True)
