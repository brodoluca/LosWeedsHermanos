import gym
import numpy as np
from gym.utils.play import play

def callback(obs_t, obs_tp1, action, rew, terminated, truncated, info):
       print("action: ", action, "reward: ", rew)

play(gym.make("CarRacing-v2", render_mode="rgb_array"), zoom=2, callback=callback, keys_to_action={
                                               "w": np.array([0, 1, 0]),
                                               "a": np.array([-1, 0, 0]),
                                               "s": np.array([0, 0, 0.8]),
                                               "d": np.array([1, 0, 0]),
                                               "wa": np.array([-1, 1, 0]),
                                               "dw": np.array([1, 1, 0]),
                                               "ds": np.array([1, 0, 0.8]),
                                               "as": np.array([-1, 0, 0.8]),
                                              }, noop=np.array([0,0,0]))
