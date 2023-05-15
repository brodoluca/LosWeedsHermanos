import gym
import numpy as np
import time

# Define the keys and their corresponding action vectors
KEYS_TO_ACTIONS = {
    "w": np.array([0, 1, 0]),
    "a": np.array([-1, 0, 0]),
    "s": np.array([0, 0, 0.8]),
    "d": np.array([1, 0, 0]),
    "wa": np.array([-1, 1, 0]),
    "dw": np.array([1, 1, 0]),
    "ds": np.array([1, 0, 0.8]),
    "as": np.array([-1, 0, 0.8]),
}

# Define the noop action vector
NOOP = np.array([0, 0, 0])

def play_continuous(env, zoom, keys_to_actions, noop, callback):
    """
    Play a gym environment with continuous control based on the time elapsed between successive key presses.

    :param env: The gym environment to play.
    :param zoom: The zoom level to use for rendering the environment.
    :param keys_to_actions: A dictionary mapping keyboard keys to action vectors.
    :param noop: The action vector to use when no keys are pressed.
    :param callback: A function to call after each step, with the elapsed time between successive key presses as argument.
    """
    # Initialize the time elapsed since each key was last pressed
    last_key_press_times = {key: 0 for key in keys_to_actions.keys()}

    # Start the game
    obs = env.reset()
    while True:
        # Render the environment
        # env.render(mode="human", zoom=zoom)

        # Check which keys are currently pressed
        keys_pressed = []
        for key in keys_to_actions.keys():
            if env.unwrapped.viewer.window.is_key_pressed(key):
                keys_pressed.append(key)

        # Determine the action vector based on the pressed keys and the time elapsed since they were last pressed
        action = noop.copy()
        for key in keys_pressed:
            time_elapsed = time.time() - last_key_press_times[key]
            last_key_press_times[key] = time.time()

            action += keys_to_actions[key] * max(0, min(time_elapsed, 1))

        # Take a step in the environment with the chosen action
        obs, _, done, _ = env.step(action)

        # Call the callback function with the elapsed time between successive key presses
        callback(time.time() - last_key_press_times[keys_pressed[0]] if len(keys_pressed) > 0 else 0)

        # Reset the environment if the episode is over
        if done:
            obs = env.reset()
            last_key_press_times = {key: 0 for key in keys_to_actions.keys()}

def callback(elapsed_time):
    print("Time elapsed:", elapsed_time)

env = gym.make("CarRacing-v2", render_mode="rgb_array")
play_continuous(env, zoom=2, keys_to_actions=KEYS_TO_ACTIONS, noop=NOOP, callback=callback)
