from typing import Any, Tuple

import numpy as np

from .done_tracker import DoneTrackerEnv


class SingleProcessEnv(DoneTrackerEnv):
    # TODO make compatible with custom env with continuous #1done
    def __init__(self, env_fn):
        super().__init__(num_envs=1)
        self.env = env_fn()
        self.num_actions = self.env.action_space.n  # expects Discrete
        self.num_continuous = self.env.continuous_action_space.shape[0]  # expects a flat Box

    def should_reset(self) -> bool:
        return self.num_envs_done == 1

    def reset(self) -> np.ndarray:
        self.reset_done_tracker()
        obs = self.env.reset()
        return obs[None, ...]

    def step(self, action) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Any]:
        # TODO expect 1 + self.num_continuous shaped action
        obs, reward, done, _ = self.env.step(action[0])  # action is supposed to be ndarray (1,)
        done = np.array([done])
        self.update_done_tracker(done)
        return obs[None, ...], np.array([reward]), done, None

    def render(self) -> None:
        self.env.render()

    def close(self) -> None:
        self.env.close()
