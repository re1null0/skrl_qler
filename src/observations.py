import numpy as np
import gymnasium as gym
from abc import ABC, abstractmethod

class Observation(ABC):
    """Base class for observation helpers."""

    def __init__(self, env):
        self.env = env

    @abstractmethod
    def space(self) -> gym.spaces.Space:
        """Return the observation space produced by :meth:`observe`."""
        raise NotImplementedError

    @abstractmethod
    def observe(self, obs):
        """Convert a raw environment observation into the desired format."""
        raise NotImplementedError

class RawObservation(Observation):
    """Return environment observations unchanged."""

    def space(self) -> gym.spaces.Space:
        return self.env.observation_space

    def observe(self, obs):
        return obs

class VectorObservation(Observation):
    """Build a flat vector from selected observation features."""

    DEFAULT_FEATURES = (
        "scan",
        "pose_x",
        "pose_y",
        "pose_yaw",
        "linear_vel",
        "steering_angle",
    )

    def __init__(self, env, features=None):
        super().__init__(env)
        self.features = tuple(features) if features is not None else self.DEFAULT_FEATURES
        self._getters = [self._make_getter(name) for name in self.features]
        size = sum(g(None, shape=True) for g in self._getters)
        self._space = gym.spaces.Box(-np.inf, np.inf, shape=(size,), dtype=np.float32)

    def _make_getter(self, name):
        if name == "scan":
            space = self.env.observation_space.get("scan", None)
            size = int(space.shape[0]) if space is not None else 0
            def f(obs, shape=False):
                if shape:
                    return size
                return np.asarray(obs.get("scan", np.zeros(size)), dtype=np.float32)
            return f
        elif name == "pose_x":
            def f(obs, shape=False):
                if shape:
                    return 1
                try:
                    x = float(self.env.unwrapped.sim.agent_poses[0][0])
                except Exception:
                    x = 0.0
                return np.asarray([x], dtype=np.float32)
            return f
        elif name == "pose_y":
            def f(obs, shape=False):
                if shape:
                    return 1
                try:
                    y = float(self.env.unwrapped.sim.agent_poses[0][1])
                except Exception:
                    y = 0.0
                return np.asarray([y], dtype=np.float32)
            return f
        elif name == "pose_yaw":
            def f(obs, shape=False):
                if shape:
                    return 1
                try:
                    yaw = float(self.env.unwrapped.sim.agent_poses[0][2])
                except Exception:
                    yaw = 0.0
                return np.asarray([yaw], dtype=np.float32)
            return f
        elif name == "linear_vel":
            def f(obs, shape=False):
                if shape:
                    return 1
                val = None
                if obs is not None:
                    val = obs.get("linear_vel")
                    if isinstance(val, np.ndarray):
                        val = float(val[0])
                if val is None:
                    try:
                        val = float(self.env.unwrapped.sim.velocities[0])
                    except Exception:
                        val = 0.0
                return np.asarray([val], dtype=np.float32)
            return f
        elif name == "steering_angle":
            def f(obs, shape=False):
                if shape:
                    return 1
                val = None
                if obs is not None:
                    val = obs.get("steering_angle")
                    if isinstance(val, np.ndarray):
                        val = float(val[0])
                if val is None:
                    try:
                        val = float(self.env.unwrapped.sim.agent_states[0][6])
                    except Exception:
                        val = 0.0
                return np.asarray([val], dtype=np.float32)
            return f
        else:
            raise ValueError(f"Unknown feature '{name}'")

    def space(self) -> gym.spaces.Space:
        return self._space

    def observe(self, obs):
        parts = [g(obs) for g in self._getters]
        return np.concatenate(parts, axis=0)

def observation_factory(env, type_: str | None, **kwargs) -> Observation:
    """Return an observation helper instance for ``env``."""
    if type_ is None or type_ == "raw":
        return RawObservation(env)
    type_low = type_.lower()
    if type_low == "vector":
        return VectorObservation(env, **kwargs)
    raise ValueError(f"Unknown observation type: {type_}")
