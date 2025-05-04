# env.py
from __future__ import annotations

import numpy as np
import warnings
import mujoco
import mujoco.viewer
from typing import Union, Tuple


def _sigmoids(x: np.ndarray, value_at_1: float, sigmoid: str) -> np.ndarray:
    if sigmoid in ('cosine', 'linear', 'quadratic'):
        if not 0 <= value_at_1 < 1:
            raise ValueError(f"`value_at_1` must be in [0, 1), got {value_at_1}.")
    else:
        if not 0 < value_at_1 < 1:
            raise ValueError(f"`value_at_1` must be in (0, 1), got {value_at_1}.")

    if sigmoid == 'gaussian':
        scale = np.sqrt(-2 * np.log(value_at_1))
        return np.exp(-0.5 * (x * scale)**2)
    elif sigmoid == 'hyperbolic':
        scale = np.arccosh(1 / value_at_1)
        return 1 / np.cosh(x * scale)
    elif sigmoid == 'long_tail':
        scale = np.sqrt(1 / value_at_1 - 1)
        return 1 / ((x * scale)**2 + 1)
    elif sigmoid == 'reciprocal':
        scale = 1 / value_at_1 - 1
        return 1 / (np.abs(x) * scale + 1)
    elif sigmoid == 'cosine':
        scale = np.arccos(2 * value_at_1 - 1) / np.pi
        scaled_x = x * scale
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', 'invalid value encountered in cos')
            cos_pi_scaled_x = np.cos(np.pi * scaled_x)
        return np.where(np.abs(scaled_x) < 1, (1 + cos_pi_scaled_x) / 2, 0.0)
    elif sigmoid == 'linear':
        scale = 1 - value_at_1
        scaled_x = x * scale
        return np.where(np.abs(scaled_x) < 1, 1 - scaled_x, 0.0)
    elif sigmoid == 'quadratic':
        scale = np.sqrt(1 - value_at_1)
        scaled_x = x * scale
        return np.where(np.abs(scaled_x) < 1, 1 - scaled_x**2, 0.0)
    elif sigmoid == 'tanh_squared':
        scale = np.arctanh(np.sqrt(1 - value_at_1))
        return 1 - np.tanh(x * scale)**2
    else:
        raise ValueError(f"Unknown sigmoid type: {sigmoid}")


def tolerance(
    x: Union[float, np.ndarray],
    bounds: Tuple[float, float] = (0.0, 0.0),
    margin: float = 0.0,
    sigmoid: str = 'gaussian',
    value_at_margin: float = 0.1,
) -> Union[float, np.ndarray]:
    lower, upper = bounds
    if lower > upper:
        raise ValueError(f"Lower bound {lower} must be <= upper bound {upper}.")
    if margin < 0:
        raise ValueError(f"Margin {margin} must be non-negative.")

    x_arr = np.asarray(x)
    in_bounds = np.logical_and(lower <= x_arr, x_arr <= upper)

    if margin == 0.0:
        return np.where(in_bounds, 1.0, 0.0)

    d = np.where(x_arr < lower, lower - x_arr, x_arr - upper) / margin
    d = np.maximum(d, 0.0)
    out_of_bounds_reward = _sigmoids(d, value_at_margin, sigmoid)
    reward = np.where(in_bounds, 1.0, out_of_bounds_reward)
    return float(reward) if reward.shape == () else reward


class InvertedPendulumEnv:
    """Inverted Pendulum environment with smooth complex reward function."""
    xml_env: str = """
    <mujoco model="inverted pendulum">
        <visual>
            <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>
            <rgba haze="0.15 0.25 0.35 1"/>
            <global azimuth="160" elevation="-20"/>
        </visual>
        <asset>
            <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>
        </asset>
        <compiler inertiafromgeom="true"/>
        <default>
            <joint armature="0" damping="1" limited="true"/>
            <geom contype="0" friction="1 0.1 0.1" rgba="0.0 0.7 0 1"/>
            <tendon/>
            <motor ctrlrange="-3 3"/>
        </default>
        <option gravity="0 0 -9.81" integrator="RK4" timestep="0.02"/>
        <size nstack="3000"/>
        <worldbody>
            <light pos="0 0 3.5" dir="0 0 -1" directional="true"/>
            <geom name="rail" pos="0 0 0" quat="0.707 0 0.707 0" rgba="0.3 0.3 0.7 1" size="0.02 1" type="capsule" group="3"/>
            <body name="cart" pos="0 0 0">
                <joint axis="1 0 0" limited="true" name="slider" pos="0 0 0" range="-1 1" type="slide"/>
                <geom name="cart" pos="0 0 0" quat="0.707 0 0.707 0" size="0.1 0.1" type="capsule"/>
                <body name="pole" pos="0 0 0">
                    <joint axis="0 1 0" name="hinge" pos="0 0 0" range="-100000 100000" type="hinge"/>
                    <geom fromto="0 0 0 0.001 0 0.6" name="cpole" rgba="0 0.7 0.7 1" size="0.049 0.3" type="capsule"/>
                </body>
            </body>
        </worldbody>
        <actuator>
            <motor ctrllimited="true" ctrlrange="-3 3" gear="100" joint="slider" name="slide"/>
        </actuator>
    </mujoco>
    """

    def __init__(
        self,
        use_viewer: bool = False,
        max_steps: int = 1000,
    ):
        self.init_qpos = np.zeros(2)
        self.init_qvel = np.zeros(2)
        self.model = mujoco.MjModel.from_xml_string(self.xml_env)
        self.data = mujoco.MjData(self.model)
        self.viewer = mujoco.viewer.launch_passive(
            self.model, self.data) if use_viewer else None
        self.max_steps = max_steps
        self._steps = 0
        self._ball_pos = np.array([0.0, 0.0, 0.6]) #[x,z,y]
        self._last_ball_update_time = 0.0
        self.reset_model()

    def compute_reward(
        self,
        obs: np.ndarray,
        action: np.ndarray,
    ) -> float:
        x, cos_theta, sin_theta, _, theta_dot, tip_x, tip_y, ball_x, ball_y = obs
        upright = (cos_theta + 1) / 2
        centered = tolerance(x, bounds=(ball_x, ball_x), margin=3.5, sigmoid='cosine')
        centered = (1 + centered) / 2
        centered_tip_x = tolerance(tip_x, bounds=(ball_x, ball_x), margin=0.5, sigmoid='cosine')
        centered_tip_x = (1 + centered_tip_x) / 2
        centered_tip_y = tolerance(tip_y, bounds=(ball_y, ball_y), margin=0.5, sigmoid='cosine')
        centered_tip_y = (1 + centered_tip_y) / 2
        small_control = tolerance(action, bounds=(0.0, 0.0), margin=1.0, sigmoid='quadratic', value_at_margin=0.0)
        small_control = (4 + small_control) / 5
        small_theta_dot = tolerance(theta_dot, bounds=(0.0, 0.0), margin=5.0, sigmoid='cosine')
        small_theta_dot = (1 + small_theta_dot) / 2
        reward = (
            upright
            * small_control
            * small_theta_dot
            * centered_tip_x
            * centered_tip_y
            * centered
        )
        return float(reward)

    def step(self, a: np.ndarray) -> Tuple[np.ndarray, float, bool]:
        self.data.ctrl = a
        mujoco.mj_step(self.model, self.data)
        if self.viewer:
            self.viewer.sync()
        self._steps += 1
        if self.current_time - self._last_ball_update_time > 5.0:
            self._ball_pos = [np.random.rand() - 0.5, 0, 0.6]
            self._last_ball_update_time = self.current_time
            if self.viewer : self.draw_ball()
        ob = self.obs()
        reward = self.compute_reward(ob, a)
        terminated = bool(not np.isfinite(ob).all() or self._steps >= self.max_steps)
        return ob, reward, terminated

    def obs(self) -> np.ndarray:
        x_cart = self.data.qpos[0]
        theta = self.data.qpos[1]
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        qvel = self.data.qvel
        x_tip = x_cart + 0.6 * sin_theta
        y_tip = 0.6 * cos_theta
        tip_pos = np.array([x_tip, y_tip])
        ball_pos = self._ball_pos[::2] #only x,y
        return np.concatenate([
            [x_cart],
            [cos_theta],
            [sin_theta],
            qvel,
            tip_pos,
            ball_pos
        ]).ravel()

    def reset_model(self) -> np.ndarray:
        self.data.qpos = self.init_qpos
        self.data.qvel = self.init_qvel
        self.data.qpos[1] = 3.14 #np.pi
        self._steps = 0
        self._ball_pos = np.array([0.0, 0.0, 0.6])
        self._last_ball_update_time = 0.0
        return self.obs()

    def set_dt(self, new_dt: float) -> None:
        self.model.opt.timestep = new_dt

    def draw_ball(
        self,
        position: Union[np.ndarray, None] = None,
        color: list[float] = [1, 0, 0, 1],
        radius: float = 0.05,
    ) -> None:
        mujoco.mjv_initGeom(
            self.viewer.user_scn.geoms[0],
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[radius, 0, 0],
            pos=self._ball_pos,
            mat=np.eye(3).flatten(),
            rgba=np.array(color),
        )
        self.viewer.user_scn.ngeom = 1

    @property
    def current_time(self) -> float:
        return self.data.time

    def close(self) -> None:
        if self.viewer:
            self.viewer.close()
            self.viewer = None