import numpy as np
import numpy.random as npr
import pybullet as pb

from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import (
    DroneModel,
    Physics,
    ActionType,
    ObservationType,
)


class GoToTargetEnv(BaseRLAviary):
    """Go to target position env using gym_pybullet_drones. Subclasses should specify the goals."""

    ################################################################################

    def __init__(
        self,
        drone_model: DroneModel = DroneModel.CF2X,
        initial_xyzs=None,
        initial_rpys=None,
        physics: Physics = Physics.PYB,
        pyb_freq: int = 240,
        ctrl_freq: int = 30,
        gui=False,
        record=False,
    ):
        """Initialization of a single agent RL environment.

        Using the generic single agent RL superclass.

        Parameters
        ----------
        drone_model : DroneModel, optional
            The desired drone type (detailed in an .urdf file in folder `assets`).
        initial_xyzs: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial XYZ position of the drones.
        initial_rpys: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial orientations of the drones (in radians).
        physics : Physics, optional
            The desired implementation of PyBullet physics/custom dynamics.
        pyb_freq : int, optional
            The frequency at which PyBullet steps (a multiple of ctrl_freq).
        ctrl_freq : int, optional
            The frequency at which the environment steps.
        gui : bool, optional
            Whether to use PyBullet's GUI.
        record : bool, optional
            Whether to save a video of the simulation.
        obs : ObservationType, optional
            The type of observation space (kinematic information or vision)
        act : ActionType, optional
            The type of action space (1 or 3D; RPMS, thurst and torques, or waypoint with PID control)

        """
        self.workspace = np.array([[-1.5, -1.5, 0], [1.5, 1.5, 1.5]])
        self.EPISODE_LEN_SEC = 8
        self.SUCCESS_TH = 1e-1

        super().__init__(
            drone_model=drone_model,
            num_drones=1,
            initial_xyzs=initial_xyzs,
            initial_rpys=initial_rpys,
            physics=physics,
            pyb_freq=pyb_freq,
            ctrl_freq=ctrl_freq,
            gui=gui,
            record=record,
            obs=ObservationType.KIN,
            act=ActionType.PID,
        )

    def reset(self, seed: int=None, options: dict = None):
        self.TARGET_POS = npr.uniform(self.workspace[0], self.workspace[1])
        obs = super().reset(seed, options)

        target_idx = pb.createVisualShape(pb.GEOM_SPHERE, radius=5e-2, rgbaColor=[1,0,0,1])
        pb.createMultiBody(baseVisualShapeIndex=target_idx, basePosition=self.TARGET_POS)

        return obs

    ################################################################################
    def step(self, delta_act):
        state = self._getDroneStateVector(0)
        pos = state[:3]
        act = (pos + delta_act).reshape(1, -1)
        return super().step(act)

    def _computeReward(self):
        """Computes the current reward value.

        Returns
        -------
        float
            The reward.

        """
        state = self._getDroneStateVector(0)
        ret = float(np.linalg.norm(self.TARGET_POS - state[0:3]) < self.SUCCESS_TH)

        return ret

    ################################################################################

    def _computeTerminated(self):
        """Computes the current done value.

        Returns
        -------
        bool
            Whether the current episode is done.

        """
        state = self._getDroneStateVector(0)
        if np.linalg.norm(self.TARGET_POS - state[0:3]) < self.SUCCESS_TH:
            return True
        else:
            return False

    ################################################################################

    def _computeTruncated(self):
        """Computes the current truncated value.

        Returns
        -------
        bool
            Whether the current episode timed out.

        """
        state = self._getDroneStateVector(0)
        if (
            abs(state[0]) > 1.5
            or abs(state[1]) > 1.5
            or state[2] > 2.0  # Truncate when the drone is too far away
            or abs(state[7]) > 0.4
            or abs(state[8]) > 0.4  # Truncate when the drone is too tilted
        ):
            return True
        if self.step_counter / self.PYB_FREQ > self.EPISODE_LEN_SEC:
            return True
        else:
            return False

    def _computeObs(self):
        """Returns the current observation of the environment.

        Returns
        -------
        ndarray
            A Box() of shape (NUM_DRONES,H,W,4) or (NUM_DRONES,12) depending on the observation type.

        """
        drone_obs_full = self._getDroneStateVector(0)
        drone_obs_pos = drone_obs_full[:3]
        obs = np.concatenate([drone_obs_pos, self.TARGET_POS], axis=-1)

        return obs

    ################################################################################

    def _computeInfo(self):
        """Computes the current info dict(s).

        Unused.

        Returns
        -------
        dict[str, int]
            Dummy value.

        """
        return {
            "answer": 42
        }  #### Calculated by the Deep Thought supercomputer in 7.5M years
