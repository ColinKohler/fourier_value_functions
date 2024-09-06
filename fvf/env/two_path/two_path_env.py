import os

import gymnasium as gym
from gymnasium import spaces

import collections
import numpy as np
import numpy.random as npr
import pygame
import pymunk
import pymunk.pygame_util
from pymunk.vec2d import Vec2d
import shapely.geometry as sg
import cv2
import skimage.transform as st
from fvf.env.pusht.pymunk_override import DrawOptions


def pymunk_to_shapely(body, shapes):
    geoms = list()
    for shape in shapes:
        if isinstance(shape, pymunk.shapes.Poly):
            verts = [body.local_to_world(v) for v in shape.get_vertices()]
            verts += [verts[0]]
            geoms.append(sg.Polygon(verts))
        else:
            raise RuntimeError(f"Unsupported shape type {type(shape)}")
    geom = sg.MultiPolygon(geoms)
    return geom


class TwoPathEnv(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 10}
    reward_range = (0.0, 1.0)

    def __init__(
        self,
        legacy=False,
        damping=None,
        render_action=True,
        render_size=96,
        reset_to_state=None,
        render_mode="rgb_array",
    ):
        self.render_mode = render_mode
        self._seed = None
        self.seed()
        self.window_size = ws = 512  # The size of the PyGame window
        self.render_size = render_size
        self.sim_hz = 100
        # Local controller params.
        self.k_p, self.k_v = 10, 2  # PD control.z
        self.control_hz = self.metadata["video.frames_per_second"]
        # legcay set_state for data compatibility
        self.legacy = legacy

        # agent_pos, block_pos, block_angle
        ws = self.window_size
        self.observation_space = spaces.Dict(
            {
                "image": spaces.Box(
                    low=0, high=1, shape=(3, render_size, render_size), dtype=np.float32
                ),
                "agent_pos": spaces.Box(low=0, high=ws, shape=(2,), dtype=np.float32),
            }
        )
        self.render_cache = None

        # positional goal for agent
        self.action_space = spaces.Box(
            low=np.array([0, 0], dtype=np.float64),
            high=np.array([ws, ws], dtype=np.float64),
            shape=(2,),
            dtype=np.float64,
        )

        self.damping = damping
        self.render_action = render_action

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None
        self.screen = None

        self.space = None
        self.teleop = None
        self.render_buffer = None
        self.latest_action = None
        self.reset_to_state = reset_to_state

    def reset(self, seed, options):
        seed = self._seed
        self._setup()
        if self.damping is not None:
            self.space.damping = self.damping

        # use legacy RandomState for compatibility
        state = self.reset_to_state
        if state is None:
            rs = np.random.RandomState(seed=seed)
            state = np.array(
                [
                    self.window_size // 2,
                    self.window_size // 2,
                    rs.randint(self.window_size // 2 - 1, self.window_size // 2 + 1),
                    rs.randint(140, 160),
                    0,
                ]
            )
        self._set_state(state)

        observation = self._get_obs()
        return observation

    def step(self, action):
        dt = 1.0 / self.sim_hz
        self.n_contact_points = 0
        n_steps = self.sim_hz // self.control_hz
        if action is not None:
            self.latest_action = action
            for i in range(n_steps):
                # Step PD control.
                self.agent.velocity = self.k_p * Vec2d(*action)

                # acceleration = self.k_p * (action - self.agent.position) + self.k_v * (Vec2d(0, 0) - self.agent.velocity)
                # self.agent.velocity += acceleration * dt

                # Step physics.
                self.space.step(dt)

        # compute reward
        goal_body = self._get_goal_pose_body(self.goal_pose)
        # goal_geom = pymunk_to_shapely(goal_body, self.block.shapes)
        # block_geom = pymunk_to_shapely(self.block, self.block.shapes)

        # block_geom = block_geom.buffer(0)
        # goal_geom = goal_geom.buffer(0)
        # if goal_geom.intersects(block_geom):
        #    intersection_area = goal_geom.intersection(block_geom).area
        #    goal_area = goal_geom.area
        #    coverage = intersection_area / goal_area
        # else:
        #    coverage = 0
        # reward = np.clip(coverage / self.success_threshold, 0, 1)
        # done = coverage > self.success_threshold
        reward = 0
        done = False

        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, done, False, info

    def render(self):
        return self._render_frame()

    def render2(self):
        return self._render_frame()

    def teleop_agent(self):
        TeleopAgent = collections.namedtuple("TeleopAgent", ["act"])

        def act(obs):
            act = None
            mouse_position = pymunk.pygame_util.from_pygame(
                Vec2d(*pygame.mouse.get_pos()), self.screen
            )
            if self.teleop or (mouse_position - self.agent.position).length < 5:
                self.teleop = True
                act = (mouse_position - self.agent.position).x, (
                    mouse_position - self.agent.position
                ).y

            if act is not None:
                act = np.clip(act, -10, 10)
                if np.all(np.abs(act) < 1):
                    act = None

            return act

        return TeleopAgent(act)

    def _get_obs(self):
        img = self._render_frame()

        agent_pos = np.array(self.agent.position)
        img_obs = np.moveaxis(img.astype(np.float32) / 255, -1, 0)
        obs = {"image": img_obs, "agent_pos": agent_pos}

        # draw action
        if self.latest_action is not None:
            action = np.array(self.latest_action)
            coord = (action / 512 * 96).astype(np.int32)
            marker_size = int(8 / 96 * self.render_size)
            thickness = int(1 / 96 * self.render_size)
            cv2.drawMarker(
                img,
                coord,
                color=(255, 0, 0),
                markerType=cv2.MARKER_CROSS,
                markerSize=marker_size,
                thickness=thickness,
            )
        self.render_cache = img

        return obs

    def _get_goal_pose_body(self, pose):
        mass = 1
        inertia = pymunk.moment_for_box(mass, (50, 100))
        body = pymunk.Body(mass, inertia)
        body.position = pose[:2].tolist()
        return body

    def _get_info(self):
        n_steps = self.sim_hz // self.control_hz
        n_contact_points_per_step = int(np.ceil(self.n_contact_points / n_steps))
        info = {
            "pos_agent": np.array(self.agent.position),
            "vel_agent": np.array(self.agent.velocity),
            "wall_pose": np.array(list(self.wall.position) + [self.wall.angle]),
            "goal_pose": self.goal_pose,
            "n_contacts": n_contact_points_per_step,
        }
        return info

    def _render_frame(self):

        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        self.screen = canvas

        draw_options = DrawOptions(canvas)

        # Draw goal pose.
        goal_body = self._get_goal_pose_body(self.goal_pose)
        for shape in self.wall.shapes:
            goal_points = [
                pymunk.pygame_util.to_pygame(
                    goal_body.local_to_world(v), draw_options.surface
                )
                for v in shape.get_vertices()
            ]
            goal_points += [goal_points[0]]
            pygame.draw.polygon(canvas, self.goal_color, goal_points)

        # Draw agent and block.
        self.space.debug_draw(draw_options)

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # the clock is already ticked during in step for "human"

        img = np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))
        img = cv2.resize(img, (self.render_size, self.render_size))
        if self.render_action:
            if self.render_action and (self.latest_action is not None):
                action = np.array(self.latest_action)
                coord = (action / 512 * 96).astype(np.int32)
                marker_size = int(8 / 96 * self.render_size)
                thickness = int(1 / 96 * self.render_size)
                cv2.drawMarker(
                    img,
                    coord,
                    color=(255, 0, 0),
                    markerType=cv2.MARKER_CROSS,
                    markerSize=marker_size,
                    thickness=thickness,
                )
        return img

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

    def seed(self, seed=None):
        if seed is None:
            seed = np.random.randint(0, 25536)
        self._seed = seed
        self.np_random = np.random.default_rng(seed)

    def _handle_collision(self, arbiter, space, data):
        self.n_contact_points += len(arbiter.contact_point_set.points)

    def _set_state(self, state):
        if isinstance(state, np.ndarray):
            state = state.tolist()
        pos_agent = state[:2]
        pos_block = state[2:4]
        rot_block = state[4]
        self.agent.position = pos_agent
        # setting angle rotates with respect to center of mass
        # therefore will modify the geometric position
        # if not the same as CoM
        # therefore should be modified first.
        if self.legacy:
            # for compatibility with legacy data
            self.wall.position = pos_block
            self.wall.angle = rot_block
        else:
            self.wall.angle = rot_block
            self.wall.position = pos_block

        # Run physics to take effect
        self.space.step(1.0 / self.sim_hz)

    def _setup(self):
        self.space = pymunk.Space()
        self.space.gravity = 0, 0
        self.space.damping = 0
        self.teleop = False
        self.render_buffer = list()

        # Add walls.
        walls = [
            self._add_segment((5, 506), (5, 5), 2),
            self._add_segment((5, 5), (506, 5), 2),
            self._add_segment((506, 5), (506, 506), 2),
            self._add_segment((5, 506), (506, 506), 2),
        ]
        self.space.add(*walls)

        # Add agent, block, and goal zone.
        self.agent = self.add_circle((256, 100), 15)
        self.wall = self.add_box((256, 256), 200, 50)
        self.goal_color = pygame.Color("LightGreen")
        self.goal_pose = np.array([256, 300])  # x, y

        # Add collision handling
        self.collision_handeler = self.space.add_collision_handler(0, 0)
        self.collision_handeler.post_solve = self._handle_collision
        self.n_contact_points = 0

        self.max_score = 50 * 100
        self.success_threshold = 0.90  # 95% coverage.

    def _add_segment(self, a, b, radius):
        shape = pymunk.Segment(self.space.static_body, a, b, radius)
        shape.color = pygame.Color(
            "LightGray"
        )  # https://htmlcolorcodes.com/color-names
        return shape

    def add_circle(self, position, radius):
        body = pymunk.Body(10, float("inf"))
        body.position = position
        body.friction = 1
        shape = pymunk.Circle(body, radius)
        shape.color = pygame.Color("RoyalBlue")
        self.space.add(body, shape)
        return body

    def add_box(self, position, height, width):
        mass = 10000
        inertia = pymunk.moment_for_box(mass, (height, width))
        body = pymunk.Body(mass, inertia)
        body.position = position
        shape = pymunk.Poly.create_box(body, (height, width))
        shape.color = pygame.Color("LightSlateGray")
        self.space.add(body, shape)
        return body
