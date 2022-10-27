import numpy as np
from matplotlib import patches
from rps.robotarium import Robotarium
from rps.utilities import barrier_certificates


class Simulator(Robotarium):
    """ HITSZ ML Lab simulator """

    def __init__(self, number_of_robots=1, *args, **kwd):
        super(Simulator, self).__init__(number_of_robots=number_of_robots, *args, **kwd)
        self.init_environment()
        self.terminate = 0

    def render(self, show=0):
        if show:
            self.show_figure = True

    def close(self):
        self.terminate = 1

    def init_environment(self):
        # reset boundaries
        self.boundaries = [-3.1, -3.1, 6.2, 6.2]
        if self.show_figure:
            self.boundary_patch.remove()
        padding = 1

        if self.show_figure:
            # NOTE: boundaries = [x_min, y_min, x_max - x_min, y_max - y_min] ?
            self.axes.set_xlim(self.boundaries[0] - padding, self.boundaries[0] + self.boundaries[2] + padding)
            self.axes.set_ylim(self.boundaries[1] - padding, self.boundaries[1] + self.boundaries[3] + padding)

            patch = patches.Rectangle(self.boundaries[:2], *self.boundaries[2:4], fill=False, linewidth=2)
            self.boundary_patch = self.axes.add_patch(patch)

        # set barries
        self.barrier_centers = [(1, 1), (-1, 1), (0, -1)]
        self.radius = 0.1
        if self.show_figure:
            self.barrier_patches = [
                patches.Circle(self.barrier_centers[0], radius=self.radius),
                patches.Circle(self.barrier_centers[1], radius=self.radius),
                patches.Circle(self.barrier_centers[2], radius=self.radius),
                # patches.Rectangle((-0.25, 2.45), 0.55, 0.55),
                # patches.Rectangle((-0.25, -2.55), 0.55, 0.55)
            ]

            for patch in self.barrier_patches:
                patch.set(fill=True, color="#000")
                self.axes.add_patch(patch)

            # TODO: barries certs
            self.barrier_certs = [
            ]

            # set goals areas
            self.goal_patches = [
                # patches.Circle((4, 4), radius=0.24),
                # patches.Circle((-4, 4), radius=0.24),
                # patches.Circle((4, -4), radius=0.24),
                patches.Circle((-2.5, -2.5), radius=0.2),
            ]

            for patch in self.goal_patches:
                patch.set(fill=False, color='#5af')
                self.axes.add_patch(patch)

    def set_velocities(self, velocities):
        """
        velocites is a (N, 2) np.array contains (ω, v) of agents
        """
        self._velocities = velocities

    def step(self, action):
        # compute hunter's action
        """
        get robot pose 3x2
        first column is prey
        second column is hunter
        """
        poses = self.get_poses()
        # print('poses before', poses)
        # get hunter's velocity
        dxu_hunter = self.hunter_policy(poses[:, 1].reshape(-1, 1), poses[:2, 0].reshape(-1, 1))
        dxu = np.concatenate([action.reshape(-1, 1), dxu_hunter], axis=1)
        terminate = 0
        reward = 0
        # make a step
        self.set_velocities(dxu)
        self._step()
        # print('poses after', poses)
        # collision detect
        for robot in range(2):
            # collision with boundaries
            padding = 0.1
            self.poses[0, robot] = self.poses[0, robot] if self.poses[0, robot] > self.boundaries[0] + padding else \
                self.boundaries[0] + padding
            self.poses[0, robot] = self.poses[0, robot] if self.poses[0, robot] < self.boundaries[0] + self.boundaries[
                2] - padding else self.boundaries[0] + self.boundaries[2] - padding
            self.poses[1, robot] = self.poses[1, robot] if self.poses[1, robot] > self.boundaries[1] + padding else \
                self.boundaries[1] + padding
            self.poses[1, robot] = self.poses[1, robot] if self.poses[1, robot] < self.boundaries[0] + self.boundaries[
                3] - padding else self.boundaries[1] + self.boundaries[3] - padding

            # collision with barriers
            for barrier in self.barrier_centers:
                tempA = self.poses[:2, robot] - np.array(barrier)
                dist = np.linalg.norm(tempA)

                if dist < self.radius + padding:
                    tempA = tempA / dist * (self.radius + padding)
                    self.poses[:2, robot] = tempA + np.array(barrier)
                    if robot == 0:
                        reward -= 5

        # collision with prey
        tempB = self.poses[:2, 1] - self.poses[:2, 0]
        dist_temp = np.linalg.norm(tempB)
        if dist_temp < self.radius:
            tempB = tempB / dist_temp * (self.radius)
            self.poses[:2, 1] = tempB + np.array(self.poses[:2, 0])
            # self.terminate = 1
            reward -= 10

        # whether reach goal area
        tempC = self.poses[:2, 0] - np.array([-2.5, -2.5])
        dist_C = np.linalg.norm(tempC)
        # print(dist_C)
        if dist_C < 0.2:
            self.terminate = 1
            reward += 500
        elif dist_C < 1:
            reward += 25
        elif dist_C < 2:
            reward += 20
        elif dist_C < 3:
            reward += 15
        elif dist_C < 4:
            reward += 10
        elif dist_C < 5:
            reward += 5
        else:
            pass

        # compute the reward
        reward = reward + self.get_reward(poses[:, 0], action) + 10.0 / dist_C
        state = np.append(self.poses[:, 0], self.poses[:, 1])
        # state = np.append(state, dist_C)
        info = None
        return state, reward, self.terminate, info

    def _step(self, *args, **kwd):
        dxu = self._velocities
        # print('_step_dxu', dxu)
        """
        the first column is the velocity of prey
        the second column is the velocity of hunter
        """
        if self.show_figure:
            for cert in self.barrier_certs:
                dxu = cert(dxu, poses)

        super(Simulator, self).set_velocities(range(self.number_of_robots), dxu)
        super(Simulator, self).step(*args, **kwd)
        # print("self.poses", self.poses)

    def evaluate(self, action):
        poses = self.get_poses()
        # print('poses before', poses)
        # get hunter's velocity
        dxu_hunter = self.hunter_policy(poses[:, 1].reshape(-1, 1), poses[:2, 0].reshape(-1, 1))
        velocities = np.concatenate([action.reshape(-1, 1), dxu_hunter], axis=1)
        velocities[0, 0] = velocities[0, 0] if np.abs(velocities[0, 0]) < self.max_linear_velocity else \
            self.max_linear_velocity * np.sign(velocities[0, 0])
        velocities[1, 0] = velocities[1, 0] if np.abs(velocities[1, 0]) < self.max_angular_velocity else \
            self.max_angular_velocity * np.sign(velocities[1, 0])
        velocities[0, 1] = velocities[0, 1] if np.abs(velocities[0, 1]) < self.max_linear_velocity_hunter else \
            self.max_linear_velocity_hunter * np.sign(velocities[0, 1])
        velocities[1, 1] = velocities[1, 1] if np.abs(velocities[1, 1]) < self.max_angular_velocity_hunter else \
            self.max_angular_velocity_hunter * np.sign(velocities[1, 1])
        # Update dynamics of agents
        poses[0, :] = poses[0, :] + self.time_step * np.cos(poses[2, :]) * velocities[0, :]
        poses[1, :] = poses[1, :] + self.time_step * np.sin(poses[2, :]) * velocities[0, :]
        poses[2, :] = poses[2, :] + self.time_step * velocities[1, :]
        # Ensure angles are wrapped
        poses[2, :] = np.arctan2(np.sin(poses[2, :]), np.cos(poses[2, :]))
        # collision detect
        for robot in range(2):
            # collision with boundaries
            padding = 0.1
            poses[0, robot] = poses[0, robot] if poses[0, robot] > self.boundaries[0] + padding else \
                self.boundaries[0] + padding
            poses[0, robot] = poses[0, robot] if poses[0, robot] < self.boundaries[0] + self.boundaries[
                2] - padding else self.boundaries[0] + self.boundaries[2] - padding
            poses[1, robot] = poses[1, robot] if poses[1, robot] > self.boundaries[1] + padding else \
                self.boundaries[1] + padding
            poses[1, robot] = poses[1, robot] if poses[1, robot] < self.boundaries[0] + self.boundaries[
                3] - padding else self.boundaries[1] + self.boundaries[3] - padding

            # collision with barriers
            for barrier in self.barrier_centers:
                tempA = poses[:2, robot] - np.array(barrier)
                dist = np.linalg.norm(tempA)

                if dist < self.radius + padding:
                    tempA = tempA / dist * (self.radius + padding)
                    poses[:2, robot] = tempA + np.array(barrier)

        # collision with prey
        tempB = poses[:2, 1] - poses[:2, 0]
        dist_temp = np.linalg.norm(tempB)
        if dist_temp < self.radius:
            tempB = tempB / dist_temp * (self.radius)
            poses[:2, 1] = tempB + np.array(poses[:2, 0])

        # whether reach goal area
        tempC = poses[:2, 0] - np.array([-2.5, -2.5])
        dist_C = np.linalg.norm(tempC)
        if dist_C < 0.2:
            terminate = 1

        # compute the reward
        reward = self.get_reward(poses[:, 0], action)
        return reward

    def reset(self, initial_conditions=np.array([[2], [2], [0]])):
        assert initial_conditions.shape[1] > 0, "the initial conditions must not be empty"
        assert initial_conditions.shape[1] < 3, "More than 2 robot's initial conditions receive"
        if initial_conditions.shape[1] == 1:
            self.poses = np.concatenate([initial_conditions.reshape(-1, 1), np.zeros((3, 1), dtype=float)], axis=1)
        elif initial_conditions.shape[1] == 2:
            self.poses = initial_conditions
        # temp = np.array([2, 2]) - np.array([-2.5, -2.5])
        # dist = np.linalg.norm(temp)
        state = np.append(self.poses[:, 0], self.poses[:, 1])
        self.terminate = 0
        return state

    def hunter_policy(self, hunter_states, prey_positions):
        _, N = np.shape(hunter_states)
        dxu = np.zeros((2, N))
        # print('states', hunter_states)
        # print('positions', hunter_states)
        pos_error = prey_positions - hunter_states[:2][:]
        rot_error = np.arctan2(pos_error[1][:], pos_error[0][:])
        dist = np.linalg.norm(pos_error, axis=0)

        dxu[0][:] = 0.8 * (dist + 0.2) * np.cos(rot_error - hunter_states[2][:])
        dxu[1][:] = 3 * dist * np.sin(rot_error - hunter_states[2][:])

        return dxu

    ############### Add Your Code Here ##############################
    def get_reward(self, prey_state, action):
        # add you own reward function here
        hunter_state = self.poses[:, 1]
        reward = np.linalg.norm(hunter_state[:2] - prey_state[:2])
        return reward
