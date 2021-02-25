from gym import Env
from gym.spaces import Discrete
import random


class Switchboard(Env):
    def __init__(self):
        super(Switchboard, self).__init__()
        self.observation_space = Discrete(5)
        self.action_space = Discrete(11)  # 5 variables, each on or off + do nothing
        self.lights = [False]*5  # all lights are off
        self.U = random.choices([False, True], k=5)  # random context

        self.SCM = [lambda: self.lights[4] or self.U[0],
                    lambda: self.lights[0] or self.lights[2] or self.U[1],
                    lambda: self.lights[0] or self.lights[4] or self.U[2],
                    lambda: self.lights[2] or self.U[3],
                    lambda: self.U[4]]

    def reset(self):
        # get a random instatiation of the light
        self.U = random.choices([False, True], k=5)

    def step(self, action):
        # apply functions of SCM until no changes are done anymore
        while True:
            old_lights = self.lights
            self.lights = [self.SCM[i]() for i in range(5)]
            if old_lights == self.lights:
                break

        self.reset()

    def render(self, mode='human'):
        if mode=='human':
            print(self.lights)