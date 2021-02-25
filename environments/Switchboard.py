from gym import Env
from gym.spaces import Discrete
import random
from agents.SwitchboardAgent import SwitchboardAgent
import copy


class Switchboard(Env):
    def __init__(self):
        super(Switchboard, self).__init__()
        self.observation_space = Discrete(5)
        self.lights = [False]*5  # all lights are off
        self.U = random.choices([False, True], k=5)  # random context

        self.SCM = [lambda: self.lights[4] or self.U[0],
                    lambda: self.lights[0] or self.lights[2] or self.lights[4] or self.U[1],
                    lambda: self.lights[4] or self.U[2],
                    lambda: self.lights[2] or self.U[3],
                    lambda: self.U[4]]

        self.agent = SwitchboardAgent(len(self.lights))
        self.action_space = Discrete(len(self.agent.actions))

    def reset(self):
        pass

    def step(self, action):
        # get a random instatiation of the light. Simulation the unobserved (natural) change of the environment
        self.U = random.choices([False, True], k=5)

        # apply intervention to the SCM
        interv_scm = copy.deepcopy(self.SCM)
        if action != (None, None):
            interv_scm[action[0]] = lambda: action[1]

        # apply functions of SCM until no changes are done anymore
        while True:
            old_lights = self.lights
            self.lights = [interv_scm[i]() for i in range(5)]
            if old_lights == self.lights:
                break

        self.reset()

    def render(self, mode='human'):
        if mode=='human':
            print(self.lights)