from typing import List, Callable, Tuple, NoReturn

from gym import Env
from gym.spaces import Discrete
import random
from agents.SwitchboardAgent import SwitchboardAgent
import copy


class Switchboard(Env):
    Agent: SwitchboardAgent
    Function = Callable[[], bool]
    SCM: List[Function]
    U: List[bool]
    Lights: List[bool]

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

    def reset(self) -> NoReturn:
        pass

    def step(self, action: int) -> Tuple[List[bool], float, bool, dict]:
        self.agent.current_action = self.agent.actions[action]

        # get a random instantiation of the light. Simulation the unobserved (natural) change of the environment
        self.U = random.choices([False, True], k=5)

        # apply intervention to the SCM
        interv_scm = copy.deepcopy(self.SCM)
        if self.agent.current_action != (None, None):
            interv_scm[self.agent.current_action[0]] = lambda: self.agent.current_action[1]

        # apply functions of SCM until no changes are done anymore
        while True:
            old_lights = self.lights
            self.lights = [interv_scm[i]() for i in range(5)]
            if old_lights == self.lights:
                break
        self.agent.store_observation(self.lights)

        self.reset()

    def render(self, mode: str = 'human') -> NoReturn:
        if mode == 'human':
            out = ''
            for i in range(len(self.lights)):
                if self.lights[i]:
                    out += '|'
                else:
                    out += 'O'
                if self.agent.current_action[0] == i:
                    out += '*'
                out += '\t'
            print(out)
