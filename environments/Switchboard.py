from typing import List, Callable, Tuple, NoReturn, Any

from gym import Env
from gym.spaces import Discrete, Box
import random
from agents.SwitchboardAgent import SwitchboardAgent, get_switchboard_causal_graph, get_wrong_switchboard_causal_graph
import copy
import numpy as np


class Switchboard(Env):
    Agent: SwitchboardAgent
    Function = Callable[[], bool]
    SCM: List[Function]
    U: List[bool]
    Lights: List[bool]

    def __init__(self):
        super(Switchboard, self).__init__()
        self.lights = [False]*5  # all lights are off
        self.U = random.choices([False, True], k=5)  # random context

        self.SCM = [lambda: self.lights[4] or self.U[0],
                    lambda: self.lights[0] or self.lights[2] or self.lights[4] or self.U[1],
                    lambda: self.lights[4] or self.U[2],
                    lambda: self.lights[2] or self.U[3],
                    lambda: self.U[4]]

        self.agent = SwitchboardAgent(len(self.lights), get_wrong_switchboard_causal_graph())
        self.action_space = Discrete(len(self.agent.actions))
        self.observation_space = Box(0, 1, (int(5*2+5*(5-1)/2),))
        self.latest_evaluation = self.agent.evaluate_causal_model()
        self.current_action = (None, None, None)

    def reset(self) -> List[float]:
        return self.get_obs_vector()

    def step(self, action: int) -> Tuple[List[Any], float, bool, dict]:
        self.current_action = self.agent.actions[action]

        # get a random instantiation of the light. Simulation the unobserved (natural) change of the environment
        self.U = random.choices([False, True], k=5)
        interv_scm = copy.deepcopy(self.SCM)

        # apply action
        if self.current_action[0] == 0:  # intervention action
            interv_scm[self.current_action[1]] = lambda: self.current_action[2]
            action_successful = True
        elif self.current_action[0] == 1:
            action_successful = self.agent.update_model(self.current_action)
        elif self.current_action[0] == None:
            action_successful = True

        # apply functions of SCM until no changes are done anymore
        while True:
            old_lights = self.lights
            self.lights = [interv_scm[i]() for i in range(5)]
            if old_lights == self.lights:
                break
        self.agent.store_observation(self.lights, self.current_action)

        # determine state after action
        state = self.get_obs_vector()

        # let the episode end when the causal model is altered and evaluate new graph
        if self.current_action[0] == 1:
            new_eval = self.agent.evaluate_causal_model()
            # graph_improved = new_eval >= self.latest_evaluation
            self.latest_evaluation = new_eval
            done = True
        else:
            done = False

        # compute reward
        if not action_successful:
            reward = -2
        elif self.latest_evaluation == -2 and self.current_action[0] == 1:  # not enough data has been collected. Need intervention
            reward = -2
        else:
            reward = self.latest_evaluation
        # elif self.current_action[0] == 0 or self.current_action[0] == None:  # intervention
        #     reward = 0
        # elif graph_improved:
        #     reward = 1
        # elif not graph_improved:
        #     reward = -1

        print(self.current_action, '\treward', reward)

        # # show changed network
        # if action_successful and self.current_action[0] == 1:
        #     self.agent.display_causal_model()

        return state, reward, done, {}

    def get_obs_vector(self) -> List[float]:
        intervention_one_hot = [1.0 if self.current_action[1] == i else 0.0 for i in range(len(self.lights))]
        graph_state = self.agent.get_graph_state()
        state = [float(l) for l in self.lights]  # convert bool to int
        state.extend(intervention_one_hot)
        state.extend(graph_state)
        return np.array(state)

    def render(self, mode: str = 'human') -> NoReturn:
        if mode == 'human':
            out = ''
            for i in range(len(self.lights)):
                if self.lights[i]:
                    out += '|'
                else:
                    out += 'O'
                if self.current_action[1] == i:
                    out += '*'
                out += '\t'
            print(out)
