from typing import List, Callable, Tuple, NoReturn, Any

from gym import Env
from gym.spaces import Discrete
import random
from agents.SwitchboardAgent import SwitchboardAgent, get_switchboard_causal_graph, get_wrong_switchboard_causal_graph
import copy


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
        self.observation_space = Discrete(5*2+5*(5-1)/2)
        self.latest_evaluation = self.agent.evaluate_causal_model()

    def reset(self) -> NoReturn:
        pass

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
        intervention_one_hot = [1 if self.current_action[1] == i else 0 for i in range(len(self.lights))]
        graph_state = self.agent.get_graph_state()
        state = [int(l) for l in self.lights]  # convert bool to int
        state.extend(intervention_one_hot)
        state.extend(graph_state)

        # let the episode end when the causal model is altered
        if self.current_action[0] == 1:
            done = True
        else:
            done = False

        return state, 0.0, done, {}

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
            print(out + str(self.current_action))
