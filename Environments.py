from typing import List, Callable, Tuple, NoReturn, Any

from gym import Env
from gym.spaces import Discrete, Box
import random
from Agents import SwitchboardAgentDQN, get_switchboard_causal_graph, get_wrong_switchboard_causal_graph
import copy
import numpy as np


class Switchboard(Env):
    Agent: SwitchboardAgentDQN
    Function = Callable[[], bool]
    U: List[bool]
    Lights: List[bool]

    def __init__(self):
        super(Switchboard, self).__init__()
        # initialize causal model
        self.SCM = StructuralCausalModel()
        self.SCM.add_exogenous_vars([('U' + str(i), True, random.choice, {'seq': [True, False]}) for i in range(5)])
        self.SCM.add_endogenous_vars(
            [
                ('X0', False, lambda x4, u0: x4 or u0, {'x4': 'X4', 'u0': 'U0'}),
                ('X1', False, lambda x0, x2, x4, u1: x0 or x2 or x4 or u1, {'x0': 'X0', 'x2': 'X2', 'x4': 'X4', 'u1': 'U1'}),
                ('X2', False, lambda x4, u2: x4 or u2, {'x4': 'X4', 'u2': 'U2'}),
                ('X3', False, lambda x2, u3: x2 or u3, {'x2': 'X2', 'u3': 'U3'}),
                ('X4', False, lambda u4: u4, {'u4': 'U4'})
            ])

        self.lights = [False]*5  # all lights are off

        self.agent = SwitchboardAgentDQN(len(self.lights), get_switchboard_causal_graph())
        self.action_space = Discrete(len(self.agent.actions))
        self.observation_space = Box(0, 1, (int((5*2)+5*(5-1)/2),))
        self.latest_evaluation = self.agent.evaluate_causal_model()
        self.current_action = (None, None, None)
        self.rewards = []

    def reset(self) -> np.ndarray:
        self.agent.random_reset_causal_model()
        return self.get_obs_vector()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        self.current_action = self.agent.actions[action]

        interv_scm = copy.deepcopy(self.SCM)
        # apply action
        action_successful = False
        if self.current_action[0] == 0:  # intervention action
            interv_scm.do_interventions([('X'+str(self.current_action[1]), lambda: self.current_action[2])])
            action_successful = True
        elif self.current_action[0] == 1:
            action_successful = self.agent.update_model_per_action(self.current_action)
        elif self.current_action[0] == None:
            action_successful = True

        # determine the states of the lights according to the causal structure
        self.lights = interv_scm.get_next_instantiation()[0]

        self.agent.store_observation_per_action(self.lights, self.current_action)

        # determine state after action
        state = self.get_obs_vector()

        # let the episode end when the causal model is fully learned (loss reaching threshold of -0.006)
        if self.current_action[0] == 1:  # only check if the model actually changed.
            done = self.agent.graph_is_learned()
        else:
            done = False

        # compute reward
        if not action_successful:  # illegal action was taken
            reward = -10
        elif done:  # the graph has been learned
            reward = 5
            self.agent.display_causal_model()
            self.reset()
        else:  # intervention, non-intervention, graph-changing
            reward = 0
        # elif self.current_action[0] == 0 or self.current_action[0] == None:  # intervention
        #     reward = 0
        # elif graph_improved:
        #     reward = 1
        # elif not graph_improved:
        #     reward = -1
        self.rewards.append(reward)
        print(self.current_action, '\treward', reward)

        return state, reward, done, {}

    def get_obs_vector(self) -> np.ndarray:
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


class StructuralCausalModel:
    def __init__(self):
        self.endogenous_vars = {}
        self.exogenous_vars = {}
        self.functions = {}
        self.exogenous_distributions = {}

    def add_endogenous_var(self, name: str, value: Any, function: Callable, param_varnames: dict):
        # ensure unique names
        assert name not in self.exogenous_vars.keys(), 'Variable already exists'
        assert name not in self.endogenous_vars.keys(), 'Variable already exists in endogenous vars'

        self.endogenous_vars[name] = value
        self.functions[name] = (function, param_varnames)

    def add_endogenous_vars(self, vars: List[Tuple[str, Any, Callable, dict]]):
        for v in vars:
            self.add_endogenous_var(v[0], v[1], v[2], v[3])

    def add_exogenous_var(self, name: str, value: Any, distribution: Callable, distribution_kwargs: dict):
        # ensure unique names
        assert name not in self.exogenous_vars.keys(), 'Variable already exists'
        assert name not in self.endogenous_vars.keys(), 'Variable already exists in endogenous vars'

        self.exogenous_vars[name] = value
        self.exogenous_distributions[name] = (distribution, distribution_kwargs)

    def add_exogenous_vars(self, vars: List[Tuple[str, Any, Callable, dict]]):
        for v in vars:
            self.add_exogenous_var(v[0], v[1], v[2], v[3])

    def remove_var(self, name: str):
        if name in self.endogenous_vars.keys():
            assert name in self.endogenous_vars, 'Variable not in list of endogenous vars'

            del self.endogenous_vars[name]
            del self.functions[name]

        else:
            assert name in self.exogenous_vars, 'Variable not in list of exogenous vars'

            del self.exogenous_vars[name]
            del self.exogenous_distributions[name]

    def get_next_instantiation(self) -> Tuple[List, List]:
        """
        Returns a new instantiation of variables consistent with the causal structure and for a sample from the
        exogenous distribution
        :return: Instantiation of endogenous and exogenous variables
        """
        random.seed()
        # update exogenous vars
        for key in self.exogenous_vars:
            dist = self.exogenous_distributions[key]
            self.exogenous_vars[key] = dist[0](**dist[1])

        # update endogenous vars until converge
        while True:
            old_obs = copy.copy(self.endogenous_vars)

            for key in old_obs:
                # get the values for the parameters needed in the functions
                params = {}
                for n in self.functions[key][1]:  # parameters of functions
                    if self.functions[key][1][n] in self.endogenous_vars.keys():
                        params[n] = self.endogenous_vars[self.functions[key][1][n]]
                    else:
                        params[n] = self.exogenous_vars[self.functions[key][1][n]]

                # Update variable according to its function and parameters
                self.endogenous_vars[key] = self.functions[key][0](**params)

            if old_obs == self.endogenous_vars:
                break

        return list(self.endogenous_vars.values()), list(self.exogenous_vars.values())

    def do_interventions(self, interventions: List[Tuple[str, Callable]]):
        """
        Replaces the functions of the SCM with the given interventions

        :param interventions: List of tuples
        """
        random.seed()
        for interv in interventions:
            self.endogenous_vars[interv[0]] = interv[1]()  # this is probably redundat with the next line
            self.functions[interv[0]] = (interv[1], {})


