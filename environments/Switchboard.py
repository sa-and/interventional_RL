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
        self.rewards = []

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
        self.rewards.append(reward)
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
            assert name in self.endogenous_vars, 'Variable not in list of edogenous vars'

            del self.endogenous_vars[name]
            del self.functions[name]

        else:
            assert name in self.exogenous_vars, 'Variable not in list of edogenous vars'

            del self.exogenous_vars[name]
            del self.exogenous_distributions[name]

    def get_next_instantiation(self) -> Tuple[List, List]:
        """
        Returns a new instantiation of variables consistent with the causal structure and for a sample from the
        exogenous distribution
        :return: Instantiation of endogenous and exogenous variables
        """
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

    def get_intervened_scm(self, interventions: List[Tuple[str, Callable]]):
        """
        Creates an SCM in which all nodes in interventions (on endogenous variables)
        have been set to the function provided. The provided functions are assumed to have no parameters

        :param interventions: List of tuples
        :return: StructuralCausalModel
        """
        interv_scm = copy.deepcopy(self)
        for interv in interventions:
            interv_scm.endogenous_vars[interv[0]] = interv[1]()  # this is probably redundat with the next line
            interv_scm.functions[interv[0]] = (interv[1], {})

        return interv_scm

