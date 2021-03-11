from typing import List, Callable, Tuple, NoReturn, Any

from gym import Env
import random
from Agents import CausalAgent, DiscreteSwitchboardAgent, ContinuousSwitchboardAgent, get_switchboard_causal_graph
import copy
import numpy as np
import networkx as nx


class Switchboard(Env):
    Agent: DiscreteSwitchboardAgent
    Function = Callable[[], bool]
    Lights: List[bool]

    def __init__(self, agent: CausalAgent):
        super(Switchboard, self).__init__()
        # initialize causal model
        self.SCM = make_switchboard_scm_without_context()

        self.lights = [False]*5  # all lights are off

        assert type(agent) == DiscreteSwitchboardAgent or type(agent) == ContinuousSwitchboardAgent, \
            'Wrong agent for this environment'

        self.agent = agent
        self.action_space = self.agent.action_space
        if type(self.agent) == DiscreteSwitchboardAgent:
            self.current_action = (None, None, None)
        elif type(self.agent) == ContinuousSwitchboardAgent:
            self.current_action = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.observation_space = self.agent.observation_space
        self.prev_action = None
        self.old_obs = []
        for i in range(self.agent.state_repeats):
            self.old_obs.append([0.0 for i in range(int(self.observation_space.shape[0]/self.agent.state_repeats))])

        self.rewards = []

    def reset(self) -> np.ndarray:
        self.agent.random_reset_causal_model()
        # reset observations
        self.old_obs = []
        for i in range(self.agent.state_repeats):
            self.old_obs.append([0.0 for i in range(int(self.observation_space.shape[0] / self.agent.state_repeats))])

        return self.get_obs_vector()

    def step(self, action) -> Tuple[np.ndarray, float, bool, dict]:
        self.current_action = self.agent.get_action_from_actionspace_sample(action)

        interv_scm = copy.deepcopy(self.SCM)
        # apply action
        action_successful = False
        if self.current_action[0] == 0:  # intervention action
            interv_scm.do_interventions([('X'+str(self.current_action[1]), lambda: self.current_action[2])])
            action_successful = True
        # elif self.current_action[0] == 1 and self.current_action[-1] == 2 and self.prev_action[-1] == 2:  # don't allow reversal of edge that has just been reversed
        #     if type(self.agent) == DeterministicSwitchboardAgent and self.current_action[1] == self.prev_action[1]:
        #         action_successful = False
        #     elif type(self.agent) == ContinuousSwitchboardAgent and\
        #             self.current_action[3] == self.prev_action[3] and\
        #             self.current_action[4] == self.prev_action[4]:
        #         action_successful = False
        elif self.current_action[0] == 1:
            action_successful = self.agent.update_model_per_action(self.current_action)
        elif self.current_action[0] == None or self.current_action[0] == -1:
            action_successful = True

        # determine the states of the lights according to the causal structure
        self.lights = interv_scm.get_next_instantiation()[0]

        self.agent.store_observation_per_action(self.lights, self.current_action)

        # determine state after action
        self.last_observation = self.get_obs_vector()

        # let the episode end when the causal model is fully learned (loss reaching threshold of -0.006)
        if self.current_action[0] == 1:  # only check if the model actually changed.
            done = self.agent.graph_is_learned()
            almost_learned = nx.graph_edit_distance(self.agent.causal_model,
                                                    get_switchboard_causal_graph()) \
                             == 2
            very_almost_learned = nx.graph_edit_distance(self.agent.causal_model,
                                                    get_switchboard_causal_graph()) \
                             == 1
        else:
            done = False
            almost_learned = False
            very_almost_learned = False

        # compute reward
        if not action_successful:  # illegal action was taken
            reward = -1
        elif almost_learned:
            reward = 2
        elif very_almost_learned:
            reward = 3
        elif done:  # the graph has been learned
            reward = 30
            self.agent.display_causal_model()
            self.reset()
        else:  # intervention, non-intervention, graph-changing
            reward = 0

        self.prev_action = self.current_action
        self.rewards.append(reward)
        if type(self.agent) == ContinuousSwitchboardAgent:
            print([round(a) for a in self.current_action], '\treward', reward)
        else:
            print(self.current_action, '\treward', reward)

        return self.last_observation, reward, done, {}

    def get_obs_vector(self) -> np.ndarray:
        # push old observations
        for i in range(1, len(self.old_obs)):
            self.old_obs[i-1] = self.old_obs[i]
            
        intervention_one_hot = [1.0 if self.current_action[1] == i else 0.0 for i in range(len(self.lights))]
        graph_state = self.agent.get_graph_state()
        state = [float(l) for l in self.lights]  # convert bool to int
        state.extend(intervention_one_hot)
        state.extend(graph_state)
        self.old_obs[-1] = state
        return np.array(self.old_obs).flatten()

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


def make_switchboard_scm_with_context():
    SCM = StructuralCausalModel()
    SCM.add_exogenous_vars([('U' + str(i), True, random.choice, {'seq': [True, False]}) for i in range(5)])
    SCM.add_endogenous_vars(
        [
            ('X0', False, lambda x4, u0: x4 or u0, {'x4': 'X4', 'u0': 'U0'}),
            (
            'X1', False, lambda x0, x2, x4, u1: x0 or x2 or x4 or u1, {'x0': 'X0', 'x2': 'X2', 'x4': 'X4', 'u1': 'U1'}),
            ('X2', False, lambda x4, u2: x4 or u2, {'x4': 'X4', 'u2': 'U2'}),
            ('X3', False, lambda x2, u3: x2 or u3, {'x2': 'X2', 'u3': 'U3'}),
            ('X4', False, lambda u4: u4, {'u4': 'U4'})
        ])

    return SCM


def make_switchboard_scm_without_context():
    SCM = StructuralCausalModel()
    SCM.add_endogenous_vars(
        [
            ('X0', False, lambda x4: x4, {'x4': 'X4'}),
            ('X1', False, lambda x0, x2, x4: x0 or x2 or x4, {'x0': 'X0', 'x2': 'X2', 'x4': 'X4'}),
            ('X2', False, lambda x4: x4, {'x4': 'X4'}),
            ('X3', False, lambda x2: x2, {'x2': 'X2'}),
            ('X4', False, lambda: False, {})
        ])

    return SCM