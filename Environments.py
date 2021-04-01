from typing import List, Callable, Tuple, NoReturn, Any

from gym import Env
import random
from Agents import CausalAgent, DiscreteSwitchboardAgent, ContinuousSwitchboardAgent,\
    get_switchboard_causal_graph, get_almost_right_switchboard_causal_graph, get_blank_switchboard_causal_graph
import copy
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


class Switchboard(Env):
    Agent: DiscreteSwitchboardAgent
    Function = Callable[[], bool]
    Lights: List[bool]

    def __init__(self, agent: CausalAgent, fixed_episode_length: bool = False):
        super(Switchboard, self).__init__()
        self.metrics = {'ep_lengths': [],
                        'rewards': []}
        self.fixed_episode_length = fixed_episode_length

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
        self.last_observation = None
        self.old_obs = []
        for i in range(self.agent.state_repeats):
            self.old_obs.append([0.0 for i in range(int(self.observation_space.shape[0]/self.agent.state_repeats))])

        self.steps_this_episode = 0

    def reset(self) -> np.ndarray:
        self.steps_this_episode = 0
        # self.agent.set_causal_model(get_blank_switchboard_causal_graph())
        self.agent.random_reset_causal_model()
        # reset observations
        self.old_obs = []
        for i in range(self.agent.state_repeats):
            self.old_obs.append([0.0 for i in range(int(self.observation_space.shape[0] / self.agent.state_repeats))])

        return self.get_obs_vector()

    def step(self, action) -> Tuple[np.ndarray, float, bool, dict]:
        self.current_action = self.agent.get_action_from_actionspace_sample(action)

        # apply action
        interv_scm = copy.deepcopy(self.SCM)
        action_successful = False
        if self.current_action[0] == 0:  # intervention action
            interv_scm.do_interventions([('X'+str(self.current_action[1]), lambda: self.current_action[2])])
            action_successful = True
        elif self.current_action[0] == 1:
            action_successful = self.agent.update_model_per_action(self.current_action)
        elif self.current_action[0] == None or self.current_action[0] == -1:
            action_successful = True

        self.steps_this_episode += 1

        # determine the states of the lights according to the causal structure
        self.lights = interv_scm.get_next_instantiation()[0]

        self.agent.store_observation_per_action(self.lights, self.current_action)

        # reverse all wrong edges, this could eventually speed up learning
        #self.agent.reverse_wrong_edges(0.1)

        # determine state after action
        self.last_observation = self.get_obs_vector()

        if self.fixed_episode_length:
            done, reward = self.do_fixed_eval(action_successful, 20)
        else:
            # let the episode end when the causal model is fully learned (loss reaching threshold of -0.006)
            done, reward = self.do_flexible_eval(action_successful)

        self.prev_action = self.current_action
        self.metrics['rewards'].append(reward)
        if type(self.agent) == ContinuousSwitchboardAgent:
            print([round(a) for a in self.current_action], '\treward', reward)
        else:
            print(self.current_action, '\treward', reward)

        return self.last_observation, reward, done, {}

    def do_fixed_eval(self, action_successful: bool,
                      length_per_episode: int,
                      allow_unsuccessful_actions: bool = True) -> Tuple[bool, float]:
        '''
        Ends the episode after 'length_per_episode' steps. Here we only give a reward for the
        achievement of the goal at the end of each episode. A negative reward for illegal actions
        is still returned.
        :param action_successful:
        :param length_per_episode:
        :param allow_unsuccessful_actions:
        :return:
        '''
        done = almost_done = very_almost_done = learned = False
        if self.steps_this_episode >= length_per_episode:
            done = True
            n_wrong_edges = self.agent.has_wrong_edges(0.1)
            n_missing_edges = self.agent.has_missing_edges(0.1)
            learned = (n_wrong_edges + n_missing_edges == 0)
            almost_done = (n_wrong_edges + n_missing_edges < 4) and (n_wrong_edges + n_missing_edges >= 2)
            very_almost_done = (n_wrong_edges + n_missing_edges < 2) and (n_wrong_edges + n_missing_edges > 0)

        if not action_successful and not allow_unsuccessful_actions:  # illegal action was taken
            reward = -1
        elif almost_done:
            reward = 2
        elif very_almost_done:
            reward = 5
        elif learned:
            reward = 30
            self.reset()
        else:
            reward = 0

        return done, reward

    def do_flexible_eval(self, action_successful: bool, allow_unsuccessful_actions: bool = True) -> Tuple[bool, float]:
        '''
        Ends the episode whenever a graph-altering action is performed
        :param action_successful:
        :param allow_unsuccessful_actions:
        :return:
        '''
        if self.current_action[0] == 1:  # only check if the model actually changed.
            n_wrong_edges = self.agent.has_wrong_edges(0.1)
            n_missing_edges = self.agent.has_missing_edges(0.1)
            done = (n_wrong_edges + n_missing_edges == 0)
            almost_done = (n_wrong_edges + n_missing_edges < 4) and (n_wrong_edges + n_missing_edges >= 2)
            very_almost_done = (n_wrong_edges + n_missing_edges < 2) and (n_wrong_edges + n_missing_edges > 0)

        else:
            done = False
            almost_done = False
            very_almost_done = False

        # compute reward
        if not action_successful and not allow_unsuccessful_actions:  # illegal action was taken
            reward = -1
        elif almost_done:
            reward = 2
        elif very_almost_done:
            reward = 5
        elif done:  # the graph has been learned
            reward = 30
            self.metrics['ep_lengths'].append(self.steps_this_episode)
            self.reset()
        else:  # intervention, non-intervention, graph-changing
            reward = 0
        return done, reward

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


class BoolSCMGenerator:
    '''
    Class to help creating SCMs with boolean variables and relationships as in the switchboard environment
    '''

    def __init__(self, n_endo: int, n_exo: int, allow_exo_confounders: bool = False):
        self.allow_exo_confounders = allow_exo_confounders

        # create var names
        self.endo_vars = ['X' + str(i) for i in range(n_endo)]
        self.exo_vars = ['U' + str(i) for i in range(n_exo)]

        # determine potential causes for each endogenous var
        self.potential_causes = {}
        exo_copy = copy.deepcopy(self.exo_vars)
        for v in self.endo_vars:
            if allow_exo_confounders:
                self.potential_causes[v] = self.endo_vars + self.exo_vars
            else:
                self.potential_causes[v] = self.endo_vars + [exo_copy.pop()]
            self.potential_causes[v].remove(v)

    def create_random(self) -> StructuralCausalModel:
        # get random causes
        # causes = {}
        # for key, cs in self.potential_causes.items():
        #     causes[key] = random.sample(cs, random.randint(1, len(cs)))
        #
        # # functions = {}
        # # for v in causes:
        # #     def function(**cs):
        # #         ret = False
        # #         for c in causes[v]:
        # #             ret = ret or cs[c]
        # #         return ret
        # #     functions[v] = lambda **cs: any(cs[c] for c in causes[v])
        # # pass
        # #
        # scm = StructuralCausalModel()
        # functions = []
        # for v in causes:
        #     functions.append(lambda **kwargs: any([kwargs[c] for c in causes[v]]))
        #     scm.add_endogenous_var(v, False, copy.deepcopy(lambda **cs: any(cs[c] for c in causes[v])), {c:c for c in causes[v]})
        #
        # return scm
        # # ('X0', False, lambda x4, u0: x4 or u0, {'x4': 'X4', 'u0': 'U0'})

        ##########################
        # causes = {u: [] for u in self.endo_vars}
        # # select random exo vars
        # exos = random.sample(self.exo_vars, random.randint(0, len(self.exo_vars)))
        # remaining = copy.deepcopy(self.endo_vars)
        # processed = []
        #
        # # Determine which nodes can still be roots
        # previous_layer = []
        # possible_roots = [r for r in remaining]
        # for u in exos:
        #     for v in self.endo_vars:
        #         if v[-1] == u[-1]:
        #             possible_roots.remove(v)
        #             previous_layer.append(v)
        #             causes[v].append(u)
        #
        # # randomly select roots
        # roots = random.sample(possible_roots, random.randint(0, len(possible_roots)))
        # previous_layer.extend(roots)
        # [remaining.remove(r) for r in previous_layer]
        #
        # while len(remaining) > 0:
        #     next_layer = random.sample(remaining, random.randint(1, len(remaining)))
        #     prev_new = [n for n in next_layer]
        #     for n in next_layer:
        #         parents = random.sample(previous_layer + next_layer,
        #                                 random.randint(1, len(previous_layer + next_layer)))
        #         if n in parents:
        #             parents.remove(n)
        #         next_layer.remove(n)
        #         causes[n].extend(parents)
        #     previous_layer = prev_new
        #     [remaining.remove(p) for p in previous_layer]

        # generate fully connected graph
        graph = nx.DiGraph()
        [graph.add_node(u) for u in self.exo_vars]
        [graph.add_node(v) for v in self.endo_vars]
        for n, cs in self.potential_causes.items():
            [graph.add_edge(c, n) for c in cs]
            
        # delete random edges until acyclic
        while not nx.is_directed_acyclic_graph(graph):
            random_edge = random.sample(graph.edges, 1)[0]
            graph.remove_edge(random_edge[0], random_edge[1])

        fig, ax = plt.subplots()
        nx.draw_circular(graph, ax=ax, with_labels=True)
        fig.show()

        # create scm
        scm = StructuralCausalModel()
        for n in graph.nodes:
            parents = [p for p in graph.predecessors(n)]

            def make_f(parents):
                def f(**kwargs):
                    res = False
                    for p in parents:
                        res = res or kwargs[p]
                    return res
                return f

            if n[0] == 'X':
                scm.add_endogenous_var(n, False, make_f(parents), {p:p for p in parents})
            else:
                scm.add_exogenous_var(n, False, random.choice, {'seq': [True, False]})
        return scm

    def create_n(self, n: int) -> StructuralCausalModel:
        pass


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