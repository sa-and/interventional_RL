from typing import Tuple, List, Optional, NoReturn, Union, Any, overload
from abc import ABC, abstractmethod
from causalnex.structure import StructureModel
from causalnex.network import BayesianNetwork
from causalnex.inference import InferenceEngine
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from pandas import DataFrame
from itertools import combinations, permutations
import random


class CausalAgent(ABC):
    var_names: Union[int, List[str]]
    causal_model: StructureModel
    collected_data: dict

    def __init__(self, vars: Union[int, List[str]], causal_graph: StructureModel = None):
        if type(vars) == int:
            self.var_names = ['x' + str(i) for i in range(vars)]
        else:
            self.var_names = vars

        # initialize causal model
        if causal_graph:
            self.causal_model = causal_graph
        else:
            self.causal_model = StructureModel()
            [self.causal_model.add_node(name) for name in self.var_names]
            self.random_reset_causal_model()

        # initialize the storages for observational and interventional data.
        self.collected_data = {}

    def random_reset_causal_model(self):
        all_pairs = [(v[0], v[1]) for v in permutations(self.var_names, 2)]
        random.shuffle(all_pairs)
        for p in all_pairs:
            self.update_model(p, random.choice([0, 1, 2]))

    def update_model(self, edge: Tuple[str, str], manipulation: int) -> bool:
        '''Updates model according to action and returns the success of the operation
        0 = remove edge
        1 = add edge
        2 = reverse edge
        '''

        if manipulation == 0:  # remove edge if exists
            if self.causal_model.has_edge(edge[0], edge[1]):
                self.causal_model.remove_edge(edge[0], edge[1])
                removed_edge = (edge[0], edge[1])
            elif self.causal_model.has_edge(edge[1], edge[0]):
                self.causal_model.remove_edge(edge[1], edge[0])
                removed_edge = (edge[1], edge[0])
            else:
                return False

            if nx.number_weakly_connected_components(self.causal_model) > 1:  # disconnected graph
                self.causal_model.add_edge(removed_edge[0], removed_edge[1])
                return False

        elif manipulation == 1:  # add edge
            if not self.causal_model.has_edge(edge[0], edge[1]):  # only add edge if not already there
                self.causal_model.add_edge(edge[0], edge[1])
            else:
                return False

            if not nx.is_directed_acyclic_graph(self.causal_model):  # check if became cyclic
                self.causal_model.remove_edge(edge[0], edge[1])
                return False

        elif manipulation == 2:  # reverse edge
            if self.causal_model.has_edge(edge[0], edge[1]):
                self.causal_model.remove_edge(edge[0], edge[1])
                self.causal_model.add_edge(edge[1], edge[0])
                added_edge = (edge[1], edge[0])
            elif self.causal_model.has_edge(edge[1], edge[0]):
                self.causal_model.remove_edge(edge[1], edge[0])
                self.causal_model.add_edge(edge[0], edge[1])
                added_edge = edge
            else:
                return False

            if not nx.is_directed_acyclic_graph(self.causal_model):  # check if became cyclic
                self.causal_model.remove_edge(added_edge[0], added_edge[1])
                self.causal_model.add_edge(added_edge[1], added_edge[0])
                return False

        return True

    @abstractmethod
    def update_model_per_action(self, action: Any):
        raise NotImplementedError

    def get_est_avg_causal_effect(self, query: str, intervened_var: str, val1: Any, val2: Any) -> float:
        exp_val1 = self._get_expected_value(self.get_est_postint_distrib(query, intervened_var, val1))
        exp_val2 = self._get_expected_value(self.get_est_postint_distrib(query, intervened_var, val2))

        return exp_val1 - exp_val2

    @staticmethod
    def _get_expected_value(distribution: pd.Series) -> float:
        if type(distribution.index[0] == bool):
            distribution = distribution.rename({True: 1, False: 0})
        return sum(distribution.index.values * distribution._values)

    def get_est_postint_distrib(self, query: str, intervened_var: str, val: Any) -> pd.Series:
        '''Computes and returns P(Query | do(action))'''
        key = '('+intervened_var+','+str(val)+')'
        query_dataframe = self.collected_data[key]
        query_dataframe = query_dataframe.groupby(query).size()/len(query_dataframe)
        return query_dataframe.rename('P('+query+'|do'+key+')')

    def store_observation(self, obs: List[Any], intervened_var: Optional[str], val: Any):
        """
        Stores the observation of the environment in the appropriate dataframe.
        If no intervention is made (intervened_var=None), the observation is saved in the purely observational
        DataFrame.

        :param obs: Observation list of values
        :param intervened_var: On which var to intervene. Can be None
        :param val: The assigned value of the intervened variable
        """
        if intervened_var == None:
            key = '(None, None)'
        else:
            key = '(' + intervened_var + ',' + str(val) + ')'

        # put the observation in the right dataframe
        obs_dict = {self.var_names[i]: obs[i] for i in range(len(self.var_names))}

        if key in self.collected_data:
            self.collected_data[key] = self.collected_data[key].append(obs_dict, ignore_index=True)
        else:
            self.collected_data[key] = obs_dict

    @abstractmethod
    def store_observation_per_action(self, obs: List[Any], action: Any):
        raise NotImplementedError

    def get_est_cond_distr(self, query: str, var: str, val: Any) -> DataFrame:
        obs_data = self.collected_data['(None, None)']
        obs_data = obs_data[obs_data[var] == val]
        obs_data = obs_data.groupby(query).size() / len(obs_data)
        return obs_data.rename('P('+query+'|'+var+'='+str(val)+')')

    def display_causal_model(self) -> NoReturn:
        fig, ax = plt.subplots()
        nx.draw_circular(self.causal_model, ax=ax, with_labels=True)
        fig.show()

    def evaluate_causal_model(self) -> float:
        '''
        Evaluate how well the estimated model of the agent fits the interventional data collected from the
        actual environment.
        :return:
        '''
        if len(self.collected_data['(None, None)']) < 100:  # minimal amout of observations for evaluation
            return -2
        # estimate bayesian network from the structure of the model and observational data
        bn = BayesianNetwork(self.causal_model)
        if type(self.collected_data) == bool:
            bn.fit_node_states_and_cpds(self.collected_data['(None, None)'].replace([True, False], [1, 0]))
        else:
            raise NotImplementedError

        ie = InferenceEngine(bn)
        var_pairs = [(v[0], v[1]) for v in permutations(self.var_names, 2)]
        var_pairs = [pair for pair in var_pairs if self.is_legal_intervention(pair[0])]  # only legal pairs

        losses = []
        for pair in var_pairs:
            for val in [0, 1]:  # TODO: generalize this to the actual domain of the variables (maybe through bn.Node_states)
                did = False  # TODO: move this right after the start of outer loop or filter before loop
                try:
                    ie.do_intervention(pair[0], val)
                    did = True
                    predicted_dist = pd.Series(ie.query()[pair[1]])
                    est_true_distribution = self.get_est_postint_distrib(pair[1], pair[0], bool(val))

                    if len(self.collected_data[str((pair[0], bool(val)))]) > 4:  # minimal size for interventional distributions
                        expvalpred = self._get_expected_value(predicted_dist)
                        expvaltrue = self._get_expected_value(est_true_distribution)
                        losses.append((expvalpred-expvaltrue)**2)

                except ValueError as e:
                    print(e)

                if did:
                    ie.reset_do(pair[0])

        if len(losses) == 0:  # all interventional distributions were too small
            return -2
        else:
            return -sum(losses)/len(losses)

    def graph_is_learned(self, threshold) -> bool:
        eval = self.evaluate_causal_model()
        print(eval)
        return eval > threshold

    def get_graph_state(self) -> List[float]:
        '''
        Get a list of values that represents the state of an edge in the causal graph for each possible graph.
        The edges are ordered in lexographical order.

        Example:
        In a 3 node graph there are the potential edges: 0-1, 0-2, 1-2. The list [0, 0.5, 1] represents the
        graph 0x1, 0->2, 1<-2, where x means that there is no edge.

        :return: state of the graph
        '''
        graph_state = []
        possible_edges = [e for e in combinations(self.var_names, 2)]
        for e in possible_edges:
            if self.causal_model.has_edge(e[0], e[1]):
                graph_state.append(0.5)
            elif self.causal_model.has_edge(e[1], e[0]):
                graph_state.append(1.0)
            else:
                graph_state.append(0.0)
        return graph_state

    def is_legal_intervention(self, interv_var: str) -> bool:
        """
        Checks if performing an intervention disconnects the graph. If it does, it is not a legal intervention
        for the causalnex library.
        :param interv_var: variable to intervene on
        :return: legal
        """
        model = self.causal_model.copy()
        nodes = nx.nodes(model)
        for n in nodes:
            if model.has_edge(n, interv_var):
                model.remove_edge(n, interv_var)
        is_connected = nx.number_weakly_connected_components(model) <= 1
        return is_connected


class SwitchboardAgentDQN(CausalAgent):
    current_mode: str
    action = Tuple[Optional[int], Optional[Union[int, Tuple[str, str]]], Optional[Union[bool, int]]]
    intv_action = Tuple[Optional[int], Optional[bool]]
    actions: List[action]

    def __init__(self, n_switches: int, causal_graph: StructureModel = None):
        super(SwitchboardAgentDQN, self).__init__(n_switches, causal_graph)

        # create a list of actions that can be performed on the switchboard
        # actions for interventions represented as (0, variable, value)
        self.actions = [(0, i, True) for i in range(n_switches)]
        self.actions.extend([(0, i, False) for i in range(n_switches)])
        # actions for graph manipulation represented as (1, edge, operation)
        # where operation can be one of: delete = 0, add = 1, reverse = 2
        edges = [e for e in combinations(self.var_names, 2)]
        for i in range(3):
            extensions = [(1, edge, i) for edge in edges]
            self.actions.extend([(1, edge, i) for edge in edges])
        self.actions.append((None, None, None))

    def store_observation_per_action(self, obs: List[bool], current_action: action):
        if current_action[0] == 1 or current_action[0] == None:  # no itervention
            self.store_observation(obs, None, None)
        else:
            self.store_observation(obs, current_action[1], current_action[2])

    def update_model_per_action(self, action: action) -> bool:
        '''Updates model according to action and returns the success of the operation'''
        assert action[0] == 1, "Action is not a b model manipulation."
        edge = action[1]
        manipulation = action[2]

        return self.update_model(edge, manipulation)


def get_switchboard_causal_graph() -> StructureModel:
    model = StructureModel()
    [model.add_node(name) for name in ['x'+str(i) for i in range(5)]]
    model.add_edge('x0', 'x1')
    model.add_edge('x2', 'x1')
    model.add_edge('x2', 'x3')
    model.add_edge('x4', 'x0')
    model.add_edge('x4', 'x1')
    model.add_edge('x4', 'x2')
    return model


def get_wrong_switchboard_causal_graph() -> StructureModel:
    model = StructureModel()
    [model.add_node(name) for name in ['x'+str(i) for i in range(5)]]
    model.add_edge('x1', 'x0')
    model.add_edge('x1', 'x2')
    model.add_edge('x1', 'x4')
    model.add_edge('x3', 'x2')
    model.add_edge('x0', 'x4')
    model.add_edge('x2', 'x4')
    return model
