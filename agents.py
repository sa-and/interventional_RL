from typing import Tuple, List, Optional, NoReturn, Union, Any
from abc import ABC, abstractmethod
from causalnex.structure import StructureModel
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from pandas import DataFrame
from itertools import combinations, permutations
import random
from gym.spaces import Discrete, Box
import numpy as np


class CausalAgent(ABC):
    var_names: Union[int, List[str]]
    causal_model: StructureModel
    collected_data: dict
    actions: List[Any]
    state_repeats: int
    action_space: Union[Discrete, Box]

    def __init__(self, vars: Union[int, List[str]], causal_graph: StructureModel = None, env_type: str = 'Switchboard'):
        self.env_type = env_type
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

        self.action_space = None
        self.observation_space = None
        self.actions = []
        self.state_repeats = None
        self.current_action = None

    def set_causal_model(self, causal_model: StructureModel):
        self.causal_model = causal_model

    def random_reset_causal_model(self):
        all_pairs = [(v[0], v[1]) for v in permutations(self.var_names, 2)]
        random.shuffle(all_pairs)
        for p in all_pairs:
            self.update_model(p, random.choice([0, 1, 2]))

    def update_model(self, edge: Tuple[str, str],
                     manipulation: int,
                     allow_disconnecting: bool = True,
                     allow_cycles: bool = True) -> bool:
        '''Updates model according to action and returns the success of the operation
        0 = remove edge
        1 = add edge
        2 = reverse edge

        allow_disconnecting: False if actions resulting in a disconnected graph should be illegal
        '''

        if manipulation == 0:  # remove edge if exists
            if self.causal_model.has_edge(edge[0], edge[1]):
                self.causal_model.remove_edge(edge[0], edge[1])
                removed_edge = (edge[0], edge[1])
            # elif self.causal_model.has_edge(edge[1], edge[0]):
            #     self.causal_model.remove_edge(edge[1], edge[0])
            #     removed_edge = (edge[1], edge[0])
            else:
                return False

            # disconnected graph
            if not allow_disconnecting and nx.number_weakly_connected_components(self.causal_model) > 1:
                self.causal_model.add_edge(removed_edge[0], removed_edge[1])
                return False

        elif manipulation == 1:  # add edge
            if not self.causal_model.has_edge(edge[0], edge[1]):  # only add edge if not already there
                self.causal_model.add_edge(edge[0], edge[1])
            else:
                return False

            if not nx.is_directed_acyclic_graph(self.causal_model) and not allow_cycles:  # check if became cyclic
                self.causal_model.remove_edge(edge[0], edge[1])
                return False

        elif manipulation == 2:  # reverse edge
            if self.causal_model.has_edge(edge[0], edge[1]):
                self.causal_model.remove_edge(edge[0], edge[1])
                self.causal_model.add_edge(edge[1], edge[0])
                added_edge = (edge[1], edge[0])
            # elif self.causal_model.has_edge(edge[1], edge[0]):
            #     self.causal_model.remove_edge(edge[1], edge[0])
            #     self.causal_model.add_edge(edge[0], edge[1])
            #     added_edge = edge
            else:
                return False

            if not nx.is_directed_acyclic_graph(self.causal_model) and not allow_cycles:  # check if became cyclic
                self.causal_model.remove_edge(added_edge[0], added_edge[1])
                self.causal_model.add_edge(added_edge[1], added_edge[0])
                return False

        return True

    def get_est_avg_causal_effect(self, query: str, intervened_var: str, val1: Any, val2: Any) -> float:
        exp_val1 = self._get_expected_value(self.get_est_postint_distrib(query, intervened_var, val1))
        exp_val2 = self._get_expected_value(self.get_est_postint_distrib(query, intervened_var, val2))

        return exp_val1 - exp_val2

    def get_all_avg_causal_effects(self):
        edges = [e for e in combinations(self.var_names, 2)]
        edges.extend([(e[1], e[0]) for e in edges])
        effects = []
        for e in edges:
            effects.append((e, self.get_est_avg_causal_effect(e[1], e[0], True, False)))
        return effects

    def compare_edge_to_data(self, edge: Tuple[str, str], threshold: float = 0.0) -> bool:
        '''
        Checks whether the edge of the model corresponds to an actual causal effect in the interventional data. So for
        a given edge A -> B it checks whether P(B|do(A=true)) != P(B|do(A=false)) holds in the collected interventional
        data set.
        Note that this direct effect suggested in the model could actually be an indirect one in the data. Still, this
        methods returns true in this case.

        :param edge: The edge to be checked. E.g. ('x1', 'x2')
        :param threshold: The value from which on the effect is assumed to be present.
        :return: Whether the causal edge is backed up by the interventional data.
        '''
        assert self.causal_model.has_edge(edge[0], edge[1]), 'The given edge is not part of the current model.'

        if '(' + edge[0] + ',True)' in self.collected_data and '(' + edge[0] + ',False)' in self.collected_data:
            est_causal_effect = self.get_est_avg_causal_effect(edge[1], edge[0], True, False)
            if abs(est_causal_effect) >= threshold:
                return True
            else:
                return False

        elif '(' + edge[0] + ',0.0)' in self.collected_data and '(' + edge[0] + ',5.0)' in self.collected_data:
            est_causal_effect = self.get_est_avg_causal_effect(edge[1], edge[0], 0.0, 5.0)
            if abs(est_causal_effect) >= threshold:
                return True
            else:
                return False

        return True

    def has_wrong_edges(self, threshold: float = 0.0) -> int:
        '''
        Determines how many edges in the current causal model do not have a causal effect in the interventional
        data set that is bigger than the given threshold.
        :param threshold:
        :return: number of 'wrong' edges
        '''
        count = 0
        for e in self.causal_model.edges:
            if not self.compare_edge_to_data(e, threshold):
                count += 1
        return count

    def reverse_wrong_edges(self, threshold: float = 0.0) -> NoReturn:
        '''
        Checks all edges whether they are the wrong way around and reverses those that are.

        :param threshold:
        '''
        wrong_edges = []
        for e in self.causal_model.edges:
            if not self.compare_edge_to_data(e, threshold):
                wrong_edges.append(e)

        for e in wrong_edges:
            self.update_model(e, 2)
    
    def edge_is_missing(self, edge: Tuple[str, str], threshold: float = 0.0) -> bool:
        '''
        Checks whether for the given edge (which is not part of the model) there is a causal effect in the collected
        interventional data. If true, there should be a directed path between edge[0] and edge[1] in the model but
        there is none. This means that along the path from edge[0] to edge[1] at least one edge is missing.
        
        :param edge: Edge to check
        :param threshold: Value from which on the effect is to be considered an actual effect.
        :return: Whether, according to the interventional data there should be an edge but is none.
        '''
        if edge in self.causal_model.edges:
            return False

        elif nx.has_path(self.causal_model, edge[0], edge[1]):
            return False
        
        else:
            if '(' + edge[0] + ',True)' in self.collected_data and '(' + edge[0] + ',False)' in self.collected_data:
                effect = self.get_est_avg_causal_effect(edge[1], edge[0], True, False)
                return abs(effect) >= threshold
            elif '(' + edge[0] + ',0.0)' in self.collected_data and '(' + edge[0] + ',5.0)' in self.collected_data:
                effect = self.get_est_avg_causal_effect(edge[1], edge[0], 0.0, 5.0)
                return abs(effect) >= threshold
            else:
                return True

        
    def has_missing_edges(self, threshold: float = 0.0) -> int:
        '''
        Returns the maximal number of missing edges in the model according to the collected interventional
        data. The maximum number is returned because the exact number cannot be determined with an intervention
        on a single variable.

        :param threshold:
        :return:

        Example
        ---------
        Let the ground truth causal model be A -> B -> C and the causal model of the agent A   B -> C (missing
        edge between A and B). This method will return 2. This is because the edge between A and B induces an indirect
        effect of A on C which cannot be distilled from a direct effect that could be present from A to C.

        The collected interventional data with the intervention only on one variable cannot distinguish between
        A -> B -> C and A -> B -> C, hence a maximum of 2 edges are missing.
                        |         ^
                        - - - - - |
        Important: Once there is any path from A to C, no edge is considered to be missing.
        e.g. applied to the model A -> B -> C this method returns 0
        e.g. applied to the model A -> C <- B this method returns 1 as the edge A -> B is missing.

        '''
        missing_edges = 0
        # check which causal relationships are missing in the graph
        for n in self.causal_model.nodes:
            # iterate over all nodes that do not already have an edge from n
            for nn in nx.non_neighbors(self.causal_model, n):
                current_edge = (str(n), str(nn))
                if self.edge_is_missing(current_edge, threshold):
                    missing_edges += 1
        
        return missing_edges

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
        obs_dict = pd.DataFrame().append({self.var_names[i]: obs[i] for i in range(len(self.var_names))},
                                         ignore_index=True)

        if key in self.collected_data:
            self.collected_data[key] = self.collected_data[key].append(obs_dict, ignore_index=True)
        else:
            self.collected_data[key] = obs_dict

    def get_est_cond_distr(self, query: str, var: str, val: Any) -> DataFrame:
        obs_data = self.collected_data['(None, None)']
        obs_data = obs_data[obs_data[var] == val]
        obs_data = obs_data.groupby(query).size() / len(obs_data)
        return obs_data.rename('P('+query+'|'+var+'='+str(val)+')')

    def display_causal_model(self) -> NoReturn:
        fig, ax = plt.subplots()
        nx.draw_circular(self.causal_model, ax=ax, with_labels=True)
        fig.show()

    def graph_is_learned(self, threshold: float = 0.0) -> bool:
        n_wrong_edges = self.has_wrong_edges(threshold)
        print('wrong edges: ', n_wrong_edges)
        n_missing_edges = self.has_missing_edges(threshold)
        print('missing edges: ', n_missing_edges)
        return n_wrong_edges == 0 and n_missing_edges == 0

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

    @staticmethod
    def _get_expected_value(distribution: pd.Series) -> float:
        if type(distribution.index[0] == bool):
            distribution = distribution.rename({True: 1, False: 0})
        return sum(distribution.index.values * distribution._values)

    @abstractmethod
    def get_action_from_actionspace_sample(self, sample: Any):
        raise NotImplementedError

    @abstractmethod
    def store_observation_per_action(self, obs: List[Any]):
        raise NotImplementedError

    @abstractmethod
    def update_model_per_action(self, action: Any):
        raise NotImplementedError


class DiscreteAgent(CausalAgent):
    current_mode: str
    state_repeats: int

    def __init__(self, n_vars: int, causal_graph: StructureModel = None, state_repeats: int = 1, env_type: str = 'Switchboard'):
        super(DiscreteAgent, self).__init__(n_vars, causal_graph, env_type)
        # create a list of actions that can be performed
        if self.env_type == 'Switchboard':
            # actions for interventions represented as (0, variable, value)
            self.actions = [(0, i, True) for i in range(n_vars)]
            self.actions.extend([(0, i, False) for i in range(n_vars)])
            self.observation_space = Box(0, 1,
                                         (state_repeats * (int((n_vars * 2) + n_vars * (n_vars - 1) / 2)),))
        elif self.env_type == 'Dasgupta':
            self.actions = [(0, i, 5.0) for i in range(n_vars)]
            self.actions.extend([(0, i, 0.0) for i in range(n_vars)])
            self.observation_space = Box(-7.0, 7.0, (state_repeats*(int((n_vars * 2) + n_vars * (n_vars - 1) / 2)),))
        else:
            raise NotImplementedError('Environment type not supported. Choose "Switchboard" or "Dasgupta"')

        # actions for graph manipulation represented as (1, edge, operation)
        # where operation can be one of: delete = 0, add = 1, reverse = 2
        edges = [e for e in combinations(self.var_names, 2)]
        edges.extend([(e[1], e[0]) for e in edges])
        for i in range(3):
            extensions = [(1, edge, i) for edge in edges]
            self.actions.extend([(1, edge, i) for edge in edges])
        self.actions.append((None, None, None))
        self.current_action = (None, None, None)

        self.action_space = Discrete(len(self.actions))
        self.state_repeats = state_repeats

    def store_observation_per_action(self, obs: List[bool]):
        if self.current_action[0] == 1 or self.current_action[0] == None:  # no itervention
            self.store_observation(obs, None, None)
        else:
            self.store_observation(obs, self.var_names[self.current_action[1]], self.current_action[2])

    def update_model_per_action(self, action) -> bool:
        '''Updates model according to action and returns the success of the operation'''
        assert action[0] == 1, "Action is not a b model manipulation."
        edge = action[1]
        manipulation = action[2]

        return self.update_model(edge, manipulation)

    def get_action_from_actionspace_sample(self, sample: int):
        return self.actions[sample]


class ContinuousSwitchboardAgent(CausalAgent):
    action = List[int]  # (action_type, inter_var, inter_val, edge_u, edge_v, mani)

    def __init__(self, n_switches: int, causal_graph: StructureModel = None, state_repeats: int = 1):
        super(ContinuousSwitchboardAgent, self).__init__(n_switches, causal_graph)

        self.action_space = Box(low=np.array([-1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
                                high=np.array([1.0,
                                               float(len(self.var_names)),
                                               1.0,
                                               float(len(self.var_names)),
                                               float(len(self.var_names)),
                                               2.0]),
                                shape=(6,))
        self.state_repeats = state_repeats
        self.observation_space = Box(0,
                                     1,
                                     (state_repeats * (int((n_switches * 2) + n_switches * (n_switches - 1) / 2)),))
        self.current_action = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    def store_observation_per_action(self, obs: List[Any]):
        if self.current_action[0] == -1 or self.current_action[0] == 1:  # no itervention
            self.store_observation(obs, None, None)
        else:
            self.store_observation(obs, self.var_names[self.current_action[1]], bool(self.current_action[2]))

    def update_model_per_action(self, action: action) -> bool:
        if action[3] == action[4]:  # edge on the same node
            return False
        return self.update_model((self.var_names[action[3]], self.var_names[action[4]]), action[5])

    def get_action_from_actionspace_sample(self, sample: List[float]) -> List[int]:
        action = [round(s) for s in sample]

        # action type one of -1, 0, 1
        if action[0] < 0:
            action[0] = -1
        elif action[0] > 1:
            action[0] = 1

        # interv var
        if action[1] < 0:
           action[1] = 0
        elif action[1] >= len(self.var_names):
            action[1] = len(self.var_names) - 1

        # value for inter var
        if action[2] < 1:
            action[2] = 0
        else:
            action[2] = 1

        # edge u
        if action[3] < 0:
            action[3] = 0
        elif action[3] >= len(self.var_names):
            action[3] = len(self.var_names) - 1

        # edge v
        if action[4] < 0:
           action[4] = 0
        elif action[4] >= len(self.var_names):
            action[4] = len(self.var_names) - 1

        # manipulation
        if action[5] < 0:
            action[5] = 0
        elif action[5] > 2:
            action[5] = 2

        return action


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


def get_almost_right_switchboard_causal_graph() -> StructureModel:
    model = StructureModel()
    [model.add_node(name) for name in ['x'+str(i) for i in range(5)]]
    model.add_edge('x1', 'x0')
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


def get_blank_switchboard_causal_graph():
    model = StructureModel()
    [model.add_node(name) for name in ['x'+str(i) for i in range(5)]]
    return model
