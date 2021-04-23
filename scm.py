from typing import List, Callable, Tuple, Any
from scipy.special import comb
import copy
import random
import networkx as nx
from tqdm import tqdm
import pickle
from abc import ABC, abstractmethod


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
            if type(self.exogenous_vars[key]) == bool:
                self.exogenous_vars[key] = random.choice([True, False])
            else:
                # TODO: understand why parametrized version below produces always the same sequence for bool
                dist = self.exogenous_distributions[key]
                res = dist[0](**dist[1])
                self.exogenous_vars[key] = res

        # update endogenous vars
        structure_model = CausalGraphGenerator.create_graph_from_scm(self)
        node_order = [n for n in nx.topological_sort(structure_model)]
        # propagate causal effects along the topological ordering
        for node in node_order:
            # get the values for the parameters needed in the functions
            params = {}
            for param in self.functions[node][1]:  # parameters of functions
                if self.functions[node][1][param] in self.endogenous_vars.keys():
                    params[param] = self.endogenous_vars[self.functions[node][1][param]]
                else:
                    params[param] = self.exogenous_vars[self.functions[node][1][param]]

            # Update variable according to its function and parameters
            self.endogenous_vars[node] = self.functions[node][0](**params)

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


class CausalGraphGenerator:
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
                if not len(exo_copy) == 0:
                    self.potential_causes[v] = self.endo_vars + [exo_copy.pop()]
                else:
                    self.potential_causes[v] = self.endo_vars + []
            self.potential_causes[v].remove(v)

        self.fully_connected_graph = self._make_fully_connected_dag()

    def _make_fully_connected_dag(self) -> nx.DiGraph:
        """
        Creates and returns a fully connected graph. In this graph the exogenous variables have only outgoing edges.
        """
        graph = nx.DiGraph()
        [graph.add_node(u) for u in self.exo_vars]
        [graph.add_node(v) for v in self.endo_vars]
        for n, cs in self.potential_causes.items():
            [graph.add_edge(c, n) for c in cs]
        return graph

    def create_random_graph(self) -> Tuple[nx.DiGraph, set]:
        """
        Creates and returns a random StructualCausalModel by first creating a fully connected graph and then
        randomly deleting one edge after the other until it is acyclic.

        :return: the random SCM and the set of edges that have been removed
        """
        # generate fully connected graph
        graph = copy.deepcopy(self.fully_connected_graph)

        # delete random edges until acyclic
        removed_edges = set()
        while not nx.is_directed_acyclic_graph(graph):
            random_edge = random.sample(graph.edges, 1)[0]
            removed_edges.add(random_edge)
            graph.remove_edge(random_edge[0], random_edge[1])

        # create scm
        return graph, removed_edges

    @staticmethod
    def create_graph_from_scm(scm: StructuralCausalModel) -> nx.DiGraph:
        graph = nx.DiGraph()

        # create nodes
        [graph.add_node(var) for var in scm.endogenous_vars]

        for var in scm.functions:
            for parent in scm.functions[var][1]:
                if parent in scm.endogenous_vars:
                    graph.add_edge(parent, var)

        return graph

    @staticmethod
    def max_n_dags(n_vertices: int) -> int:
        """
        Computes the maximal number of different DAGs over n_vertices nodes. Implemented as in Robinson (1973)

        :param n_vertices:
        :return: max number of dags
        """
        if n_vertices < 0:
            return 0
        elif n_vertices == 0:
            return 1
        else:
            summ = 0
            for k in range(1, n_vertices + 1):
                summ += (-1) ** (k - 1) * comb(n_vertices, k) * 2 ** (
                        k * (n_vertices - k)) * CausalGraphGenerator.max_n_dags(n_vertices - k)
            return int(summ)


class SCMGenerator(ABC):
    """Interface for SCM generators"""
    def __init__(self, n_endo: int, n_exo: int, allow_exo_confounders: bool = False):
        self.graph_generator = CausalGraphGenerator(n_endo, n_exo, allow_exo_confounders)

    def create_random(self) -> Tuple[StructuralCausalModel, set]:
        """
        Creates and returns a random StructualCausalModel by first creating a fully connected graph and then
        randomly deleting one edge after the other until it is acyclic.

        :return: the random SCM and the set of edges that have been removed
        """
        graph, removed_edges = self.graph_generator.create_random_graph()
        return self.create_scm_from_graph(graph), removed_edges

    def create_n(self, n: int) -> List[StructuralCausalModel]:
        """
        Create n different random SCMs
        :param n: how many SCMs to create
        :return: list of the SCMs
        """
        # check whether more DAGs are to be created then combinatorically possible. Only do this if n is not
        # too big becaus computation takes forever for n > 20 and for such values ther exist over 2.3e+72
        # different graphs either way
        if n > CausalGraphGenerator.max_n_dags(len(self.graph_generator.endo_vars)):
            n = CausalGraphGenerator.max_n_dags(len(self.graph_generator.endo_vars))

        scms = []
        rem_edges_list = []
        resampled = 0
        print('Creating SCMs...')
        pbar = tqdm(total=n - 1)
        while len(scms) < n - 1:
            scm, rem_edges = self.create_random()
            if any([rem_edges == other for other in rem_edges_list]):
                resampled += 1
                continue
            else:
                scms.append(scm)
                rem_edges_list.append(rem_edges)
                pbar.update(1)
        pbar.close()
        print(resampled, 'models resampled')

        return scms

    @staticmethod
    @abstractmethod
    def create_scm_from_graph(graph: nx.DiGraph) -> StructuralCausalModel:
        """Defines how an SCM is created from a given causal graph"""
        pass

    @staticmethod
    @abstractmethod
    def _make_f(parents: List[str]):
        """Method that defines how the parents of a node are treated w.r.t. the causal relationship of
        a variable"""
        pass


class BoolSCMGenerator(SCMGenerator):
    '''
    Class to help creating SCMs with boolean variables and relationships as in the switchboard environment
    '''

    def __init__(self, n_endo: int, n_exo: int, allow_exo_confounders: bool = False):
        super(BoolSCMGenerator, self).__init__(n_endo, n_exo, allow_exo_confounders)

    @staticmethod
    def create_scm_from_graph(graph: nx.DiGraph) -> StructuralCausalModel:
        """
        Takes a networkx graph and builds a scm according to it's hierarchical structure. The functions causing the
        values of the variables are fixed as a boolean or over their causes.

        :param graph: networkx graph
        :return: SCM according to the graph with boolean or functions
        """
        scm = StructuralCausalModel()
        for n in graph.nodes:
            parents = [p for p in graph.predecessors(n)]
            if n[0] == 'X':
                scm.add_endogenous_var(n, False, BoolSCMGenerator._make_f(parents), {p: p for p in parents})
            else:
                scm.add_exogenous_var(n, False, random.choice, {'seq': [True, False]})
        return scm

    @staticmethod
    def _make_f(parents: List[str]):
        """
        Creates a boolean OR function over the causes/parents of a variable.

        :param parents: causes of the variable
        :return: callable
        """

        def f(**kwargs):
            res = False
            for p in parents:
                res = res or kwargs[p]
            return res

        return f

    @staticmethod
    def make_switchboard_scm_with_context():
        SCM = StructuralCausalModel()
        SCM.add_exogenous_vars([('U' + str(i), True, random.choice, {'seq': [True, False]}) for i in range(5)])
        SCM.add_endogenous_vars(
            [
                ('X0', False, lambda x4, u0: x4 or u0, {'x4': 'X4', 'u0': 'U0'}),
                (
                    'X1', False, lambda x0, x2, x4, u1: x0 or x2 or x4 or u1,
                    {'x0': 'X0', 'x2': 'X2', 'x4': 'X4', 'u1': 'U1'}),
                ('X2', False, lambda x4, u2: x4 or u2, {'x4': 'X4', 'u2': 'U2'}),
                ('X3', False, lambda x2, u3: x2 or u3, {'x2': 'X2', 'u3': 'U3'}),
                ('X4', False, lambda u4: u4, {'u4': 'U4'})
            ])

        return SCM

    @staticmethod
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

    @staticmethod
    def make_obs_equ_3var_envs():
        # two 3-var networks with observationally identical distributions
        scm1 = StructuralCausalModel()
        scm1.add_exogenous_vars([('U' + str(i), True, random.choice, {'seq': [True, False]}) for i in range(3)])
        scm1.add_endogenous_vars(
            [('X0', False, lambda u0: u0, {'u0': 'U0'}),
             ('X1', False, lambda x0, x2: x0 or x2, {'x0': 'X0', 'x2': 'X2'}),
             ('X2', False, lambda x0: x0, {'x0': 'X0'})])
        scm2 = StructuralCausalModel()
        scm2.add_exogenous_vars([('U' + str(i), True, random.choice, {'seq': [True, False]}) for i in range(3)])
        scm2.add_endogenous_vars(
            [('X0', False, lambda u0: u0, {'u0': 'U0'}),
             ('X1', False, lambda x0: x0, {'x0': 'X0'}),
             ('X2', False, lambda x0: x0, {'x0': 'X0'})])
        return scm1, scm2

    @staticmethod
    def load_dataset(path: str) -> List[StructuralCausalModel]:
        with open(path, 'rb') as f:
            dic = pickle.load(f)
        return dic


class DasguptaSCMGenerator(SCMGenerator):
    """To generate SCMs as described in Dasgupta et al. (2019), Causal Reasoning from Meta-reinforcement Learning"""
    def __init__(self, n_vars: int):
        super(DasguptaSCMGenerator, self).__init__(n_vars, 0, False)

    def create_scm_from_graph(self, graph: nx.DiGraph) -> StructuralCausalModel:
        roots = []
        for node in graph:
            n_pred = sum([1 for _ in graph.predecessors(node)])
            if n_pred == 0:
                roots.append(node)
        scm = StructuralCausalModel()

        # add unobserved var
        exo = roots[0]
        scm.add_exogenous_var(exo, 0.0, random.gauss, {'mu': 0.0, 'sigma': 0.1})

        # add roots with distribution N(0, 0.1)
        for root in roots[1:]:
            scm.add_endogenous_var(root, 0.0, lambda: random.gauss(0.0, 0.1), {})

        # add all others with N(mu=sum(w_{ij}Xj), sigma=0.1) for Xj in PA(Xi) and w_{ij} in [-1, 1]
        for node in graph:
            if node not in roots:
                parents = [p for p in graph.predecessors(node)]
                scm.add_endogenous_var(node, 0.0, DasguptaSCMGenerator._make_f(parents), {p: p for p in parents})

        return scm

    @staticmethod
    def _make_f(parents: List[str]):
        def f(**kwargs):
            mu = 0.0
            for p in parents:
                mu += random.choice([-1, 1]) * kwargs[p]
            return random.gauss(mu, 0.1)
        return f