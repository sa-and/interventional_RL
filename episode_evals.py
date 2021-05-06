from abc import ABC, abstractmethod
from typing import Tuple
from agents import CausalAgent
import networkx as nx
from scm import CausalGraphGenerator


def directed_shd(predicted: nx.DiGraph, target: nx.DiGraph) -> int:
    assert len(predicted.nodes) == len(target.nodes), 'Graphs need to have the same amount of nodes'

    # this corresponds to the SHD (structural Hamming distance or SHD (Tsamardinos et al., 2006;
    # with the difference that we consider undirected edges as bidirected edges and flip is considered
    # as 2 errors instead of 1
    differences = 0
    for node in predicted.adj:
        # check which edges are too much in the predicted graph
        for parent in predicted.adj[node]:
            if not parent.upper() in target.adj[node.upper()]:
                differences += 1
        # check which edges are missing in the predicted graph
        for parent in target.adj[node.upper()]:
            if not parent.lower() in predicted.adj[node]:
                differences += 1

    return differences


class EvalFunc(ABC):
    """Interface for evaluation function for the RL learning"""
    effect_threshold: float
    steps_this_episode: int

    def __init__(self, agent: CausalAgent):
        super(EvalFunc, self).__init__()
        self.agent = agent
        self.steps_this_episode = 0

    @abstractmethod
    def evaluate_step(self, action_successful: bool, allow_unsuccessful_actions: bool = True) -> Tuple[bool, float]:
        """
        Calculates for each step whether the episode is done and what the reward for that step is

        :param action_successful: whether the action which was taken was executed successfully
        :param allow_unsuccessful_actions: Whether unsuccessful actions are allowed
        :return: Whether the episode is done and the reward of the current step
        """
        raise NotImplementedError

    @abstractmethod
    def _eval_model(self):
        """Defines how the model is evaluated"""
        raise NotImplementedError


# ----------------------------------------------------------------------------
# Functions evaluating the model based on the structural difference to the target graph
class StructureEvalFunc(EvalFunc):
    def __init__(self, agent: CausalAgent, graph: nx.DiGraph):
        super(StructureEvalFunc, self).__init__(agent)
        self.compare_graph = graph

    def set_compare_graph(self, graph: nx.DiGraph):
        self.compare_graph = graph

    def _eval_model(self):
        diff = directed_shd(self.agent.causal_model, self.compare_graph)**3
        return -diff


class FixedLengthStructEpisode(StructureEvalFunc):
    def __init__(self, agent: CausalAgent, graph: nx.DiGraph, ep_length: int):
        super(FixedLengthStructEpisode, self).__init__(agent, graph)
        self.episode_length = ep_length

    def evaluate_step(self, action_successful: bool, allow_unsuccessful_actions: bool = True) -> Tuple[bool, float]:
        self.steps_this_episode += 1

        done = False
        # Evaluate when the episode length is reached
        if self.steps_this_episode >= self.episode_length:
            done = True
            self.steps_this_episode = 0
            reward = self._eval_model()
            if reward == 0:
                reward = 30

        elif not action_successful and not allow_unsuccessful_actions:  # illegal action was taken
            reward = -1
        else:
            reward = 0

        if done:
            print('episode reward', reward)

        return done, reward

# ----------------------------------------------------------------------------
# Functions comparing the causal structure of the model to the interventional data that has been collected
class InferenceEvalFunc(EvalFunc):
    """
    Interface for evaluating each step for the interventional RL environments based on the inference
    power the estimated graph has w.r.t. the collected interventional data
    """
    agent: CausalAgent

    def __init__(self, agent: CausalAgent, effect_threshold: float):
        super(InferenceEvalFunc, self).__init__(agent)
        self.effect_threshold = effect_threshold

    def _eval_model(self):
        n_wrong_edges = self.agent.has_wrong_edges(self.effect_threshold)
        n_missing_edges = self.agent.has_missing_edges(self.effect_threshold)
        learned = (n_wrong_edges + n_missing_edges == 0)
        almost_done = (n_wrong_edges + n_missing_edges < 4) and (n_wrong_edges + n_missing_edges >= 2)
        very_almost_done = (n_wrong_edges + n_missing_edges < 2) and (n_wrong_edges + n_missing_edges > 0)
        return learned, very_almost_done, almost_done


class EachStepInfGoalCheck(InferenceEvalFunc):
    # WARNING: Outdated. This class was not tested extensively as it will not be used (for now)
    def __init__(self, agent: CausalAgent, effect_threshold: float):
        super(EachStepInfGoalCheck, self).__init__(agent, effect_threshold)

    def evaluate_step(self, action_successful: bool, allow_unsuccessful_actions: bool = True) -> Tuple[bool, float]:
        """
        Ends the episode whenever a graph-altering action is performed
        :param action_successful:
        :param allow_unsuccessful_actions:
        :return:
        """
        self.steps_this_episode += 1

        done = almost_done = very_almost_done = False
        # Evaluate every time a structure action is taken
        if self.agent.current_action[0] == 1:  # only check if the model actually changed.
            done, very_almost_done, almost_done = self._eval_model()

        # compute reward
        if not action_successful and not allow_unsuccessful_actions:  # illegal action was taken
            reward = -1
        elif almost_done:
            reward = 2
        elif very_almost_done:
            reward = 5
        elif done:  # the graph has been learned
            reward = 30
            self.steps_this_episode = 0
        else:  # intervention, non-intervention, graph-changing
            reward = 0
        return done, reward


class FixedLengthInfEpisode(InferenceEvalFunc):
    length_per_episode: int

    def __init__(self, agent: CausalAgent, effect_threshold: float, length_per_episode: int):
        super(FixedLengthInfEpisode, self).__init__(agent, effect_threshold)
        self.length_per_episode = length_per_episode

    def evaluate_step(self, action_successful: bool, allow_unsuccessful_actions: bool = True) -> Tuple[bool, float]:
        """
        Ends the episode after 'length_per_episode' steps. Here we only give a reward for the
        achievement of the goal at the end of each episode. A negative reward for illegal actions
        is still returned.
        :param action_successful:
        :param allow_unsuccessful_actions:
        :return:
        """
        self.steps_this_episode += 1

        done = almost_done = very_almost_done = learned = False
        # Evaluate when the episode length is reached
        if self.steps_this_episode >= self.length_per_episode:
            done = True
            self.steps_this_episode = 0
            learned, very_almost_done, almost_done = self._eval_model()

        if not action_successful and not allow_unsuccessful_actions:  # illegal action was taken
            reward = -1
        elif almost_done:
            reward = 2
        elif very_almost_done:
            reward = 5
        elif learned:
            reward = 30
        else:
            reward = 0

        if done:
            print('episode reward', reward)

        return done, reward


class TwoPhaseFixedInfEpisode(InferenceEvalFunc):
    information_phase_length: int
    task_phase_length: int

    def __init__(self, agent: CausalAgent,
                 effect_threshold: float,
                 information_phase_length: int,
                 task_phase_length: int):
        super(TwoPhaseFixedInfEpisode, self).__init__(agent, effect_threshold)
        self.information_phase_length = information_phase_length
        self.task_phase_length = task_phase_length
        self.rewards = []

    def evaluate_step(self, action_successful: bool, allow_unsuccessful_actions: bool = True) -> Tuple[bool, float]:
        """
        Computes rewards analogous to Dasgupta. Where the first is a information phase in which only observations or
        interventions should be taken and a task phase in which only structure actions should be taken. If this is
        violated a reward of -1 is given so that the model learns to avoid this. The rest of the evaluation is as in
        the fixed episode length case where at the end of each episode the model is evaluated and a reward determined
        according to the fitness of the model

        :param action_successful:
        :param allow_unsuccessful_actions:
        :return:
        """
        self.steps_this_episode += 1

        done = almost_done = very_almost_done = learned = False
        # evaluate when the episode length is reached
        if self.steps_this_episode >= (self.information_phase_length + self.task_phase_length):
            done = True
            self.steps_this_episode = 0
            learned, very_almost_done, almost_done = self._eval_model()

        if not done:
            # if structure action is taken in information phase
            if (self.agent.current_action[0] == 1 or self.agent.current_action[0] == None) and self.steps_this_episode <= self.information_phase_length:
                reward = -0.5

            # if listening action is taken in task phase
            elif self.agent.current_action[0] == 0 and self.steps_this_episode > self.information_phase_length:
                reward = -0.5

            # if an illegal action was taken
            elif not action_successful and not allow_unsuccessful_actions:
                reward = -1

            else:
                reward = 0

        else:
            # rewards for building correct graphs. Will only be True if episode length is reached
            # if almost_done:
            #     reward = 2
            # elif very_almost_done:
            #     reward = 5
            if learned:
                reward = 30
            else:
                reward = 0

        self.rewards.append(reward)
        if done:
            print('episode reward:', sum(self.rewards), ' eval reward:', reward)
            self.rewards = []

        return done, reward


class NoEval(InferenceEvalFunc):
    """Does nothing. Used when applying the policy so there are no prints and rewards"""
    def __init__(self):
        super(NoEval, self).__init__(None, None)

    def evaluate_step(self, action_successful: bool, allow_unsuccessful_actions: bool = True) -> Tuple[bool, float]:
        return False, 0.0
