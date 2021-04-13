from abc import ABC, abstractmethod
from typing import Tuple, List, Any
from Agents import CausalAgent


class EvalFunc(ABC):
    """
    Interface for evaluating each step for the interventional RL environments
    """
    def __init__(self, agent: CausalAgent, effect_threshold: float):
        super(EvalFunc, self).__init__()
        self.agent = agent
        self.effect_threshold = effect_threshold
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

    def _eval_model(self):
        n_wrong_edges = self.agent.has_wrong_edges(self.effect_threshold)
        n_missing_edges = self.agent.has_missing_edges(self.effect_threshold)
        learned = (n_wrong_edges + n_missing_edges == 0)
        almost_done = (n_wrong_edges + n_missing_edges < 4) and (n_wrong_edges + n_missing_edges >= 2)
        very_almost_done = (n_wrong_edges + n_missing_edges < 2) and (n_wrong_edges + n_missing_edges > 0)
        return learned, very_almost_done, almost_done


class EachStepGoalCheck(EvalFunc):
    # WARNING: This class was not tested extensively as it will not be used (for now)
    def __init__(self, agent: CausalAgent, effect_threshold: float):
        super(EachStepGoalCheck, self).__init__(agent, effect_threshold)

    def evaluate_step(self, action_successful: bool, allow_unsuccessful_actions: bool = True) -> Tuple[bool, float]:
        """
        Ends the episode whenever a graph-altering action is performed
        :param action_successful:
        :param allow_unsuccessful_actions:
        :return:
        """
        self.steps_this_episode += 1

        done = almost_done = very_almost_done = False
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


class FixedLengthEpisode(EvalFunc):
    def __init__(self, agent: CausalAgent, effect_threshold: float, length_per_episode: int):
        super(FixedLengthEpisode, self).__init__(agent, effect_threshold)
        self.length_per_episode = length_per_episode

    def evaluate_step(self, action_successful: bool, allow_unsuccessful_actions: bool = True) -> Tuple[bool, float]:
        """
        Ends the episode after 'length_per_episode' steps. Here we only give a reward for the
        achievement of the goal at the end of each episode. A negative reward for illegal actions
        is still returned.
        :param action_successful:
        :param length_per_episode:
        :param allow_unsuccessful_actions:
        :return:
        """
        self.steps_this_episode += 1

        done = almost_done = very_almost_done = learned = False
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

        return done, reward


class TwoPhaseFixedEpisode(EvalFunc):
    def __init__(self, agent: CausalAgent, effect_threshold: float, information_phase_length: int, task_phase_length: int):
        super(TwoPhaseFixedEpisode, self).__init__(agent, effect_threshold)
        self.information_phase_length = information_phase_length
        self.task_phase_length = task_phase_length

    def evaluate_step(self, action_successful: bool, allow_unsuccessful_actions: bool = True) -> Tuple[bool, float]:
        raise NotImplementedError
