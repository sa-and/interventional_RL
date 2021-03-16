from Environments import Switchboard
from Agents import DiscreteSwitchboardAgent, ContinuousSwitchboardAgent
from stable_baselines.a2c import A2C
from stable_baselines.common.policies import MlpLstmPolicy
from stable_baselines.deepq.policies import MlpPolicy as dqnMlpPolicy
from stable_baselines import DQN
import random
import stable_baselines.common.vec_env as venv
import matplotlib.pyplot as plt
from stable_baselines.ddpg.policies import MlpPolicy as ddpgMlpPolicy
from stable_baselines.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec
from stable_baselines import DDPG
from Agents import get_blank_switchboard_causal_graph


def create_switchboard_a2c_fixed():
    agent = ContinuousSwitchboardAgent(5)
    a = Switchboard(agent, fixed_episode_length=True)
    return a


def create_switchboard_a2c_dynamic():
    agent = ContinuousSwitchboardAgent(5)
    a = Switchboard(agent, fixed_episode_length=False)
    return a


def train_switchboard_a2c(steps: int, workers: int = 8, fixed_length: bool = False):
    if fixed_length:
        switchboard = venv.DummyVecEnv([create_switchboard_a2c_fixed for i in range(workers)])
    else:
        switchboard = venv.DummyVecEnv([create_switchboard_a2c_dynamic for i in range(workers)])

    # data collection phase
    for i in range(500):
        a = switchboard.envs[0].action_space.sample()
        a = [a for i in range(workers)]
        switchboard.step(a)
    print('data collection phase done\n\n\n\n\n\n\n\n\n\n')

    model = A2C(MlpLstmPolicy, switchboard,
                learning_rate=0.0001,
                policy_kwargs={'net_arch': [50,
                                            'lstm',
                                            {'pi': [15],
                                             'vf': [10]}],
                               'n_lstm': 50},
                epsilon=0.05,
                n_steps=5,
                n_cpu_tf_sess=8)

    model.learn(steps)
    title = 'A2C, fixed = ' + str(fixed_length)
    plt.title(title)
    plt.plot(switchboard.envs[0].rewards)
    plt.show()
    return model, switchboard


def train_switchboard_dqn(steps: int, fixed_length):
    agent = DiscreteSwitchboardAgent(5, state_repeats=3, causal_graph=get_blank_switchboard_causal_graph())
    switchboard = Switchboard(agent, fixed_episode_length=fixed_length)

    # data collection phase
    for i in range(1000):
        switchboard.step(-1)
    data_actions = [i for i in range(10)]
    for i in range(500):
        a = random.sample(data_actions, k=1)[0]
        switchboard.step(a)

    model = DQN(dqnMlpPolicy, switchboard,
                buffer_size=100000,
                learning_rate=0.001,
                policy_kwargs={'layers': [90, 50, 45]},
                exploration_final_eps=0.05,
                batch_size=64,
                n_cpu_tf_sess=8,
                prioritized_replay=True)

    model.learn(steps)

    title = 'DQN, fixed = ' + str(fixed_length)
    plt.title(title)
    plt.plot(switchboard.rewards)
    plt.show()
    return model, switchboard


def train_switchboard_ddpg(steps: int):
    agent = ContinuousSwitchboardAgent(5, state_repeats=3)
    switchboard = Switchboard(agent)

    n_actions = switchboard.action_space.shape[-1]
    param_noise = None
    action_noise = NormalActionNoise(0, 0.1)

    model = DDPG(ddpgMlpPolicy,
                 switchboard,
                 param_noise=param_noise,
                 action_noise=action_noise,
                 policy_kwargs={'layers': [90, 40, 10]},
                 n_cpu_tf_sess=8,
                 buffer_size=50000)
    model.learn(steps)

    plt.title('ddpg')
    plt.plot(switchboard.rewards)
    plt.show()
    return model, switchboard

#check = check_env(swtchbrd)

model, board = train_switchboard_dqn(500000, fixed_length=False)
#model = DQN.load('models/exp3.zip', swtchbrd)

model.save('models/exp10')
