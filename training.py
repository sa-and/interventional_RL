from Environments import Switchboard
from Agents import SwitchboardAgentDQN
from stable_baselines.a2c import A2C
from stable_baselines.common.policies import MlpLstmPolicy
from stable_baselines.deepq.policies import MlpPolicy
from stable_baselines import DQN
import random
import stable_baselines.common.vec_env as venv


def create_switchboard_a2c():
    agent = SwitchboardAgentDQN(5)
    a = Switchboard(agent)
    return a


def train_switchboard_a2c(steps: int):
    switchboard = venv.DummyVecEnv([create_switchboard_a2c])
    # data collection phase
    for i in range(1000):
        switchboard.envs[0].step(-1)

    data_actions = [i for i in range(10)]
    for i in range(500):
        a = random.sample(data_actions, k=1)[0]
        switchboard.envs[0].step(a)

    model = A2C(MlpLstmPolicy, switchboard,
                learning_rate=0.001,
                policy_kwargs={'net_arch': [24,
                                            30,
                                            'lstm',
                                            {'pi': [45],
                                             'vf': [10]}],
                               'n_lstm': 30},
                epsilon=0.005)

    model.learn(steps)
    return model, switchboard


def train_switchboard_dqn(steps: int):
    agent = SwitchboardAgentDQN(5, state_repeats=3)
    switchboard = Switchboard(agent)

    # data collection phase
    for i in range(1000):
        switchboard.step(-1)
    data_actions = [i for i in range(10)]
    for i in range(500):
        a = random.sample(data_actions, k=1)[0]
        switchboard.step(a)

    model = DQN(MlpPolicy, switchboard,
                buffer_size=10000,
                learning_rate=0.001,
                policy_kwargs={'layers': [64, 50, 45]},
                exploration_final_eps=0.005)

    model.learn(steps)
    return model, switchboard

#check = check_env(swtchbrd)

model, _ = train_switchboard_dqn(1000)
#model = DQN.load('models/exp3.zip', swtchbrd)

model.save('models/exp4')
