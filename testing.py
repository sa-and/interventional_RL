from Environments import Switchboard, StructuralCausalModel
from stable_baselines.a2c import A2C
from stable_baselines.common.policies import MlpLstmPolicy
from stable_baselines.common.env_checker import check_env
import matplotlib.pyplot as plt
import random
import stable_baselines.common.vec_env as venv

from Agents import ContinuousSwitchboardAgent


def create_switchboard_a2c():
    agent = ContinuousSwitchboardAgent(5)
    a = Switchboard(agent, fixed_episode_length=True)
    return a


def train_switchboard_a2c(steps: int, workers: int = 8):
    switchboard = venv.DummyVecEnv([create_switchboard_a2c for i in range(workers)])
    # data collection phase
    # for i in range(1000):
    #     switchboard.envs[0].step(-1)
    #
    # data_actions = [i for i in range(10)]
    # for i in range(500):
    #     a = random.sample(data_actions, k=1)[0]
    #     switchboard.envs[0].step(a)

    model = A2C(MlpLstmPolicy, switchboard,
                learning_rate=0.0001,
                policy_kwargs={'net_arch': [50,
                                            'lstm',
                                            {'pi': [15],
                                             'vf': [10]}],
                               'n_lstm': 50},
                epsilon=0.05,
                n_steps=20,
                n_cpu_tf_sess=8)

    model.learn(steps)
    plt.title('A2C fixed length')
    plt.plot(switchboard.envs[0].rewards)
    plt.show()
    return model, switchboard


model, _ = train_switchboard_a2c(200000)

model.save('models/exp9')