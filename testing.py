from scm import DasguptaSCMGenerator, BoolSCMGenerator
from stable_baselines import ACER
from environments import SCMEnvironment
import stable_baselines.common.vec_env as venv
from agents import DiscreteAgent
from episode_evals import FixedLengthEpisode
from stable_baselines.common.policies import MlpLstmPolicy


def make_sb(scm):
    def f():
        agent = DiscreteAgent(4, env_type='Dasgupta')
        return SCMEnvironment(agent, FixedLengthEpisode(agent, 0.1, 30), scm=scm)
    return f


if __name__ == '__main__':
    # agent = DiscreteAgent(4, env_type='Dasgupta')
    # gen = DasguptaSCMGenerator(5)
    # scms = gen.create_n(4)
    # env = venv.DummyVecEnv([make_sb(scm=e) for e in scms])
    #
    # for i in range(200):
    #     action = [env.action_space.sample() for s in scms]
    #     env.step(action)
    #     i += 1
    #
    # model = ACER(MlpLstmPolicy, env,
    #              policy_kwargs={'net_arch': [40,
    #                                          'lstm',
    #                                          {'pi': [40],
    #                                           'vf': [10]}],
    #                             'n_lstm': 50},
    #
    #              n_steps=10,
    #              n_cpu_tf_sess=8,
    #              replay_ratio=10,
    #              buffer_size=500000,
    #              gamma=0.999
    #              )
    # model.learn(1000000)
    # model.save('experiments/actual/exptesttwo/model')