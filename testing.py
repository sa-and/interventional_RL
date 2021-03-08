from environments.Switchboard import Switchboard, StructuralCausalModel
from stable_baselines.a2c import A2C
from stable_baselines.common.policies import MlpLstmPolicy
from stable_baselines.common.env_checker import check_env
import matplotlib.pyplot as plt
import random
import stable_baselines.common.vec_env as venv

def create_switchboard():
    a = Switchboard()
    return a

swtchbrd = venv.DummyVecEnv([create_switchboard])
#check = check_env(swtchbrd)

# data collection phase
for i in range(5000):
    swtchbrd.envs[0].step(-1)
data_actions = [i for i in range(10)]
for i in range(2000):
    a = random.sample(data_actions, k=1)[0]
    swtchbrd.envs[0].step(a)

# for i in range(1000):
#     rnd_action = swtchbrd.action_space.sample()
#     _, reward, _, _ = swtchbrd.step(rnd_action)
#     swtchbrd.render()
#     print(str(reward)+'\t')

swtchbrd.envs[0].agent.display_causal_model()
model = A2C(MlpLstmPolicy,
            swtchbrd,
            learning_rate=0.001,
            policy_kwargs={'net_arch': [24,
                                        'lstm',
                                        {'pi': [45],
                                         'vf': [10]}],
                           'n_lstm': 30},
            epsilon=0.005)
#model = DQN.load('models/exp3.zip', swtchbrd)

model.learn(200000)
pred = model.predict([swtchbrd.envs[0].get_obs_vector()])
print(model.predict([swtchbrd.envs[0].get_obs_vector()]))
model.save('models/exp4')
swtchbrd.agent.display_causal_model()
plt.plot(swtchbrd.rewards)
plt.show()
# print(swtchbrd.agent.get_est_postint_distrib('x0', (2, True)))
# print()
# print(swtchbrd.agent.get_est_postint_distrib('x0', (2, False)))
# print()
# print(swtchbrd.agent.get_est_avg_causal_effect('x0', (2, True), (2, False)))
# print()
# print(swtchbrd.agent.get_est_cond_distr('x0', ('x2', True)))
# print()
# print(swtchbrd.agent.get_est_cond_distr('x0', ('x2', False)))
# print(swtchbrd.agent.evaluate_causal_model())
