from environments.Switchboard import Switchboard, StructuralCausalModel
from stable_baselines3.dqn import DQN
from stable_baselines3.dqn.policies import MlpPolicy
from stable_baselines3.common.env_checker import check_env
import matplotlib.pyplot as plt
import random

swtchbrd = Switchboard()
check = check_env(swtchbrd)

#data collection phase
for i in range(5000):
    swtchbrd.step(-1)
data_actions = [i for i in range(10)]
for i in range(1000):
    a = random.sample(data_actions, k=1)[0]
    swtchbrd.step(a)

# for i in range(1000):
#     rnd_action = swtchbrd.action_space.sample()
#     _, reward, _, _ = swtchbrd.step(rnd_action)
#     swtchbrd.render()
#     print(str(reward)+'\t')

swtchbrd.agent.display_causal_model()
# model = DQN(MlpPolicy, swtchbrd, learning_rate=0.001, policy_kwargs={'net_arch': [44, 50, 45]}, buffer_size=10000)
model = DQN.load('models/exp3.zip', swtchbrd)
model.learn(200000)
model.save('models/exp3_cont')
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
