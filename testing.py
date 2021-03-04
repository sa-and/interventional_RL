from environments.Switchboard import Switchboard, StructuralCausalModel
from stable_baselines3.dqn import DQN
from stable_baselines3.dqn.policies import MlpPolicy
from stable_baselines3.common.env_checker import check_env
import matplotlib.pyplot as plt
import random as rnd

swtchbrd = Switchboard()
check = check_env(swtchbrd)
#pure observation phase
for i in range(500):
    swtchbrd.step(-1)
    swtchbrd.render()

# for i in range(1000):
#     rnd_action = swtchbrd.action_space.sample()
#     _, reward, _, _ = swtchbrd.step(rnd_action)
#     swtchbrd.render()
#     print(str(reward)+'\t')

swtchbrd.agent.display_causal_model()
model = DQN(MlpPolicy, swtchbrd, learning_rate=0.001, policy_kwargs={'net_arch': [24, 30, 45]}, buffer_size=10000)
model.learn(2000)
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
