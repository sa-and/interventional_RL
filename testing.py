from environments.Switchboard import Switchboard, StructuralCausalModel
from stable_baselines3.dqn import DQN
from stable_baselines3.dqn.policies import MlpPolicy
from stable_baselines3.common.env_checker import check_env
import matplotlib.pyplot as plt
import random as rnd

exo_vars = [('U'+str(i), True, rnd.choice, {'seq': [True, False]}) for i in range(5)]
endo_vars = [('X0', False, lambda x4, u0: x4 or u0, {'x4': 'X4', 'u0': 'U0'}),
             ('X1', False, lambda x0, x2, x4, u1: x0 or x2 or x4 or u1, {'x0': 'X0', 'x2': 'X2', 'x4': 'X4', 'u1': 'U1'}),
             ('X2', False, lambda x4, u2: x4 or u2, {'x4': 'X4', 'u2': 'U2'}),
             ('X3', False, lambda x2, u3: x2 or u3, {'x2': 'X2', 'u3': 'U3'}),
             ('X4', False, lambda u4: u4, {'u4': 'U4'})]
mod = StructuralCausalModel()
mod.add_exogenous_vars(exo_vars)
mod.add_endogenous_vars(endo_vars)
for i in range(10):
    print(mod.get_next_instantiation()[0])

intv = mod.get_intervened_scm([('X4', lambda: False)])
for i in range(10):
    print(intv.get_next_instantiation()[0])
swtchbrd = Switchboard()
check = check_env(swtchbrd)
#pure observation phase
for i in range(500):
    swtchbrd.step(-1)
    swtchbrd.render()

#for i in range(1000):
    # rnd_action = swtchbrd.action_space.sample()
    # _, reward, _, _ = swtchbrd.step(rnd_action)
    # swtchbrd.render()
    # print(str(reward)+'\t')

# swtchbrd.agent.display_causal_model()
# model = DQN(MlpPolicy, swtchbrd, learning_rate=0.001, policy_kwargs={'net_arch': [24, 30, 45]}, buffer_size=10000)
# model.learn(200)
# swtchbrd.agent.display_causal_model()
# plt.plot(swtchbrd.rewards)
# plt.show()
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
