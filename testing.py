from environments.Switchboard import Switchboard

swtchbrd = Switchboard()
for i in range(1000):
    rnd_action = swtchbrd.action_space.sample()
    swtchbrd.step(rnd_action)
    swtchbrd.render()
print()
