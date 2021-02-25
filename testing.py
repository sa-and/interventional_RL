from environments.Switchboard import Switchboard

swtchbrd = Switchboard()
for i in range(100):
    swtchbrd.step(3)
    swtchbrd.render()
