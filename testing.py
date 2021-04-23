
from scm import DasguptaSCMGenerator, BoolSCMGenerator

gen = DasguptaSCMGenerator(5)
scm = gen.create_scm_from_graph(gen.graph_generator.create_random_graph()[0])
for i in range(10):
    print(scm.get_next_instantiation())
scms = gen.create_n(5000)
print()
