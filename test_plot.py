import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from networkx.algorithms import community

# plt.matshow(np.random.random((50,50))*100)
# plt.colorbar()
# plt.show()

n =  400
m = 5
seed = 0
G = nx.barabasi_albert_graph(n, m, seed)