from canonical_system import CanonicalSystem
import numpy as np

alpha = 48
n_bfs = 10
cs = CanonicalSystem(alpha=alpha/2)

c = np.exp(-cs.alpha * np.linspace(0, 1, n_bfs))

h = 1.0 / np.gradient(c)**2
print(h)