import numpy as np

A = np.array([[3,0], [1,2]]).T
ev, ea = np.linalg.eig(A)
ev
ea
