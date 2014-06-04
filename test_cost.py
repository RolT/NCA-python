# -*- coding: utf-8 -*-

import nca_cost
import numpy as np


N = 300
aux = (np.concatenate([0.5*np.ones((N/2, 1)),
                       np.zeros((N/2, 1)), 1.1*np.ones((N/2, 1))], axis=1))
X = np.concatenate([np.random.rand(N/2, 3),
                    np.random.rand(N/2, 3) + aux])

y = np.concatenate([np.concatenate([np.ones((N/2, 1)), np.zeros((N/2, 1))]),
                    np.concatenate([np.zeros((N/2, 1)), np.ones((N/2, 1))])], axis = 1)
X = X.T
y = y[:, 0]
A = np.array([[1, 0, 0], [0, 1, 0]])
print nca_cost.cost(A, X, y)
print nca_cost.cost_g(A, X, y)
