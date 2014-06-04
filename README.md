Neighborhood classification analysis
http://www.cs.toronto.edu/~fritz/absps/nca.pdf

How to use:
create an NCA object and use the fit function to train the data, see docstrings and demo for example of usage.

NB:
Default parameters are really bad, you should at least set maxiter (in options) or tol everytime you use the algorithm.
Metric and dim agruments are redundant, you should only set one