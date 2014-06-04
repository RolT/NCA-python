# -*- coding: utf-8 -*-
"""
Created on Thu May 22 16:50:00 2014

@author: thiolliere
"""
import numpy as np
import scipy.optimize as opt
import nca_cost


class NCA(object):

    def __init__(self, metric=None, dim=None,
                 threshold=None, objective='Mahalanobis', **kwargs):
        """Classification and/or dimensionality reduction with the neighborhood
        component analysis.

        The algorithm apply the softmax function on the transformed space and
        tries to maximise the leave-one-out classification.

        Parameters:
        -----------
        metric : array-like, optional
            The initial distance metric, if not precised, the algorithm will
            use a poor projection of the Mahalanobis distance.
            shape = [dim, n_features] with dim <= n_features being the
            dimension of the output space
        dim : int, optional
            The number of dimensions to keep for dimensionality reduction. If
            not precised, the algorithm wont perform dimensionality reduction.
        threshold : float, otpional
            Threshold for the softmax function, set it higher to discard
            further neighbors.
        objective : string, optional
            The objective function to optimize. The two implemented cost
            functions are for Mahalanobis distance and KL-divergence.
        **kwargs : keyword arguments, optional
            See scipy.optimise.minimize for the list of additional arguments.
            Those arguments include:

            method : string
                The algorithm to use for optimization.
            options : dict
                a dictionary of solver options
            hess, hessp : callable
                Hessian matrix
            bounds : sequence
                Bounds for variables
            constraints : dict or sequence of dict
                Constraints definition
            tol : float
                Tolerance for termination

        Attributes:
        -----------
        metric : array-like
            The trained disctance metric
        """
        self.metric = metric
        self.dim = dim
        self.threshold = threshold
        if objective == 'Mahalanobis':
            self.objective = nca_cost.cost
        elif objective == 'KL-divergence':
            self.objective = nca_cost.cost_g
        self.kwargs = kwargs

    def fit(self, X, y):
        """Fit the model using X as training data and y as target values.

        Parameters:
        -----------
        X : array-like
            Training data, shape = [n_features, n_samples]
        y : array-like
            Target values, shape = [n_samples]
        """
        if self.metric is None:
            if self.dim is None:
                self.metric = np.eye(np.size(X, 1))
                self.dim = np.size(X, 1)
            else:
                self.metric = np.eye(self.dim, np.size(X, 1) - self.dim)

        res = opt.minimize(fun=self.objective,
                           x0=self.metric,
                           args=(X, y, self.threshold),
                           jac=True,
                           **self.kwargs
                           )

        self.metric = np.reshape(res.x,
                                 (np.size(res.x) / np.size(X, 0),
                                  np.size(X, 0)))

    def fit_transform(self, X, y):
        """Fit the model with X and apply the dimensionality reduction on X.

        Parameters:
        -----------
        X : array-like
            Training data, shape = [n_features, n_samples]
        y : array-like
            Target values, shape = [n_samples]

        Returns:
        --------
        X_new : array-like
            shape = [dim, n_samples]
        """
        self.fit(self, X, y)
        return np.dot(self.metric, X)

    def score(self, X, y):
        """Returns the proportion of X correctly classified by the leave-one-
        out classification

        Parameters:
        -----------
        X : array-like
            Training data, shape = [n_features, n_samples]
        y : array-like
            Target values, shape = [n_samples]

        Returns:
        --------
        score : float
            The proportion of X correctly classified
        """
        return 1 - nca_cost.cost(self.metric, X, y)[0]/np.size(X, 1)

    def getParameters(self):
        """Returns a dictionary of the parameters
        """
        return dict(metric=self.metric, dim=self.dim, objective=self.objective,
                    threshold=self.threshold, **self.kwargs)
