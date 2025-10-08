# -*- coding: utf-8 -*-
"""Exercise 3.

Least Square
"""

import numpy as np


def least_squares(y, tx):
    """Calculate the least squares solution.
       returns mse, and optimal weights.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        mse: scalar.

    >>> least_squares(np.array([0.1,0.2]), np.array([[2.3, 3.2], [1., 0.1]]))
    (array([ 0.21212121, -0.12121212]), 8.666684749742561e-33)
    """
    # ***************************************************
    # COPY YOUR CODE FROM EX03 HERE
    # least squares: TODO
    # returns optimal weights, MSE
    # ***************************************************
    w= np.linalg.solve(tx.T @ tx, tx.T @ y)
    e = y - (tx @ w)
    mse= 1/(len(y)) * np.sum(e**2)
    return w, mse


def least_squares_withlamda(y,tx):
    """Calculate the least squares solution with lamda=0.001.
       returns mse, and optimal weights.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        mse: scalar.

  """
    # ***************************************************
    # least squares with lamda=0.001
    # returns optimal weights, MSE
    # ***************************************************
    lambda_=0.00
    w= np.linalg.solve(tx.T @ tx + 2 * len(y) * lambda_ * np.eye(tx.shape[1]), tx.T @ y)
    e = y - (tx @ w)
    mse= 1/(len(y)) * np.sum(e**2)
    return w, mse