# -*- coding: utf-8 -*-
"""Exercise 3.

Least Square
"""

import numpy as np


def least_squares(y, tx):
    """calculate the least squares."""
    # ***************************************************
    # INSERT YOUR CODE HERE
    # least squares: TODO
    # returns mse, and optimal weights
    # ***************************************************
    lambda_=0.001
    w= np.linalg.solve(tx.T @ tx + 2 * len(y) * lambda_ * np.eye(tx.shape[1]), tx.T @ y)
    e = y - (tx @ w)
    mse= 1/(len(y)) * np.sum(e**2)
    return w, mse