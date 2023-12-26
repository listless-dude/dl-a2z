"""
Analytic solution for regression as an optimization problem:
w* = (X'X)^(-1)X'y
Solution is unique when X'X is invertible.
"""
import numpy as np

def LinearRegressionAnalytic(X):
    