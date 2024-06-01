# Imports
import numpy as np
import math

# Constants
G = 6.67430e-11

# Functions
def GaussianElimination(A:np.array, b:np.array) -> np.array:
    n = len(A)
    for i in range(n):
        # Partial pivoting
        max_row = i
        for j in range(i+1, n):
            if abs(A[j, i]) > abs(A[max_row, i]):
                max_row = j
        A[[i, max_row]] = A[[max_row, i]]
        b[[i, max_row]] = b[[max_row, i]]
        
        # Elimination
        for j in range(i+1, n):
            factor = A[j, i] / A[i, i]
            A[j, i:] -= factor * A[i, i:]
            b[j] -= factor * b[i]
    
    # Back substitution
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        x[i] = (b[i] - np.dot(A[i, i+1:], x[i+1:])) / A[i, i]
    
    return x

# Classes
class ThreeBodyProblem:
    def __init__(self, m:tuple, X1:tuple, X2:tuple, X3:tuple, T, t0=0, n=1000) -> None:
        # Set system parameters
        self.m = m
        self.X1 = X1
        self.X2 = X2
        self.X3 = X3
        self.T = T
        self.t0 = t0
        self.n = n
        self.delta = (T-t0)/n

        # Solve the system
        # Call Runge-Kutta 4-4 method for solving the system
        self._RungeKutta44()
        # Call Spline interpolation to get continuous functions
        self._SplineInterpolation()

    def f(self, X, t) -> function:
        pass

    def _RungeKutta44(self) -> np.array:
        self.X = np.array([])
        self.t = np.array([])
        x = np.array([self.X1, self.X2, self.X3])
        t = self.t0
        for i in range(self.n):
            self.X = np.append(self.X, x)
            self.t = np.append(self.t, t)
            k1 = self.f(self.X, self.t)
            k2 = self.f(self.X + self.delta/2*k1, self.t + self.delta/2)
            k3 = self.f(self.X + self.delta/2*k2, self.t + self.delta/2)
            k4 = self.f(self.X + self.delta*k3, self.t + self.delta)
            x += self.delta/6*(k1 + 2*k2 + 2*k3 + k4)
            t += self.delta

    def _SplineInterpolation(self) -> list[function]:
        self.S_params = []
        for i in range(self.n-1):
            A = np.array()
            b = np.array()
            self.S_params.append(GaussianElimination(A, b))
    
    def S(self, t) -> np.array:
        pass