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
    def __init__(self, m:tuple, X11:tuple, X12:tuple, X13:tuple, X21:tuple, X22:tuple, X23:tuple, T, t0=0, n=1000) -> None:
        # Set system parameters
        self.m = m
        self.X11 = X11
        self.X12 = X12
        self.X13 = X13
        self.X21 = X21
        self.X22 = X22
        self.X23 = X23
        self.T = T
        self.t0 = t0
        self.n = n
        self.delta = (T-t0)/n

        # Solve the system
        # Call Runge-Kutta 4-4 method for solving the system
        self._RungeKutta44()
        # Call Spline interpolation to get continuous functions
        self._SplineInterpolation()

    def f(self, X, t):
        m1, m2, m3 = self.m
        x_dot = np.zeros(len(X))
        x_dot[0:3] = X[10:13]
        x_dot[3:6] = X[13:16]
        x_dot[6:9] = X[16:19]
        x_dot[10:13] = -G*m2*(X[0:3]-X[3:6])/np.linalg.norm(X[0:3]-X[3:6])**3 - G*m3*(X[0:3]-X[6:9])/np.linalg.norm(X[0:3]-X[6:9])**3
        x_dot[13:16] = -G*m1*(X[3:6]-X[0:3])/np.linalg.norm(X[3:6]-X[0:3])**3 - G*m3*(X[3:6]-X[6:9])/np.linalg.norm(X[3:6]-X[6:9])**3
        x_dot[16:19] = -G*m1*(X[6:9]-X[0:3])/np.linalg.norm(X[6:9]-X[0:3])**3 - G*m2*(X[6:9]-X[3:6])/np.linalg.norm(X[6:9]-X[3:6])**3
        return x_dot

    def _RungeKutta44(self) -> np.array:
        self.Xs = np.array([])
        self.t = np.array([])
        x = np.array([self.X11, self.X12, self.X13, self.X21, self.X22, self.X2])
        t = self.t0
        for i in range(self.n):
            self.Xs = np.append(self.Xs, x)
            self.t = np.append(self.t, t)
            k1 = self.f(x, self.t)
            k2 = self.f(x + self.delta/2*k1, self.t + self.delta/2)
            k3 = self.f(x + self.delta/2*k2, self.t + self.delta/2)
            k4 = self.f(x + self.delta*k3, self.t + self.delta)
            x += (self.delta/6)*(k1 + 2*k2 + 2*k3 + k4)
            t += self.delta

    def _SplineInterpolation(self) -> list:
        self.S_params = []
        for i in range(1,self.n-1):
            t_diff1 = self.t[i] - self.t[i-1]
            t_diff2 = self.t[i+1] - self.t[i]
            zeros = np.zeros(len(self.Xs[i]))
            base = np.ones(len(self.Xs[i]))
            A = np.array([
                [base,zeros,zeros,zeros,zeros,zeros,zeros,zeros],
                [base, t_diff1*base, (t_diff1**2)*base, (t_diff1**3)*base, zeros, zeros, zeros, zeros],
                [zeros,zeros,zeros,zeros,base,zeros,zeros,zeros],
                [zeros,zeros,zeros,zeros,base, t_diff2*base, (t_diff2**2)*base, (t_diff2**3)*base],
                [zeros,base,(2*t_diff1)*base,(3*t_diff1**2)*base,zeros,-base,zeros,zeros],
                [zeros,zeros,2*base,(6*t_diff1)*base,zeros,zeros,-2*base,zeros],
                [zeros,base,zeros,zeros,zeros,zeros,zeros,zeros],
                [zeros,zeros,zeros,zeros,zeros,base,(2*t_diff2)*base,(3*t_diff2**2)*base]
            ])
            b = np.array([self.Xs[i-1], self.Xs[i], self.Xs[i], self.Xs[i+1], zeros, zeros, self.f(self.Xs[i-1],self.t[i-1]), self.f(self.Xs[i+1],self.t[i+1])])
            self.S_params.append(GaussianElimination(A, b)[0:4])
    
    def S(self, t) -> np.array:
        ind = 0
        for idx, num in enumerate(self.t):
            if num >= t:
                ind = idx
        params = self.S_params[ind]
        t_diff = self.t[ind-1] - self.t[ind-2]
        return params[0] + params[1]*t_diff + params[2]*(t_diff)**2 + params[3]*(t_diff)**3