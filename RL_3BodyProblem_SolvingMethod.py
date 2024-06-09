# Imports
import numpy as np
import math

# Functions
def GaussianElimination(A:np.array, b:np.array) -> np.array:
    n = len(b)
    for i in range(n-1):
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
    def __init__(self, m:tuple, X: np.array, T, t0=0, n=1000, G=6.67430e-11) -> None:
        # Set system parameters
        self.m = m
        self.X = X
        self.T = T
        self.t0 = t0
        self.n = n
        self.delta = (T-t0)/n
        self.G = G

        # Solve the system
        # Call Runge-Kutta 4-4 method for solving the system
        self._RungeKutta44()
        # Call Spline interpolation to get continuous functions
        self._SplineInterpolation()

    def f(self, X, t):
        m1, m2, m3 = self.m
        x_dot = np.zeros(len(X))
        x_dot[0:3] = X[9:12]
        x_dot[3:6] = X[12:15]
        x_dot[6:9] = X[15:18]
        x_dot[9:12] = -self.G*m2*(X[0:3]-X[3:6])/np.linalg.norm(X[0:3]-X[3:6])**3 - self.G*m3*(X[0:3]-X[6:9])/np.linalg.norm(X[0:3]-X[6:9])**3
        x_dot[12:15] = -self.G*m1*(X[3:6]-X[0:3])/np.linalg.norm(X[3:6]-X[0:3])**3 - self.G*m3*(X[3:6]-X[6:9])/np.linalg.norm(X[3:6]-X[6:9])**3
        x_dot[15:18] = -self.G*m1*(X[6:9]-X[0:3])/np.linalg.norm(X[6:9]-X[0:3])**3 - self.G*m2*(X[6:9]-X[3:6])/np.linalg.norm(X[6:9]-X[3:6])**3
        return x_dot

    def _RungeKutta44(self) -> np.array:
        self.t = np.array([])
        x = self.X
        t = self.t0
        self.Xs = x
        for i in range(self.n):
            self.Xs = np.vstack((self.Xs, x))
            self.t = np.append(self.t, t)
            k1 = self.f(x, t)
            k2 = self.f(x + self.delta/2*k1, t + self.delta/2)
            k3 = self.f(x + self.delta/2*k2, t + self.delta/2)
            k4 = self.f(x + self.delta*k3, t + self.delta)
            x += (self.delta/6)*(k1 + 2*k2 + 2*k3 + k4)
            t += self.delta

    def _SplineInterpolation(self) -> list:
        self.S_params = []
        for i in range(1,self.n-1):
            t_diff1 = self.t[i] - self.t[i-1]
            t_diff2 = self.t[i+1] - self.t[i]
            zeros = np.zeros(shape=(len(self.Xs[i]), len(self.Xs[i])))
            base = np.identity(len(self.Xs[i]))
            zero = np.zeros(len(self.Xs[i]))
            A = np.vstack((
                np.hstack((base,zeros,zeros,zeros,zeros,zeros,zeros,zeros)),
                np.hstack((base, t_diff1*base, (t_diff1**2)*base, (t_diff1**3)*base, zeros, zeros, zeros, zeros)),
                np.hstack((zeros,zeros,zeros,zeros,base,zeros,zeros,zeros)),
                np.hstack((zeros,zeros,zeros,zeros,base, t_diff2*base, (t_diff2**2)*base, (t_diff2**3)*base)),
                np.hstack((zeros,base,(2*t_diff1)*base,(3*t_diff1**2)*base,zeros,-base,zeros,zeros)),
                np.hstack((zeros,zeros,2*base,(6*t_diff1)*base,zeros,zeros,-2*base,zeros)),
                np.hstack((zeros,base,zeros,zeros,zeros,zeros,zeros,zeros)),
                np.hstack((zeros,zeros,zeros,zeros,zeros,base,(2*t_diff2)*base,(3*t_diff2**2)*base))
            ))
            b = np.hstack((self.Xs[i-1], self.Xs[i], self.Xs[i], self.Xs[i+1], zero, zero, self.f(self.Xs[i-1],self.t[i-1]), self.f(self.Xs[i+1],self.t[i+1])))
            self.S_params.append(GaussianElimination(A, b)[0:72])
    
    def S(self, t) -> np.array:
        ind = 0
        for idx, num in enumerate(self.t):
            if num <= t:
                ind = idx
        params = self.S_params[ind]
        t_diff = self.t[ind-1] - self.t[ind-2]
        return params[0:18] + params[18:36]*t_diff + params[36:54]*(t_diff)**2 + params[54:72]*(t_diff)**3