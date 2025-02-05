import numpy as np
import csdl_alpha as csdl

class Integrators:
    def __init__(self, func, time_interval, init_conditions, num_steps=100):
        self.func = func
        self.t_0, self.t_F = time_interval
        self.init_conditions = init_conditions
        self.num_steps = num_steps
        self.h = (self.t_F - self.t_0) / num_steps
        self.t = csdl.linear_combination(self.t_0, self.t_F, num_steps + 1)
        self.y = csdl.Variable(value=np.zeros((num_steps + 1, init_conditions.shape[0])))
        self.y = self.y.set(csdl.slice[0, :], init_conditions)

    def integrate(self, method='trapezoid', *args):
        methods = {
            'trapezoid': self._trapezoid,
            'rk4': self._rk4,
            'backEuler': self._back_euler
        }
        
        if method not in methods:
            raise ValueError(f"Unknown method '{method}'. Available methods: {list(methods.keys())}")
        
        return methods[method](*args)

    def _trapezoid(self, *args):
        for i in csdl.frange(self.num_steps):
            t_i, y_i = self.t[i], self.y[i]
            dydt_i = self.func(t_i, y_i, *args)
            y_i1_guess = y_i + self.h * dydt_i
            dydt_i1 = self.func(t_i + self.h, y_i1_guess, *args)
            y_i1 = y_i + (self.h / 2) * (dydt_i + dydt_i1)
            self.y = self.y.set(csdl.slice[i + 1, :], y_i1)
        return self.t, self.y

    def _rk4(self, *args):
        for i in csdl.frange(self.num_steps):
            t_i, y_i = self.t[i], self.y[i]
            k1 = self.func(t_i, y_i, *args)
            k2 = self.func(t_i + self.h / 2, y_i + self.h * (k1 / 2), *args)
            k3 = self.func(t_i + self.h / 2, y_i + self.h * (k2 / 2), *args)
            k4 = self.func(t_i + self.h, y_i + self.h * k3, *args)
            y_i1 = y_i + (self.h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
            self.y = self.y.set(csdl.slice[i + 1, :], y_i1)
        return self.t, self.y

    def _back_euler(self, *args):
        for i in csdl.frange(self.num_steps):
            y_prev = self.y[i, :]
            y_next = csdl.Variable(name='y_next', shape=self.y[i+1, :].shape)
            residual = y_next - y_prev - self.h * self.func(self.t[i+1], y_next, *args)
            solver = csdl.nonlinear_solvers.Newton('xue hua piao piao', tolerance=1e-8)
            solver.add_state(y_next, residual, initial_value=self.y[i+1, :])
            solver.run()
            self.y = self.y.set(csdl.slice[i + 1, :], y_next)
        return self.t, self.y

    def _crank_nicolson(self, *args):
        for i in csdl.frange(self.num_steps):
            y_prev = self.y[i, :]
            y_next = csdl.Variable(name='y_next', shape=self.y[i+1, :].shape)
            residual = y_next - y_prev - self.h * self.func(self.t[i+1], y_next, *args)
            residual = y_next - y_prev - 0.5 * self.h * (self.func(self.t[i], y_prev, *args) + self.func(self.t[i+1], y_next, *args))
            solver = csdl.nonlinear_solvers.Newton('bei feng xiao xiao', tolerance=1e-8)
            solver.add_state(y_next, residual, initial_value=self.y[i+1, :])
            solver.run()
            self.y = self.y.set(csdl.slice[i + 1, :], y_next)
        return self.t, self.y
    
    @classmethod
    def solve(cls, func, time_interval, init_conditions, *args, num_steps=100, method='trapezoid'):
        """
        A convenience class method that allows you to run integration in one call.
        
        Parameters:
            func: The function (ODE) to integrate.
            time_interval: A tuple (t0, tF) defining the integration interval.
            init_conditions: The initial conditions as a NumPy array.
            *args: Any additional arguments required by func.
            num_steps: Number of integration steps (default 100).
            method: Integration method to use ('trapezoid', 'rk4', or 'backEuler').
        
        Returns:
            A tuple (t, y) where t is the time array and y the integrated result.
        """
        instance = cls(func, time_interval, init_conditions, num_steps)
        return instance.integrate(method, *args)
