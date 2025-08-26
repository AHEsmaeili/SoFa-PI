import os
import gc
import tqdm
import torch
import pickle
import logging
import warnings
import numpy as np
import networkx as nx
from os.path import join
from scipy import signal
import matplotlib.pyplot as plt
from copy import copy, deepcopy
from multiprocessing import Pool
import collections.abc
import cupy as cp


def is_sequence(x):
    if isinstance(x, collections.abc.Sized):
        return True
    else:
        return False

def get_module(engine="gpu"):
    if engine == "gpu":
        return cp.get_array_module(cp.array([1]))
    else:
        return cp.get_array_module(np.array([1]))

def tohost(x):
    return cp.asnumpy(x)

def todevice(x):
    return cp.asarray(x)

def move_data(x, engine):
    if engine == "cpu":
        return tohost(x)
    elif engine == "gpu":
        return todevice(x)
    
def where_is(x):
    if isinstance(x, np.ndarray):
        return "cpu"
    if x.device == cp.cuda.Device(0):
        return "gpu"
    else:
        return "cpu"

class JR:

    valid_parms = ["weights", "delays", "dt", "t_end", "G", "A", "a", "B", "b", "mu", "SC",
                   "nstart", "t_end", "t_cut", "sigma", "C0", "C1", "C2", "C3",
                    "record_step", "C_vec", "decimate",
                   "vmax", "r", "v0", "data_path", "control", "seed", "ns", "engine",
                   "integration_method", "selected_node_indices"]

    def __init__(self, par: dict = {}):

        self.check_parameters(par)
        self._par = self.get_default_parameters()
        self._par.update(par)

        for item in self._par.items():
            name = item[0]
            value = item[1]
            setattr(self, name, value)

        assert(self.SC is not None), "SC must be provided"
        self._xp = get_module(self.engine)
        self.SC = move_data(self.SC, self.engine)

        self.num_nodes = self.nn = self.SC.shape[0]
        self.num_simulations = self.ns

        if not is_sequence(self.C0):
            self.C0 *= self._xp.ones((self.nn, self.ns))
        if not is_sequence(self.C1):
            self.C1 *= self._xp.ones((self.nn, self.ns))
        if not is_sequence(self.C2):
            self.C2 *= self._xp.ones((self.nn, self.ns))
        if not is_sequence(self.C3):
            self.C3 *= self._xp.ones((self.nn, self.ns))
        
        self.G = move_data(self.G, self.engine)
        self.C0 = move_data(self.C0, self.engine)
        self.C1 = move_data(self.C1, self.engine)
        self.C2 = move_data(self.C2, self.engine)
        self.C3 = move_data(self.C3, self.engine)

    def check_parameters(self, par):
        for key in par.keys():
            if key not in self.valid_parms:
                raise ValueError("Invalid parameter: " + key)

    def __call__(self, ):
        print("Jansen-Rit Model")
        return self._par

    def __str__(self) -> str:
        return "Jansen-Rit Model"

    def get_default_parameters(self) -> dict:
        '''
        Default parameters for the Jansen-Rit model

        Parameters
        ----------
        nn : int
            number of nodes

        Returns
        -------
        params : dict
            default parameters
        '''
        params = {
            "G": 1.0,
            "A": 3.25,
            "B": 22.0,
            "v": 6.0,
            "r": 0.56,
            "v0": 6.0,
            'vmax': 0.005,
            "C0": 1.0 * 135.0,
            "C1": 0.8 * 135.0,
            "C2": 0.25 * 135.0,
            "C3": 0.25 * 135.0,
            "a": 0.1,
            "b": 0.05,
            "mu": 0.24,
            "sigma": 0.01,
            "decimate": 1,
            "dt": 0.01,
            "t_end": 1000.0,
            "t_cut": 500.0,
            "engine": "cpu",
            "integration_method": "euler",
            "ns": 1,
            "SC": None,
        }
        return params

    def set_initial_state(self, nn, ns):

        y0 = self._xp.random.uniform(-1, 1, (nn, ns))
        y1 = self._xp.random.uniform(-500, 500, (nn, ns))
        y2 = self._xp.random.uniform(-50, 50, (nn, ns))
        y3 = self._xp.random.uniform(-6, 6, (nn, ns))
        y4 = self._xp.random.uniform(-20, 20, (nn, ns))
        y5 = self._xp.random.uniform(-500, 500, (nn, ns))
        return self._xp.vstack((y0, y1, y2, y3, y4, y5))

    def S(self, x):
        return self.vmax / (1.0 + self._xp.exp(self.r*(self.v0-x)))

    def f_sys(self, x0, t):

        nn = self.nn
        ns = self.ns
        mu = self.mu
        G = self.G
        C0 = self.C0
        C1 = self.C1
        C2 = self.C2
        C3 = self.C3
        A = self.A
        B = self.B
        a = self.a
        b = self.b
        Aa = A * a
        Bb = B * b
        bb = b * b
        aa = a * a
        SC = self.SC
        _xp = self._xp
        S = self.S

        x = x0[:nn, :]
        y = x0[nn:2*nn, :]
        z = x0[2*nn:3*nn, :]
        xp = x0[3*nn:4*nn, :]
        yp = x0[4*nn:5*nn, :]
        zp = x0[5*nn:6*nn, :]

        dx = _xp.zeros((6*nn, ns))
        couplings = S(SC.dot(y-z))

        dx[0:nn, :] = xp
        dx[nn:2*nn, :] = yp
        dx[2*nn:3*nn, :] = zp
        dx[3*nn:4*nn, :] = Aa * S(y-z) - 2 * a * xp - aa * x
        dx[4*nn:5*nn, :] = (Aa * (mu + C1 * S(C0 * x) + G *
                            couplings) - 2 * a * yp - aa * y)
        dx[5*nn:6*nn, :] = Bb * C3 * S(C2 * x) - 2 * b * zp - bb * z

        return dx

    def euler(self, x0, t):

        _xp = self._xp
        nn = self.nn
        dt = self.dt
        sqrt_dt = np.sqrt(dt)
        sigma = self.sigma

        x0 = x0 + dt * self.f_sys(x0, t)
        x0[4*nn:5*nn, :] += sqrt_dt * sigma * \
            _xp.random.normal(0, 1, size=(nn, self.ns))

        return x0

    def heun(self, x0, t):

        _xp = self._xp
        nn = self.nn
        ns = self.ns
        dt = self.dt
        sqrt_dt = np.sqrt(dt)
        sigma = self.sigma

        k1 = self.f_sys(x0, t) * dt
        x1 = x0 + k1
        x1[4*nn:5*nn, :] += sqrt_dt * sigma * _xp.random.randn(nn, ns)
        k2 = self.f_sys(x1, t + dt) * dt
        x0 = x0 + (k1 + k2) / 2.0
        x0[4*nn:5*nn, :] += sqrt_dt * sigma * _xp.random.randn(nn, ns)

        return x0
    
    def set_C_i(self, item, value):
        _C = getattr(self, item)
        _C = move_data(_C, self.engine)

    def simulate(self, x0=None, par: dict = {}):

        for item in par.keys():
            if is_sequence(par[item]):
                setattr(self, item, move_data(par[item], self.engine))
            else:
                setattr(self, item, par[item])
            
        if x0 is None:
            x = self.set_initial_state(self.nn, self.ns)
        else:
            x = deepcopy(x0)
            x = move_data(x, self.engine)

        self.integrator = self.euler if self.integration_method == 'euler' else self.heun
        dt = self.dt
        _xp = self._xp
        nn = self.nn
        ns = self.ns
        decimate = self.decimate
        t_end = self.t_end
        t_cut = self.t_cut
        assert(t_cut < t_end), "t_cut must be smaller than t_end"

        tspan = _xp.arange(0, t_end, dt)
        idx_cut = int(_xp.where(tspan >= t_cut)[0][0])

        x = self.set_initial_state(nn, ns)
        n_step = int((len(tspan) - idx_cut) / decimate)
        y = np.zeros((n_step, nn, ns), dtype="f") # store in host
        counter = 0

        for i in tqdm.trange(len(tspan)):

            x = self.integrator(x, tspan[i])
            if self.engine == 'gpu':
                x_ = x.get().astype('f')

            else:
                x_ = x.astype('f')

            if (i >= idx_cut) and (i % decimate == 0):
                y[counter, :, :] = x_[nn:2*nn, :] - x_[2*nn:3*nn, :]
                counter += 1

        t = tspan[tspan >= t_cut][::decimate]
        t = t.get() if self.engine == 'gpu' else t

        return {"t": t, "x": y}

