import os
import tqdm
import torch
import numpy as np
from copy import copy
from os.path import join
from JansenRitNet import JRN as _JRN


class JRN:
    valid_params = [
        "fix_seed", "seed", "G", "adj", "A", "B", "a", "b",
        "noise_mu", "noise_std", "vmax", "v0", "r",
        "C0", "C1", "C2", "C3", "dt", "method", "t_transition",
        "t_final", "control", "data_path", "RECORD_AVG",
        "initial_state", "selected_node_indices"
    ]

    def __init__(self, par={}):

        self.check_parameters(par)
        _par = self.get_default_parameters()
        _par.update(par)

        for item in _par.items():
            name = item[0]
            value = item[1]
            setattr(self, name, value)

        if self.seed is not None:
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)

        self.N = self.num_nodes = np.asarray(self.adj).shape[0]

        if self.initial_state is None:
            self.INITIAL_STATE_SET = False

        self.C0 = self.C0 * np.ones(self.N)
        self.C1 = self.C1 * np.ones(self.N)
        self.C2 = self.C2 * np.ones(self.N)
        self.C3 = self.C3 * np.ones(self.N)
        self.fix_seed = 1 if self.fix_seed else 0
        os.makedirs(join(self.data_path), exist_ok=True)

    def check_parameters(self, par):
        for key in par.keys():
            if key not in self.valid_params:
                raise ValueError("Invalid parameter: " + key)

    def get_default_parameters(self):

        par = {
            'G': 0.5,                   # global coupling strength
            "A": 3.25,                  # mV
            "B": 22.0,                  # mV
            "a": 0.1,                   # 1/ms
            "b": 0.05,                  # 1/ms
            "noise_mu": 0.24,
            "noise_std": 0.3,
            "vmax": 0.005,
            "v0": 6,                    # mV
            "r": 0.56,                  # mV
            # "selected_node_indices": None,
            "initial_state": None,

            'adj': None,
            "C0": 135.0 * 1.0,
            "C1": 135.0 * 0.8,
            "C2": 135.0 * 0.25,
            "C3": 135.0 * 0.25,

            "fix_seed": 0,
            "seed": None,

            "dt": 0.05,                 # ms
            "dim": 6,
            "method": "heun",
            "t_transition": 500.0,      # ms
            "t_final": 2501.0,          # ms
            "control": [],
            "data_path": "output",     # output directory
            "RECORD_AVG": False        # true to store large time series in file
        }
        return par

    # ---------------------------------------------------------------
    def set_initial_state(self):
        '''
        set initial state for the system of JR equations with N nodes.
        '''

        N = self.num_nodes
        y = set_initial_state(N)
        self.INITIAL_STATE_SET = True
        return y

    # -------------------------------------------------------------------------

    def set_C(self, label, val_dict):
        indices = val_dict['indices']

        if indices is None:
            indices = [list(range(self.N))]

        values = val_dict['value']
        if isinstance(values, np.ndarray):
            values = values.tolist()
        if not isinstance(values, list):
            values = [values]

        assert (len(indices) == len(values))
        C = getattr(self, label)

        for i in range(len(values)):
            C[indices[i]] = values[i]

    # -------------------------------------------------------------------------
    def simulate(self, par={}, x0=None, verbose=False):
        '''!
        integrate the system of equations for Jansen-Rit model.
        '''

        if x0 is None:
            if not self.INITIAL_STATE_SET:
                self.initial_state = self.set_initial_state()
                self.INITIAL_STATE_SET = True
                if verbose:
                    print("initial state set by default")
        else:
            self.INITIAL_STATE_SET = True

        for key in par.keys():
            if key not in self.valid_params:
                raise ValueError("Invalid parameter: " + key)
            if key in ["C0", "C1", "C2", "C3"]:
                self.set_C(key, par[key])
            else:
                setattr(self, key, par[key]['value'])

        obj = _JRN(self.N,
                   self.dt,
                   self.t_transition,
                   self.t_final,
                   self.G,
                   self.adj,
                   self.initial_state.tolist(),
                   self.A,
                   self.B,
                   self.a,
                   self.b,
                   self.r,
                   self.v0,
                   self.vmax,
                   self.C0.tolist(),
                   self.C1.tolist(),
                   self.C2.tolist(),
                   self.C3.tolist(),
                   self.noise_mu,
                   self.noise_std,
                   self.fix_seed)

        if self.method == 'euler':
            obj.eulerIntegrate()
        elif self.method == 'heun':
            obj.heunIntegrate()
        else:
            print("unkown integratiom method")
            exit(0)

        sol = np.asarray(obj.get_coordinates()).T
        times = np.asarray(obj.get_times())

        del obj

        return {"t": times, "x": sol}


def set_initial_state(nn):
    '''
    set initial state for the system of JR equations with N nodes.
    '''

    y0 = np.random.uniform(-1, 1, nn)
    y1 = np.random.uniform(-500, 500, nn)
    y2 = np.random.uniform(-50, 50, nn)
    y3 = np.random.uniform(-6, 6, nn)
    y4 = np.random.uniform(-20, 20, nn)
    y5 = np.random.uniform(-500, 500, nn)

    return np.hstack((y0, y1, y2, y3, y4, y5))


def calc_depth(lst):
    '''
    calculate the depth of a nested list
    '''
    if isinstance(lst, list):
        return 1 + max(calc_depth(item) for item in lst)
    else:
        return 0
