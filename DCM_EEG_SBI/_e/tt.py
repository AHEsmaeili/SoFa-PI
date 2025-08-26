# add path
import sys
import torch
import warnings
import numpy as np
from copy import copy
from os.path import join
from scipy import signal
sys.path.append('../lib/')
from jansen_rit import JRN
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

seed = 2
torch.manual_seed(seed)
np.random.seed(seed)

LABESSIZE = 20
plt.rcParams['axes.labelsize'] = LABESSIZE
plt.rcParams['xtick.labelsize'] = LABESSIZE
plt.rcParams['ytick.labelsize'] = LABESSIZE

from numpy.fft import fft

def simulation_wrapper(par,
                       par_dict,
                       x0
                       ):

    if torch.is_tensor(par):
        par = np.float64(par.numpy())
    else:
        par = copy(par)
    try:
        _ = len(par)
    except:
        par = [par]

    sol = JRN(par_dict)
    data = sol.simulate(par, x0)
    exit(0)

    # extract features
    # ...

    return data #, features


def fft_signal(x, t):
    dt = t[1] - t[0]
    if x.ndim == 1:
        x = x[None, :]
    N = x.shape[1]
    T = N * dt
    xf = fft(x - x.mean(axis=1, keepdims=True), axis=1)
    Sxx = 2 * dt**2 / T * (xf * xf.conj()).real
    Sxx = Sxx[:, :N//2]

    df = 1.0 / T
    fNQ = 1.0 / (2.0 * dt)
    faxis = np.arange(0, fNQ, df)
    return faxis, Sxx


def plot_ts(data, par, ax, method="welch", **kwargs):
    tspan = data['t']
    y = data['x']
    ax[0].plot(tspan, y.T, label='y1 - y2', **kwargs)

    if method == "welch":
        freq, pxx = signal.welch(y, 1000/par['dt'], nperseg=y.shape[1]//2)
    else:
        freq, pxx = fft_signal(y, tspan / 1000)
    ax[1].plot(freq, pxx.T, **kwargs)
    ax[1].set_xlim(0, 50)
    ax[1].set_xlabel("frequency [Hz]")
    ax[0].set_xlabel("time [ms]")
    ax[0].set_ylabel("y1-y2")
    ax[0].margins(x=0)
    
    plt.tight_layout()


import networkx as nx
nn = 6
SC = nx.to_numpy_array(nx.complete_graph(nn))

par_dict = {
    "G": 1.0,
    "noise_mu": 0.24,
    "noise_std": 0.1,
    "dt": 0.05,
    "C0": 135.0 * 1.0,
    "C1": 135.0 * 0.8,
    "C2": 135.0 * 0.25,
    "C3": 135.0 * 0.25,
    "adj": SC,
    "t_transition": 500.0,      # ms
    "t_final": 2501.0,          # ms
    "data_path": "output",
}

control_dict = {
    "C1": {"indices": [[0, 1], [2, 3]], "value": [135.0, 155.0]},
    "G" : {"value": 1.0},
}

data = simulation_wrapper(control_dict, par_dict, x0=None)
