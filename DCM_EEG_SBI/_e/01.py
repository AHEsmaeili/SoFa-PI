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

from numpy.fft import fft, rfft
from scipy.signal import spectrogram

def simulation_wrapper(par,
                       subname, # could be an index 
                       parameters
                       ):

    if torch.is_tensor(par):
        par = np.float64(par.numpy())
    else:
        par = copy(par)
    try:
        _ = len(par)
    except:
        par = [par]

    sol = JRN(parameters)
    data = sol.simulate(par, subname)

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


# make complete graph

import networkx as nx
nn = 2
SC = nx.to_numpy_array(nx.complete_graph(nn))

C0 = 135.0
parameters = {
    'N': nn,
    'G': 0.5,                   # global coupling strength
    'adj': SC,
    "A": 3.25,                  # mV
    "B": 22.0,                  # mV
    "a": 0.1,                   # 1/ms
    "b": 0.05,                  # 1/ms
    "noise_mu": 0.24,
    "noise_std": 0.3,
    "vmax": 0.005,
    "v0": 6,                    # mV
    "r": 0.56,                  # mV
    "C": [1.0, 0.8, 0.25, 0.25],
    "C0": np.ones(nn) * C0,
    "fix_seed": 1,  # 1 to set fixed seed for noise in C++ code.
    "seed": seed,   # for initial state and parameters in Python code

    "dt": 0.05,                 # ms
    "dim": 6,
    "method": "heun",
    "t_transition": 500.0,      # ms
    "t_final": 2501.0,          # ms
    "control": ["G"],
    "data_path": "output",     # output directory
    "RECORD_AVG": False        # true to store large time series in file
}

data = simulation_wrapper([120], str(0), parameters)
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
plot_ts(data, parameters, [ax[0], ax[1]], alpha=0.6, lw=1) # color='teal',
ax[0].set_xlim(1000, 2500)
plt.savefig("ts.png", dpi=300)
plt.close()