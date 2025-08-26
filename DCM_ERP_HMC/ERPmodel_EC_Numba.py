#!/usr/bin/env python3
# Forked from https://github.com/ins-amu/DCM_PPLs


import numpy as np
import numba
from numba import  jit


c=-0.56
constants=np.array([c])

delta, tau_i, h_i, tau_e, h_e, u =np.array([12.13, 7.77, 27.87, 5.77, 1.63, 3.94])


@jit(nopython=True)
def Sigmodal(x1, x2, delta, c):
    S=(1./(1.+np.exp(c*(x1-(delta*x2)))))-0.5
    return S


@jit(nopython=True)
def DCM_ERPmodel(state, t, params):
    
    x0, x1, x2, x3, x4, x5, x6, x7, x8 = state
    g_1, g_2, g_3, g_4  = params
    
    c = constants[0]

    dx0 = x3
    dx1 = x4
    dx2 = x5
    dx6 = x7
    dx3 = (1./tau_e) * (h_e * (g_1 * (Sigmodal(x8, x4 - x5, delta, c)) + u) - (x0 / tau_e) - 2 * x3)
    dx4 = (1./tau_e) * (h_e * (g_2 * (Sigmodal(x0, x3, delta, c))) - (x1 / tau_e) - 2 * x4)
    dx5 = (1./tau_i) * (h_i * (g_4 * (Sigmodal(x6, x7, delta, c))) - (x2 / tau_i) - 2 * x5)
    dx7 = (1./tau_e) * (h_e * (g_3 * (Sigmodal(x8, x4 - x5, delta, c))) - (x6 / tau_e) - 2 * x7)
    dx8 = x4 - x5

    return np.array([dx0, dx1, dx2, dx3, dx4, dx5, dx6, dx7, dx8])


@jit(nopython=True)
def odeint_euler(f, y0, t, *args):
    ys = np.zeros((len(t), len(y0)))
    ys[0] = y0
    for i in range(1, len(t)):
        dt = t[i] - t[i-1]
        ys[i] = ys[i-1] + dt * f(ys[i-1], t[i-1], *args)
    return ys




@jit(nopython=True)
def odeint_heun(f, y0, t, *args):
    ys = np.zeros((len(t), len(y0)))
    ys[0] = y0
    for i in range(1, len(t)):
        h = t[i] - t[i-1]
        k1 = h * f(ys[i-1], t[i-1], *args)
        k2 = h * f(ys[i-1] + k1, t[i-1] + h, *args)
        ys[i] = ys[i-1] + 0.5 * (k1 + k2)
    return ys 



@jit(nopython=True)
def odeint_rk4(f, y0, t, *args):
    ys = np.zeros((len(t), len(y0)))
    ys[0] = y0
    for i in range(1, len(t)):
        h = t[i] - t[i-1]
        k1 = h * f(ys[i-1], t[i-1], *args)
        k2 = h * f(ys[i-1] + k1/2., t[i-1] + h/2., *args)
        k3 = h * f(ys[i-1] + k2/2., t[i-1] + h/2., *args)
        k4 = h * f(ys[i-1] + k3, t[i-1] + h, *args)
        ys[i] = ys[i-1] + 1./6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return ys
