#!/usr/bin/env python3
# Forked from https://github.com/ins-amu/DCM_PPLs



import jax 
import jax.numpy as jnp
from jax import lax


c=-0.56
constants=jnp.array([c])

delta, tau_i, h_i, tau_e, h_e, u =jnp.array([8.41, 7.77, 27.87, 5.77, 1.63, 3.94])

@jax.jit
def Sigmodal(x1, x2, delta, c):
    S=(1./(1.+jnp.exp(c*(x1-(delta*x2)))))-0.5
    return S


@jax.jit
def DCM_EC_ERPmodel(state, t, params):
    
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

    return jnp.array([dx0, dx1, dx2, dx3, dx4, dx5, dx6, dx7, dx8])



def odeint_euler(f, y0, t, *args):
    def step(state, t):
        y_prev, t_prev = state
        dt = t - t_prev
        y = y_prev + dt * f(y_prev, t_prev, *args)
        return (y, t), y
    _, ys = lax.scan(step, (y0, t[0]), t[0:])

    return ys



def odeint_heun(f, y0, t, *args):
    def step(state, t):
        y_prev, t_prev = state
        h = t - t_prev
        k1 = h * f(y_prev, t_prev, *args)
        k2 = h * f(y_prev + k1, t_prev + h, *args)
        y = y_prev + 0.5 * (k1 + k2)
        return (y, t), y
    _, ys = lax.scan(step, (y0, t[0]), t[0:])
    return ys
    


def odeint_rk4(f, y0, t, *args):
    def step(state, t):
        y_prev, t_prev = state
        h = t - t_prev
        k1 = h * f(y_prev, t_prev, *args)
        k2 = h * f(y_prev + k1/2., t_prev + h/2., *args)
        k3 = h * f(y_prev + k2/2., t_prev + h/2., *args)
        k4 = h * f(y_prev + k3, t + h, *args)
        y = y_prev + 1./6 * (k1 + 2 * k2 + 2 * k3 + k4)
        return (y, t), y
    _, ys = lax.scan(step, (y0, t[0]), t[0:])
    return ys    
