#!/usr/bin/env python
# Forked from https://github.com/ins-amu/DCM_PPLs

import os
import sys
import time
import errno
import timeit
import pathlib

import numpy as np
import arviz as az
import matplotlib.pyplot as plt

import jax 
import jax.numpy as jnp
from jax import grad, vmap, lax, random
from jax.experimental.ode import odeint
from jax.lib import xla_bridge


import numpyro as npr
from numpyro import sample, plate, handlers
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, Predictive, init_to_value

import os
import multiprocessing

print ("-"*60)
print ("-"*60)
print('Config:')
print ("-"*60)
#set up for parallelizing the chains

def setup_parallelization():
    num_cores = multiprocessing.cpu_count()
    os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={num_cores}"
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_platform_name", "cpu")
    print(f"Number of CPU cores: {num_cores}")
    print(f"Using devices: {jax.devices('cpu')}")


enable_parallelization = True

if enable_parallelization:
    setup_parallelization()
    print("Parallelizing chains.")
else:
    print("Skipping parallelization setup.")


npr.set_platform("cpu")

print ("-"*60)
print ("-"*60)
print('Deependencies:')

print(f"Numpy version: {np.__version__}")
print(f"JAX version: {jax.__version__}")
print(f"Numpyro version: {npr.__version__}")
print(f"Arviz version: {az.__version__}")
print ("-"*60)
print ("-"*60)

import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter(action="ignore", category=FutureWarning)


az.style.use("arviz-darkgrid")
colors_l = ["#A4C3D9", "#7B9DBF", "#52779F", "#2A537E"] 


cwd = os.getcwd()
main_path = str(pathlib.Path.cwd().parent)
sys.path.append(main_path) # Path to import the model and solver



from ERPmodel_EC_JAX import DCM_EC_ERPmodel, odeint_euler, odeint_heun, odeint_rk4

from jax import random
rng_key = random.PRNGKey(0)


tend = 200.0
dt = 0.1
t0 = 0.0
ts = np.arange(t0, tend + dt, dt)
nt = ts.shape[0]

ns = 9
x_init=np.zeros((ns))

theta_true = np.array([0.42, 0.76, 0.15, 0.16])
n_params = theta_true.shape[0]

my_var_names = ['g_1', 'g_2', 'g_3', 'g_4']


# Run the model
# We use Euler integration (see the ForwardModel for Heun and Rk4 integratores), But don't worry about computional time! we put JAX's JIT on Odeint to make it more faster!


@jax.jit
def ERP_JAXOdeintSimulator(x_init, ts, params):

    xs_rk4 = odeint_euler(DCM_EC_ERPmodel, x_init, ts, params)    
    x_py=xs_rk4[:,8]
    
    return x_py


# The initial compilation takes a bit of time, but after that, it flies through the air!

print ("-"*60)

start_time = time.time()

xpy_jax=ERP_JAXOdeintSimulator(x_init, ts, theta_true)

print("simulation with compiling took (sec):" , (time.time() - start_time))



start_time = time.time()

xpy_jax=ERP_JAXOdeintSimulator(x_init, ts, theta_true)

print("simulation using JAX's JIT took (sec):" , (time.time() - start_time))

print ("-"*60)


### Synthetic Observation

# We assume that we only have access to the activity of pyramidal neurons, and for the sake of speeding the computational time, we downsample the simuations.


#observation noise
sigma_true = 0.1 


xpy_jax = ERP_JAXOdeintSimulator(x_init, ts, theta_true)
x_noise = np.random.normal(loc=0, scale=sigma_true, size=xpy_jax.shape)
x_py = xpy_jax + x_noise

#downsampling
ds=10
ts_obs=ts[::ds]
xpy_obs=x_py[::ds]
nt_obs=int(x_py[::ds].shape[0])

ts_obs.shape, xpy_obs.shape, nt_obs

data= { 'nt_obs': nt_obs, 'ds': ds, 'ts': ts, 'ts_obs': ts_obs, 'dt': dt, 'x_init': x_init, 'obs_err': sigma_true, 'xpy_obs': xpy_obs }


from ERPmodel_helperplot import *


plot_observation(ts, xpy_jax, ts_obs, xpy_obs);


### Prior
# Since all the parameters are positive, we place Gamma prior, according to Refs [3,4].

shape=[18.16, 29.9, 29.14, 30.77]
scale=[0.03, 0.02, 0.005, 0.007]
rate = 1. / np.array(scale)

prior_specs = dict(shape=shape, rate=rate)

def model(data, prior_specs):
    #Data
    dt = data['dt']
    ts = data['ts']
    ds = data['ds']
    nt_obs = data['nt_obs']
    x_init = data['x_init']
    obs_err= data['obs_err']
    xpy_obs = data['xpy_obs']

    # Prior               
    g_1 = npr.sample('g_1', dist.Gamma(prior_specs['shape'][0], prior_specs['rate'][0]))
    g_2 = npr.sample('g_2', dist.Gamma(prior_specs['shape'][1], prior_specs['rate'][1]))
    g_3 = npr.sample('g_3', dist.Gamma(prior_specs['shape'][2], prior_specs['rate'][2]))
    g_4 = npr.sample('g_4', dist.Gamma(prior_specs['shape'][3], prior_specs['rate'][3]))

    #Parameters    
    params_samples=[g_1, g_2, g_3, g_4]
    
    #Forward model
    xpy_hat=ERP_JAXOdeintSimulator(x_init, ts, params_samples)[::ds]
    
    # Likelihood
    with plate('data', size=nt_obs):
        xpy_model = npr.deterministic('xpy_model', xpy_hat)
        npr.sample('xpy_obs', dist.Normal(xpy_model, obs_err), obs=xpy_obs)
        xpy_ppc = npr.sample('xpy_ppc', dist.Normal(xpy_model, obs_err))


#### Prior predictive check

n_ = 100
prior_predictive = Predictive(model, num_samples=n_)
prior_predictions = prior_predictive(rng_key, data, prior_specs)

title='Prior Predictive Check'
plot_priorcheck(ts_obs, xpy_obs, prior_predictions, n_, title);


# ## NUTS sampling 

n_warmup, n_samples, n_chains= 200, 200, 4


# NUTS set up
kernel = NUTS(model, max_tree_depth=12,  dense_mass=False, adapt_step_size=True)
mcmc= MCMC(kernel, num_warmup=n_warmup, num_samples=n_samples, num_chains=n_chains, chain_method='parallel')

print ("-"*60)
print ("-"*60)
print('Running the chains:')


#RUN NUTS
start_time = time.time()

mcmc.run(rng_key, data, prior_specs, extra_fields=('potential_energy', 'num_steps', 'diverging'))

print('Terminated the sampling.')
print ("-"*60)
print ("-"*60)


print(" All Chains using NUTS' Numpyro took (sec):" , (time.time() - start_time))


lp = -mcmc.get_extra_fields()['potential_energy']
print('Expected log joint density: {:.2f}'.format(np.mean(lp)))



title='Converged chains'
plot_lp_chains(lp, n_chains, title);


az.summary(mcmc, var_names=my_var_names)


#### Posterior 


# Get posterior samples
posterior_samples = mcmc.get_samples(group_by_chain=True)
pooled_posterior_samples = mcmc.get_samples()


# vizualize with arviz

az_obj = az.from_numpyro(mcmc)


# showing the posterior samples of all chains

axes = az.plot_trace(
    az_obj,
    var_names=my_var_names,
    compact=True,
    kind="trace",
    backend_kwargs={"figsize": (10, 6), "layout": "constrained"},)

for ax, true_val in zip(axes[:, 0], theta_true):
    ax.axvline(x=true_val, color='red', linestyle='--')  
for ax, true_val in zip(axes[:, 1], theta_true):
    ax.axhline(y=true_val, color='red', linestyle='--')
    
plt.gcf().suptitle("Converged NUTS", fontsize=16)
plt.tight_layout();


chains_pooled = az_obj.posterior[my_var_names].to_array().values.reshape(n_params, -1)
params_map_pooled=calcula_map(chains_pooled)


title="Pooled Posteriors"
plot_posterior_pooled(my_var_names, theta_true, prior_predictions, chains_pooled, title);


#### Fit and Posterior predictive check 


plot_fitted(data, az_obj.posterior);


pooled_posterior_predictive = Predictive(model=model, posterior_samples=pooled_posterior_samples, 
                                                      return_sites=['xpy_ppc'])
rng_key, rng_subkey = random.split(key=rng_key)
pooled_posterior_predictive_samples = pooled_posterior_predictive(rng_subkey, data, prior_specs)

ppc_=pooled_posterior_predictive_samples['xpy_ppc']
xpy_per05_pooled=np.quantile(ppc_, 0.05, axis=0)
xpy_per95_pooled=np.quantile(ppc_, 0.95, axis=0)


title='Pooled Posterior Predictive Check'
plot_posteriorcheck(data, xpy_per05_pooled, xpy_per95_pooled, title);


print ("-"*60)
print ("-"*60)
print('The end!')
print ("-"*60)
print ("-"*60)
