#!/usr/bin/env python3
# Forked from https://github.com/ins-amu/DCM_PPLs


import numpy as np
import seaborn as sns
import arviz as az
import matplotlib.pyplot as plt

from matplotlib.collections import PathCollection
from matplotlib.legend_handler import HandlerPathCollection
marker_size = 150

colors_l = ["#A4C3D9", "#7B9DBF", "#52779F", "#2A537E"] 


def tails_percentile(my_var_names, prior_predictions, thr):
    tails_xth_percentile = {}
    for key, value in prior_predictions.items():
        if key in my_var_names:
            sorted_values = np.sort(value)[0, :] if value.shape[0] == 1 else np.sort(value)
            top_xth_percentile = sorted_values[int(0.05 * len(sorted_values))]
            tails_xth_percentile[key] = np.array(top_xth_percentile)
    return tails_xth_percentile


def calcula_map (chains_):
    if chains_.shape[0] <= 1:
        raise ValueError("Expected chains to have shape of n_params * n_samples")
    params_map = []
    for i in range(int(chains_.shape[0])):
        y=chains_[i]
        hist, bin_edges = np.histogram(y, bins=50)  # Adjust the number of bins as needed
        max_bin_index = np.argmax(hist)
        x_value_at_peak = (bin_edges[max_bin_index] + bin_edges[max_bin_index + 1]) / 2
        params_map.append(x_value_at_peak)
    return params_map


def my_axis(ax, params_labels):
    for a in ax[:, 1:].flatten():
        a.set_ylabel('')
    for i, prm_label in enumerate(params_labels):
        ax[i // 5, i % 5].set_xlabel(prm_label, fontsize=16)    


def plot_observation(ts_model, xpy_model, ts_obs, xpy_obs):
    plt.figure(figsize=(5,3))
    plt.plot(ts_model, xpy_model, color="b", lw=2,  alpha=0.8, label='model');
    plt.plot(ts_obs, xpy_obs, color="red", lw=1, marker=".", alpha=0.4, label='observation');
    plt.legend(fontsize=10, frameon=False, loc='upper right')
    plt.ylabel('Voltage [mV]', fontsize=12); 
    plt.xlabel('Time [ms]', fontsize=12); 
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    plt.tight_layout();
    #plt.savefig(os.path.join((output_dir),"Observation.png"), dpi=800)


def plot_priorcheck(ts_obs, xpy_obs, prior_predictions, n_, title):
    plt.figure(figsize=(5, 3))
    plt.plot(ts_obs, xpy_obs ,'.-', color='r', lw=1, label='observation');
    for i in range(n_):
        plt.plot(ts_obs, prior_predictions['xpy_model'][i], lw=1, alpha=0.2)
    plt.plot(ts_obs, prior_predictions['xpy_model'][i], lw=1, alpha=0.2, label='prior samples')    
    plt.title(title, fontsize=12)
    plt.legend(fontsize=10, frameon=False, loc='upper right')
    plt.ylabel('Voltage [mV]', fontsize=12); 
    plt.xlabel('Time [ms]', fontsize=12); 
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    plt.tight_layout();


def plot_lp_chains(lp, n_chains, title):

    fig, ax = plt.subplots(figsize=(8, 2))

    ax.plot(lp, label='chaque lp')
    ax.axhline(y=np.mean(lp), color='cyan', linestyle='--', label='Expected lp')
    for i in range(1, n_chains+1):
        x = i * len(lp) // n_chains
        ax.axvline(x, color='red', linestyle='--')
        ax.text(x-100, np.max(lp)-1, f'Chain {i}', color='red', fontsize=10, ha='center')
    plt.ylabel('lp', fontsize=14); 
    plt.xlabel('samples', fontsize=14); 
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    plt.legend(bbox_to_anchor=(.9, 1), loc='upper left', fontsize=10)
    plt.title(title, fontsize=14)  
    plt.tight_layout();
    return fig, ax


def plot_posterior_pooled(my_var_names, theta_true, prior_predictions, chains_pooled, title):

    params_map_pooled=calcula_map(chains_pooled)
    
    fig, ax = plt.subplots(ncols=4, nrows=1, figsize=(10, 2))
    for iprm, prm in enumerate(my_var_names) :
        a = ax[iprm]
        a.set_xlabel(prm, fontsize=12)
        a.set_ylabel('Density', fontsize=12)
        a.tick_params(axis='both', which='major', labelsize=10)
        a.axvline(theta_true[iprm], color='r', label='true', linestyle='--')
        a.axvline(params_map_pooled[iprm], color='darkblue', label='MAP', linestyle='--', lw=2.)
        sns.kdeplot(prior_predictions[prm], ax=a, color='lime', alpha=0.5, linestyle='-', lw=1,  label='prior', shade=True)
        sns.kdeplot(chains_pooled[iprm, :], ax=a, color='blue', alpha=0.2, linestyle='-', label='pooled chains', lw=1., shade=True)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    fig.suptitle(title, fontsize=14) 
    fig.tight_layout(rect=[0, 0, 1, 1])  
    return fig, ax

def plot_fitted(data, az_obj_posterior):

    ts_obs = data['ts_obs']
    xpy_obs = data['xpy_obs']
    ds = data['ds']

    n_chains = az_obj_posterior.dims['chain']

    fig, ax = plt.subplots(1, n_chains, figsize=(3*n_chains, 3))
    for ich in range(n_chains):
        if n_chains == 1 :
            a = ax 
        else :
            a = ax[ich]    
        a.plot(ts_obs, xpy_obs, '.', color='red', lw=3, label='obs')
        a.plot(ts_obs, az_obj_posterior['xpy_model'][ich, :, :].mean(axis=0), lw=2, color='b', label='fit')
        a.set_title(f'Fitted data for chain={ich+1}', fontsize=12)
        a.legend(fontsize=10, frameon=False, loc='upper right')
        a.set_xlabel('Time [ms]', fontsize=12)
        a.tick_params(axis='both', which='major', labelsize=10)
        a.tick_params(axis='both', which='minor', labelsize=8)
        if  ich==0:   
            a.set_ylabel('Voltage [mV]', fontsize=12)
    plt.tight_layout()
    return fig, ax


def plot_posteriorcheck(data, xpy_per05_pooled, xpy_per95_pooled, title):
    ts_obs = data['ts_obs']
    xpy_obs = data['xpy_obs']

    plt.figure(figsize=(5, 3))
    plt.plot(ts_obs, xpy_obs, color="red", lw=1, marker=".", alpha=0.4, label='observation');
    plt.fill_between(ts_obs, xpy_per05_pooled, xpy_per95_pooled, linewidth=2,facecolor='lightblue', edgecolor='blue', label='5-95% ppc', zorder=2, alpha=0.5)
    plt.title(title, fontsize=12)
    plt.legend(fontsize=10, frameon=False, loc='upper right')
    plt.ylabel('Voltage [mV]', fontsize=12); 
    plt.xlabel('Time [ms]', fontsize=12); 
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    plt.tight_layout();


def plot_corr(corr_vals, params_labels):
    fig = plt.figure(figsize=(5, 4))
    ax = fig.add_subplot(111)
    cmap = sns.diverging_palette(240, 10, sep=20, as_cmap=True)
    sns.heatmap(corr_vals, annot=True, robust=True, cmap=cmap, linewidths=.0, annot_kws={'size':8}, fmt=".2f", vmin=-1, vmax=1, ax=ax, xticklabels=params_labels, yticklabels=params_labels)

    for i in range(len(corr_vals)):
        for j in range(len(corr_vals)):
            text = ax.text(j + 0.5, i + 0.5, f"{corr_vals[i, j]:.2f}",
                           ha="center", va="center", color="black", fontsize=8)

    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=8)
    ax.tick_params(labelsize=12)
    return fig, ax
