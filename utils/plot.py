import json
import gzip
import _pickle as pickle
import logging
import yaml
import json
from logging import Logger
import re
from datetime import datetime
import os
import pytz
from matplotlib import pyplot as plt 
import seaborn as sns
from scipy.stats import bootstrap
import numpy as np
import pandas as pd

plt.rcParams.update({
    'font.size': 16
})

plt_params = {'linewidth': 2.2}


def plot_ci_simple(data, ax, dim=1, **kwargs):
    """
    Plots the mean and confidence interval of the given data on the specified axis.

    Parameters:
    - data: A tensor or array-like object containing the data.
    - ax: The axis object on which to plot the data.
    - dim: The dimension along which to compute the mean and confidence interval.
    - **kwargs: Additional keyword arguments to be passed to the plot function.

    Returns:
    None
    """
    mean = data.mean(dim=dim)
    std = data.std(dim=dim)
    sem95 = 1.96 * std / (len(data)**0.5) 
    ax.plot(range(len(mean)), mean, **kwargs)
    ax.fill_between(range(len(mean)), mean - sem95, mean + sem95, alpha=0.3)
    
plt_params = {'linewidth': 2.2}
def plot_ci_plus_heatmap(data, heat, labels, 
                         color='blue', 
                         linestyle='-',
                         tik_step=10, 
                         method='gaussian', 
                         do_lines=True, 
                         do_colorbar=False, 
                         shift=0.5, 
                         nums = [.99, 0.18, 0.025, 0.6],
                         labelpad=10,
                         plt_params=plt_params):
    
    fig, (ax, ax2) = plt.subplots(nrows=2, sharex=True, gridspec_kw={'height_ratios': [1, 10]}, figsize=(5, 3))
    if do_colorbar:
        fig.subplots_adjust(right=0.8) 
    plot_ci(ax2, data, labels, color=color, linestyle=linestyle, tik_step=tik_step, method=method, do_lines=do_lines, plt_params=plt_params)
    
    y = heat.mean(dim=0)
    x = np.arange(y.shape[0])+1

    extent = [x[0]-(x[1]-x[0])/2. - shift, x[-1]+(x[1]-x[0])/2. + shift, 0, 1]
    img =ax.imshow(y[np.newaxis,:], cmap="plasma", aspect="auto", extent=extent, vmin=0, vmax=14)
    ax.set_yticks([])
    #ax.set_xlim(extent[0], extent[1])
    if do_colorbar:
        cbar_ax = fig.add_axes(nums)  # Adjust these values as needed
        cbar = plt.colorbar(img, cax=cbar_ax)
        cbar.set_label('entropy', rotation=90, labelpad=labelpad)  # Adjust label and properties as needed
    plt.tight_layout()
    return fig, ax, ax2


def plot_ci_plus_heatmap(data, heat, labels, 
                         color='blue', 
                         linestyle='-',
                         tik_step=10, 
                         method='gaussian', 
                         do_lines=True, 
                         do_colorbar=False, 
                         shift=0.5, 
                         nums = [.99, 0.18, 0.025, 0.6],
                         labelpad=10,
                         plt_params=plt_params):
    
    fig, (ax, ax2) = plt.subplots(nrows=2, sharex=True, gridspec_kw={'height_ratios': [1, 10]}, figsize=(5, 3))
    if do_colorbar:
        fig.subplots_adjust(right=0.8) 
    plot_ci(ax2, data, labels, color=color, linestyle=linestyle, tik_step=tik_step, method=method, do_lines=do_lines, plt_params=plt_params)
    
    y = heat.mean(dim=0)
    x = np.arange(y.shape[0])+1

    extent = [x[0]-(x[1]-x[0])/2. - shift, x[-1]+(x[1]-x[0])/2. + shift, 0, 1]
    img =ax.imshow(y[np.newaxis,:], cmap="plasma", aspect="auto", extent=extent, vmin=0, vmax=14)
    ax.set_yticks([])
    #ax.set_xlim(extent[0], extent[1])
    if do_colorbar:
        cbar_ax = fig.add_axes(nums)  # Adjust these values as needed
        cbar = plt.colorbar(img, cax=cbar_ax)
        cbar.set_label('entropy', rotation=90, labelpad=labelpad)  # Adjust label and properties as needed
    plt.tight_layout()
    return fig, ax, ax2

def process_axis(ax, ylabel_font=13, xlabel_font=13):
    ax.spines[['right', 'top']].set_visible(False)
    #ax.set_ylabel(ylabel, fontsize=ylabel_font)
    #ax.set_xlabel(xlabel, fontsize=xlabel_font)

def plot_ci(ax, data, label, color='blue', linestyle='-', tik_step=10, method='gaussian', do_lines=True, plt_params=plt_params):
    """
    Plots the mean of the data with a confidence interval (CI) envelope on the specified axes.

    This function allows for the visualization of uncertainty around the mean estimate of the provided data
    using three different methods to calculate the confidence interval: Gaussian, non-parametric (np), and bootstrap.

    Parameters:
    - ax (matplotlib.axes.Axes): The matplotlib Axes object where the data will be plotted.
    - data (numpy.ndarray): The dataset for which the mean and confidence interval are to be calculated and plotted. 
      The dataset should be in the form of a 2D array where rows represent different observations/trials and 
      columns represent sequential data points or time steps.
    - label (str): The label for the data series to be used in the plot legend.
    - color (str, optional): The color of the plot line and confidence interval shading. Default is 'blue'.
    - linestyle (str, optional): The linestyle of the plot line. Default is '-' (solid line).
    - tik_step (int, optional): The interval between x-axis tick marks. Default is 10.
    - method (str, optional): The method used to calculate the confidence interval. Options are 'gaussian' for 
      assuming a normal distribution, 'np' for a non-parametric approach using quantiles, and 'bootstrap' for 
      using bootstrapping to estimate the CI. Default is 'gaussian'.
    - do_lines (bool, optional): If True, vertical dashed lines will be added at each x-axis tick mark for 
      better readability. Default is True.
    - plt_params (dict, optional): Additional plotting parameters to be passed to the ax.plot function call.

    Raises:
    - ValueError: If an unsupported method is specified.

    Note: The 'process_axis' function called at the end is not defined within this docstring. It should be 
    defined elsewhere to process the axis labels, ticks, and other properties.
    """
    if do_lines:
        upper = max(round(data.shape[1]/10)*10+1, data.shape[1]+1)
        ax.set_xticks(np.arange(0, upper, tik_step))
        for i in range(0, upper, tik_step):
            ax.axvline(i, color='black', linestyle='--', alpha=0.2, linewidth=1)
    if method == 'gaussian':
        mean = data.mean(dim=0)
        std = data.std(dim=0)
        data_ci = {
            'x' : np.arange(data.shape[1])+1,
            'y' : mean,
            'y_upper' : mean + (1.96/(data.shape[0]**0.5)) * std,
            'y_lower' : mean - (1.96/(data.shape[0]**0.5)) * std,
        }
    elif method == 'np':
        data_ci = {
            'x' : np.arange(data.shape[1])+1,
            'y' : np.quantile(data, 0.5, axis=0),
            'y_upper' : np.quantile(data, 0.95, axis=0),
            'y_lower' : np.quantile(data, 0.05, axis=0),
        }
    elif method == 'bootstrap':
        bootstrap_ci = bootstrap((data,), np.mean, confidence_level=0.95, method='percentile')
        data_ci = {
            'x' : np.arange(data.shape[1])+1,
            'y' : data.mean(axis=0),
            'y_upper' : bootstrap_ci.confidence_interval.high,
            'y_lower' : bootstrap_ci.confidence_interval.low,
        }

    else:
        raise ValueError('method not implemented')

    df = pd.DataFrame(data_ci)
    # Create the line plot with confidence intervals
    ax.plot(df['x'], df['y'], label=label, color=color, linestyle=linestyle, **plt_params)
    ax.fill_between(df['x'], df['y_lower'], df['y_upper'], color=color, alpha=0.3)
    process_axis(ax)
