
"""
Created on Mon Dec 07 17:33:21 2023

@author: Mulugeta W. Asres, mulugetawa@uia.no

DESMOD: Root TH3 file (generated using HcalAnalyzer) to numpy conversion for the ML HCAL DQM Digioccupancy 

Example: given in main function

"""

from numba import njit
import io
import copy
import re
import pickle
import json
import gc
import random
import os, sys
import torch
import seaborn as sns
import time
import itertools
import functools
import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from statsmodels.tsa.stattools import adfuller
import plotly.graph_objects as go

import datetime

import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm

from sklearn.metrics import mean_squared_error, r2_score, make_scorer
from scipy.stats import norm, pearsonr
import warnings
import matplotlib.ticker as ticker
from pandas import Timedelta

warnings.filterwarnings("ignore")

def save_json(filepath, datadic):
    class MyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, slice):
                # Convert slice to a string representation
                return f"slice({obj.start}, {obj.stop}, {obj.step})"
            else:
                return super(MyEncoder, self).default(obj)

    print("saving: ", filepath)
    with open(filepath, 'w') as fhandle:
        fhandle.write(json.dumps(datadic, indent=4,
                                 sort_keys=True, cls=MyEncoder))
        fhandle.close()

def load_json(filepath):
    print("loading: ", filepath)
    with open(filepath, 'r') as fhandle:
        data_dict = json.load(fhandle)
        fhandle.close()
    return data_dict

def save_csv(filepath, df, index=True, ignore_format=False):
    print('saving ', filepath)
    if not ignore_format:
        df.to_csv(filepath, float_format='%6.12f', index=index)
    else:
        df.to_csv(filepath, index=index)

def load_csv(filepath, index_col=None, filepath_full=None, **kwargs):
    print('loading ', filepath)
    return pd.read_csv(filepath, index_col=index_col, **kwargs)

def save_txt(filepath, data):
    print(filepath)
    with open(filepath, "w") as f:
        f.write(repr(data))

def save_npdata(filepath, data):
    print(filepath)
    np.save(filepath, data)

def load_npdata(filepath):
    print(filepath )
    return np.load(filepath, allow_pickle=True)

def save_pickle(filepath, obj):
    print('saving ', filepath)
    with open(filepath, 'wb') as fhandle:
        pickle.dump(obj, fhandle)
        fhandle.close()

def load_pickle(filepath, model_ext="pkl"):
    print("loading ", filepath)
    with open(filepath, 'rb') as fhandle:
        obj = pickle.load(fhandle)
        fhandle.close()
    return obj

def save_figure(filepath, fig, isshow=False, issave=False, dpi=100):
    print("saving ", filepath)
    if issave:
        fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
        if isshow:
            plt.show(fig)
        else:
            plt.close()
    if isshow:
        plt.show(fig)

def show_figure(fig):
    dummy = plt.figure()
    new_manager = dummy.canvas.manager
    new_manager.canvas.figure = fig
    fig.set_canvas(new_manager.canvas)

def plot_2d_heatmap(input_data, mask="positive", **kwargs):
    '''
    input_data: is 2D numpy array or dataframe
    kwargs: vmin=None, vmax=None, cmap=None, center=None, robust=False, annot=None, fmt=".2g", annot_kws=None, linewidths=0, linecolor="white", cbar=True, cbar_kws=None, cbar_ax=None, square=False, xticklabels="auto", yticklabels="auto",
    '''
    figsize = kwargs.pop("figsize", (4, 4))
    axes_labels = kwargs.pop("axes_labels", ["x", "y"])
    n_hemisphere = kwargs.pop("n_hemisphere", 2)
    n_major_tick_gap = kwargs.pop("n_major_tick_gap", 4)
    cmap = kwargs.pop("cmap", "Spectral_r")
    title = kwargs.pop("title", None)
    axes_ticks = kwargs.pop("axes_ticks", [np.arange(input_data.shape[0]), np.arange(input_data.shape[1])])
    
    x_arr = np.array(axes_ticks[0])
    y_arr = np.array(axes_ticks[1])
    
    if not isinstance(input_data, pd.DataFrame):
        input_data = pd.DataFrame(input_data, columns=y_arr, index=x_arr)

    hist_data = copy.deepcopy(input_data)
    hist_data.index = hist_data.index.astype(int)

    if mask == "positive":
        hist_data[~(hist_data > 0)] = np.nan
    elif isinstance(mask, np.ndarray):
        hist_data[mask] = np.nan

    fig, ax = plt.subplots(figsize=figsize)

    hist_data = hist_data.reindex(
        sorted(hist_data.columns, reverse=True), axis=1)
    hist_data = hist_data.T

    
    if n_hemisphere == 2:
        yticks = np.arange(0.5, hist_data.shape[0], 1)
        xticks = np.arange(0.5, hist_data.shape[1], 1)
        yticks_major = np.arange(0.5, hist_data.shape[0], n_major_tick_gap)
        xticks_major = np.arange(0.5, hist_data.shape[1], n_major_tick_gap)
        xticks_major[(hist_data.shape[1]//2)//n_major_tick_gap:] = xticks_major[(hist_data.shape[1]//2)//n_major_tick_gap:] + n_major_tick_gap -1
        yticklabels = hist_data.index.values.tolist()[::n_major_tick_gap]
        xticklabels = hist_data.columns.tolist()[::n_major_tick_gap]
        xticklabels[len(xticklabels)//2:] = hist_data.columns.tolist()[(hist_data.shape[1]//2)-1+n_major_tick_gap::n_major_tick_gap]

    else:
        yticks = np.arange(0.5, hist_data.shape[0], 1)
        xticks = np.arange(0.5, hist_data.shape[1], 1)

        yticks_major = np.arange(0.5, hist_data.shape[0], n_major_tick_gap)
        xticks_major = np.arange(0.5, hist_data.shape[1], n_major_tick_gap//2)
        yticklabels = hist_data.index.values.tolist()[::n_major_tick_gap]
        xticklabels = hist_data.columns.tolist()[::n_major_tick_gap//2]


    if isinstance(cmap, list):
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", cmap)

    g = sns.heatmap(hist_data, cmap=cmap, ax=ax, linewidths=0.00, 
                    **kwargs)

    g.tick_params(left=True, bottom=True, which='both') 
    g.tick_params(which='major', width=1.5, length=6)
    g.tick_params(which='minor', width=1, length=3, color='b')
    g.xaxis.set_minor_locator(ticker.FixedLocator(xticks))
    g.xaxis.set_major_locator(ticker.FixedLocator(xticks_major))
    g.yaxis.set_minor_locator(ticker.FixedLocator(yticks))
    g.yaxis.set_major_locator(ticker.FixedLocator(yticks_major))
    
    g.set_xticklabels(xticklabels, rotation=90)
    g.set_yticklabels(yticklabels)
    g.set_xlabel(axes_labels[0])
    g.set_ylabel(axes_labels[1])
    g.set_title(title)
    
    fig.show()
    fig.savefig(f"2D_heatmap_{title}.pdf")
    
def plot_hist3d_heatmap(input_data, mask="positive", **kwargs):
    figsize = kwargs.pop("figsize", (6, 6))
    axes_labels = kwargs.pop("axes_labels", ["x", "y", "z"])
    title = kwargs.pop("title", "")
    cmap = kwargs.pop("cmap", "Spectral_r")
    axes_ticks = kwargs.pop("axes_ticks", [np.arange(input_data.shape[0]), np.arange(input_data.shape[1]), np.arange(input_data.shape[2])])

    hist_data = copy.deepcopy(input_data)

    x_arr = np.array(axes_ticks[0])
    y_arr = np.array(axes_ticks[1])
    z_arr = np.array(axes_ticks[2])

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
   

    if mask == "positive":
        x, y, z = hist_data.nonzero()
    elif isinstance(mask, np.ndarray):
        hist_data[mask] = np.nan
        x, y, z = (hist_data > -np.inf).nonzero()
    else:
        x, y, z = (hist_data > -np.inf).nonzero()

    x_arr, y_arr, z_arr = x_arr[x], y_arr[y], z_arr[z]
    if isinstance(cmap, list):
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", cmap)

    color = hist_data[x, y, z]
    ax_3d = ax.scatter(x_arr, y_arr, z_arr, c=color, marker=".", cmap=cmap)
    ax.set_xlabel(axes_labels[0], labelpad=10)
    ax.set_ylabel(axes_labels[1], labelpad=10)
    ax.set_zlabel(axes_labels[2], labelpad=10)
    ax.set_title(title)

    # horizontal
    cax = fig.add_axes([ax.get_position().x0,
                        ax.get_position().y0-0.04, ax.get_position().width, 0.02])
    cbar = fig.colorbar(ax_3d, orientation="horizontal", pad=1, cax=cax)
    # fig.show()
    # fig.savefig(f"3D_heatmap_{title}.pdf")
    plt.show()

def segidx_arrayidx_mapper(idx_input, axis="ieta", istoarray=True):
    '''
    segidx_arrayidx_mapper for hcal
    idx: value or array
    axis="ieta", iphi or "depth"

    ieta_len = 64
    iphi_len = 72
    depth_len = 7

    hcal_he_seg_config = np.zeros((ieta_len, iphi_len, depth_len))
    for depth in he_depths:
        for ieta in he_ietas:
            try:
                iphi = np.array(
                    list(seg_grp.loc[((depth, ieta)), ("Phi", "set")]))
                hcal_he_seg_config[(segidx_arrayidx_mapper(-ieta, axis="ieta", istoarray=True), segidx_arrayidx_mapper(ieta, axis="ieta", istoarray=True)),
                                    segidx_arrayidx_mapper(iphi, axis="iphi", istoarray=True)[:, np.newaxis], segidx_arrayidx_mapper(depth, axis="depth", istoarray=True)] = 1
            except Exception as ex:
                print("non he segement: depth={}, ieta={}, flag: {}".format(
                    depth, ieta, ex))

    '''

    idx = idx_input.copy() if isinstance(idx_input, np.ndarray) else idx_input
    seg_len = {"ieta": 64, "iphi": 72, "depth": 7}
    hcal_side_ieta_len = seg_len["ieta"]//2
    isinput_array_type = True
    if isinstance(idx, list):
        idx = np.array(idx)
    if not isinstance(idx, np.ndarray):
        isinput_array_type = False
        idx = np.array(idx)
    # print(idx)
    if istoarray:
        assert (idx != 0).all(
        ), "invalid hcal segmentation {} index id.".format(axis)

        if axis == "ieta":
            assert (np.abs(idx) < hcal_side_ieta_len).all(
            ), "invalid hcal segmentation {} index id.".format(axis)
            side = idx > 0
            idx[side] = idx[side] + hcal_side_ieta_len - 1
            idx[~side] = idx[~side] + hcal_side_ieta_len
        else:
            assert ((idx > 0) & (idx <= seg_len[axis])).all(
            ), "invalid hcal segmentation {} index id.".format(axis)
            idx = idx-1
    else:

        assert (idx >= 0).all() & (idx < seg_len[axis]).all(
        ), "invalid hcal segmentation {} array equivalent index id.".format(axis)

        if axis == "ieta":
            side = idx > (hcal_side_ieta_len - 1)
            idx[side] = idx[side] - hcal_side_ieta_len + 1
            idx[~side] = idx[~side] - hcal_side_ieta_len
        else:
            idx = idx + 1

    return idx if isinput_array_type else idx.squeeze()

def TH3_mask_onnx(digi_map_th3_np, mask, value: int):
    print("applying detector masking (negative)...")
    print(f"digi_map_th3_np: {digi_map_th3_np.shape}, mask: {mask.shape}")
    is_np = isinstance(digi_map_th3_np, np.ndarray)
    if is_np:
        digi_map_th3_np = torch.from_numpy(digi_map_th3_np)
        mask = torch.from_numpy(mask).type(torch.bool)

    for ls in range(digi_map_th3_np.shape[0]):
        digi_map_th3_np[ls].masked_fill_(mask, value)
    if is_np:
        return digi_map_th3_np.cpu().detach().numpy()
    return  digi_map_th3_np

def TH3Obj_np_data_cleaning(TH3Obj_np, **kwargs):
    '''
    TH3Obj_np_data_cleaning, removing unfilled channels
    '''
    replace_val = kwargs.get("replace_val", 0)
    TH3Obj_np[:, :, -1] = replace_val  # unmasked but no data

    return TH3Obj_np

def torch_nanmean(v, *args, inplace=False, **kwargs):
    if not inplace:
        v = v.detach().clone()
    is_nan = torch.isnan(v)
    v[is_nan] = 0
    return v.sum(*args, **kwargs) / (~is_nan).float().sum(*args, **kwargs)

def ae(y_true, y_pred):
    return np.abs(y_true - y_pred)

def mae(y_true, y_pred, multioutput=True, axis=0):
    if multioutput:
        return np.nanmean(ae(y_true, y_pred), axis=axis)
    else:
        return np.nanmean(ae(y_true, y_pred))
    
def mae_torch(y_true, y_pred, multioutput=True, axis=(0)):
    if multioutput:
        return torch_nanmean((y_true - y_pred).abs(), axis=axis)
    else:
        return np.nanmean((y_true - y_pred).abs())

def plot_array(array, num, plot_title, save_path=None):

    if array.ndim == 6:
        first_entry = array[num, 0, :, :, :, 0]
    elif array.ndim == 3:
        first_entry = array[:,:,:]
    elif array.ndim == 4:
        first_entry = array[num,:,:,:]
    else:
        first_entry = array[num, :, :, :, 0]

    # Create a grid of coordinates corresponding to the data shape (64, 72, 7)
    x, y, z = np.meshgrid(np.arange(first_entry.shape[0]),
                          np.arange(first_entry.shape[1]),
                          np.arange(first_entry.shape[2]), indexing='ij')

    # Flatten the arrays for plotting
    x_flat = x.flatten()
    y_flat = y.flatten()
    z_flat = z.flatten()
    values_flat = first_entry.flatten()

    # Create a 3D scatter plot
    fig = plt.figure(figsize=(6,8))
    ax = fig.add_subplot(111, projection='3d')
    cax = fig.add_axes([ax.get_position().x0,
                        ax.get_position().y0-0.04, ax.get_position().width, 0.02])

    # Use values_flat as the color of the points
    scatter = ax.scatter(y_flat, x_flat, z_flat, c=values_flat, marker=".", cmap='Spectral_r')

    # Add a color bar
    fig.colorbar(scatter, orientation="horizontal", cax=cax, pad=1)

    # Set labels
    ax.set_ylabel('ieta')
    ax.set_xlabel('iphi')
    ax.set_zlabel('depth')

    ax.set_title(plot_title)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Plot saved to {save_path}") 

    # Show the plot
    plt.show()