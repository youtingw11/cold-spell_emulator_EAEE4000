import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
from glob import glob
import gcsfs
from tqdm import tqdm
import torch
import pickle

def make_dir(path):
    if os.path.exists(path) is False:
        os.makedirs(path)


def get_rmse(truth, pred):
    weights = np.cos(np.deg2rad(truth.latitude))
    return np.sqrt(((truth-pred)**2).weighted(weights).mean(['latitude', 'longitude'])).data.mean()

def plot_history(train_losses, val_losses):
    plt.figure()
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training History')
    plt.show()
    
# Utilities for normalizing the input data
def normalize(data, var, meanstd_dict):
    mean = meanstd_dict[var][0]
    std = meanstd_dict[var][1]
    return (data - mean)/std

def mean_std_plot(data,color,label,ax):
    
    mean = data.mean(['latitude','longitude'])
    std  = data.std(['latitude','longitude'])
    yr   = data.time.values

    ax.plot(yr,mean,color=color,label=label,linewidth=4)
    ax.fill_between(yr,mean+std,mean-std,facecolor=color,alpha=0.4)
    
    return yr, mean

def pytorch_train(model, optimizer, criterion, device, num_epochs, train_loader, val_loader):
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 20
    
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
       # training
       model.train()
       train_loss = 0.0
       for batch_X, batch_y in train_loader:
           batch_X = batch_X.to(device)
           batch_y = batch_y.to(device)
           # forward pass
           optimizer.zero_grad()
           outputs = model(batch_X)
           loss = criterion(outputs, batch_y)
           # backward pass
           loss.backward()
           optimizer.step()
           train_loss += loss.item()
    
        # validation
       model.eval()
       val_loss = 0.0
       with torch.no_grad():
           for batch_X, batch_y in val_loader:
               batch_X = batch_X.to(device)
               batch_y = batch_y.to(device)
               
               outputs = model(batch_X)
               loss = criterion(outputs, batch_y)
               val_loss += loss.item()
       
       train_loss /= len(train_loader)
       val_loss /= len(val_loader)
    
       train_losses.append(train_loss)
       val_losses.append(val_loss)
       
       print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
       
       if val_loss < best_val_loss:
           best_val_loss = val_loss
           patience_counter = 0
       else:
           patience_counter += 1
           if patience_counter >= patience:
               print(f'Early stopping at epoch {epoch+1}')
               break
   
    return train_losses, val_losses

def pytorch_train_VAE(model, optimizer, criterion, device, num_epochs, train_loader, val_loader, patience=20):    
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        total_train_samples = 0
        
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            batch_size = batch_X.size(0)
            
            optimizer.zero_grad()
            recon, z_mean, z_log_var = model(batch_y, batch_X)
            loss = criterion(batch_y, recon, z_mean, z_log_var)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * batch_size
            total_train_samples += batch_size
        
        train_loss /= total_train_samples
        
        model.eval()
        val_loss = 0.0
        total_val_samples = 0
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)
                batch_size = batch_X.size(0)
                
                recon, z_mean, z_log_var = model(batch_y, batch_X) 
                loss = criterion(batch_y, recon, z_mean, z_log_var)
                
                val_loss += loss.item() * batch_size
                total_val_samples += batch_size
        
        val_loss /= total_val_samples
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        print(f'Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping triggered at epoch {epoch+1}.')
                break
    
    return train_losses, val_losses


# ========================================

def load_and_build_X(
    pkl_path="data/INPUTS_DICT_cumsum.pkl",
    order=("historical", "ssp585", "ssp126", "ssp370", "hist-GHG", "hist-aer"),
    vars_keep=("BCB", "CH4", "CO2n", "SO2"),
    reindex_time=True,
):
    """
    Load forcing dictionary from .pkl, rename scenarios, select variables,
    cut years to match Y, standardize dims, and concat in time.
    """

    # 1. load raw dict
    with open(pkl_path, "rb") as f:
        ds_dict_raw = pickle.load(f)

    # 2. rename scenarios
    rename_map = {
        "hist":       "historical",
        "E213SSP585": "ssp585",
        "E213SSP126": "ssp126",
        "E213SSP370": "ssp370",
        "hist-GHG":   "hist-GHG",
        "hist-AER":   "hist-aer",
        # E213SSP534OVER, E213SSP245 intentionally dropped
    }
    ds_renamed = {
        new: ds_dict_raw[old]
        for old, new in rename_map.items()
        if old in ds_dict_raw
    }

    def year_slice_for(scen):
        # hist-like: use 1850–2014
        if scen in ("historical", "hist-GHG", "hist-aer"):
            return slice(1850, 2014)
        # SSPs: use 2015–2100 (X has no 2101)
        else:
            return slice(2015, 2100)

    datasets = []
    lengths = []

    for scen in order:
        scn_dict = ds_renamed[scen]
        data_vars = {}

        for var in vars_keep:
            arr = scn_dict[var]

            # case A: already DataArray
            if isinstance(arr, xr.DataArray):
                da = arr
            else:
                # case B: raw numpy
                arr = np.asarray(arr)
                if arr.ndim == 3:
                    da = xr.DataArray(arr, dims=("year", "latitude", "longitude"))
                elif arr.ndim == 1:
                    da = xr.DataArray(arr, dims=("year",))
                else:
                    raise ValueError(f"Unexpected shape for {scen}:{var} -> {arr.shape}")

            # apply year slice on 'year' coord
            if "year" in da.dims and "year" in da.coords:
                da = da.sel(year=year_slice_for(scen))

            # standardize dim names
            rename_dims = {}
            if "year" in da.dims:
                rename_dims["year"] = "time"
            if "lat" in da.dims:
                rename_dims["lat"] = "latitude"
            if "lon" in da.dims:
                rename_dims["lon"] = "longitude"
            if rename_dims:
                da = da.rename(rename_dims)

            data_vars[var] = da

        ds_piece = xr.Dataset(data_vars)
        datasets.append(ds_piece)
        lengths.append(ds_piece.sizes["time"])

    X = xr.concat(datasets, dim="time")

    if reindex_time:
        X = X.assign_coords(time=np.arange(X.sizes["time"]))

    return X, np.array(lengths)



def load_and_build_Y(
    base_dir,
    order=("historical", "ssp585", "ssp126", "ssp370", "hist-GHG", "hist-aer"),
    varname=None,
    reindex_time=True,
):
    """
    Load cold-spell NetCDFs, cut years to match X, rename dims, concat in time.
    """

    file_map = {
        "historical": "Historical_Daily_Tsavg_cold_spells.nc",
        "ssp585":     "SSP585_Daily_Tsavg_cold_spells.nc",
        "ssp126":     "SSP126_Daily_Tsavg_cold_spells.nc",
        "ssp370":     "SSP370_Daily_Tsavg_cold_spells.nc",
        "hist-GHG":   "HistGHG_Daily_Tsavg_cold_spells.nc",
        "hist-aer":   "HistAER_Daily_Tsavg_cold_spells.nc",
    }

    def year_slice_for(scen):
        if scen in ("historical", "hist-GHG", "hist-aer"):
            return slice(1850, 2014)
        else:
            return slice(2015, 2100)

    da_list = []
    lengths = []

    for scen in order:
        path = os.path.join(base_dir, file_map[scen])
        ds = xr.open_dataset(path, engine="netcdf4")

        # choose variable
        if varname is None:
            vname = list(ds.data_vars)[0]
        else:
            vname = varname

        da = ds[vname]

        # apply year slice on 'year' dim
        if "year" in da.dims and "year" in da.coords:
            da = da.sel(year=year_slice_for(scen))

        # standardize dims
        rename_dims = {}
        if "year" in da.dims:
            rename_dims["year"] = "time"
        if "lat" in da.dims:
            rename_dims["lat"] = "latitude"
        if "lon" in da.dims:
            rename_dims["lon"] = "longitude"
        if rename_dims:
            da = da.rename(rename_dims)

        if "time" not in da.dims:
            raise ValueError(f"'time' dim missing for {scen}, dims: {da.dims}")

        da_list.append(da)
        lengths.append(da.sizes["time"])

    Y = xr.concat(da_list, dim="time")

    if reindex_time:
        Y = Y.assign_coords(time=np.arange(Y.sizes["time"]))

    return Y, np.array(lengths)


def build_X_test_ssp245(
    pkl_path="data/INPUTS_DICT_cumsum.pkl",
    vars_keep=("BCB", "CH4", "CO2n", "SO2"),
    reindex_time=True,
):
    # load full forcing dict
    with open(pkl_path, "rb") as f:
        ds_dict_raw = pickle.load(f)

    scn_dict = ds_dict_raw["E213SSP245"]   # SSP245 scenario in the pickle

    data_vars = {}
    for var in vars_keep:
        arr = scn_dict[var]

        # DataArray vs numpy
        if isinstance(arr, xr.DataArray):
            da = arr
        else:
            arr = np.asarray(arr)
            if arr.ndim == 3:
                da = xr.DataArray(arr, dims=("year", "latitude", "longitude"))
            elif arr.ndim == 1:
                da = xr.DataArray(arr, dims=("year",))
            else:
                raise ValueError(f"Unexpected shape for SSP245:{var} -> {arr.shape}")

        # slice years 2015–2100
        if "year" in da.dims and "year" in da.coords:
            da = da.sel(year=slice(2015, 2100))

        # standardize dim names
        rename_dims = {}
        if "year" in da.dims:
            rename_dims["year"] = "time"
        if "lat" in da.dims:
            rename_dims["lat"] = "latitude"
        if "lon" in da.dims:
            rename_dims["lon"] = "longitude"
        if rename_dims:
            da = da.rename(rename_dims)

        data_vars[var] = da

    X = xr.Dataset(data_vars)

    if reindex_time:
        X = X.assign_coords(time=np.arange(X.sizes["time"]))

    return X

def build_y_test_ssp245(
    base_dir,
    varname=None,
    reindex_time=True,
):
    path = os.path.join(base_dir, "SSP245_Daily_Tsavg_cold_spells.nc")

    ds = xr.open_dataset(path, engine="netcdf4")

    # choose variable
    if varname is None:
        vname = list(ds.data_vars)[0]
    else:
        vname = varname

    da = ds[vname]

    # slice years 2015–2100
    if "year" in da.dims and "year" in da.coords:
        da = da.sel(year=slice(2015, 2100))

    # standardize dims
    rename_dims = {}
    if "year" in da.dims:
        rename_dims["year"] = "time"
    if "lat" in da.dims:
        rename_dims["lat"] = "latitude"
    if "lon" in da.dims:
        rename_dims["lon"] = "longitude"
    if rename_dims:
        da = da.rename(rename_dims)

    if "time" not in da.dims:
        raise ValueError(f"'time' dim missing for y_test SSP245, dims: {da.dims}")

    if reindex_time:
        da = da.assign_coords(time=np.arange(da.sizes["time"]))

    return da

def load_Y_baseline_nasa(base_dir, varname=None, reindex_time=True):
    """
    Return only the historical (baseline) Y from the cold-spell files.

    Output:
        Y_hist  : xarray.DataArray with dims (time, latitude, longitude)
        n_years : int, number of time steps
    """
    Y_all, lengths = load_and_build_Y(
        base_dir,
        order=("historical",),   # only historical
        varname=varname,
        reindex_time=reindex_time,
    )

    # since order has only one element, Y_all is just historical concatenated
    n_years = int(lengths[0])
    return Y_all, n_years
