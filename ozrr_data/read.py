from ozrr.data import xr_time_series
import pandas as pd
import xarray as xr
from pathlib import Path, PosixPath

from .conventions import SERIES_VARNAME, TIME_DIM_NAME


def data_load(fn):
    return pd.read_csv(fn, skiprows=range(0, 15))


def load_rr_series(fn):
    rr_series = data_load(fn)
    return xr_time_series(rr_series)


def get_catchment_id(fn: PosixPath):
    return fn.name.split(".")[0].split("_")[-1]


def load_bivariate_series(
    folder: PosixPath, glob="series*.csv", get_catchment_id=get_catchment_id
):
    files = [f for f in folder.glob(glob)]
    station_ids = [get_catchment_id(f) for f in files]
    wpb_df = [pd.read_csv(f) for f in files]
    return wpb_df, station_ids
