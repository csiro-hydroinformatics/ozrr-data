"""Access arrangements to the dataset by Lerat et al. [REF]
"""
from pathlib import Path
from typing import Callable, List, Optional, Union
import pandas as pd
import xarray as xr
import numpy as np
import os

from ozrr_data.data import to_xarray_ts, xr_time_series

from .read import data_load, load_rr_series, get_catchment_id

from .conventions import (
    DAILY_SERIES_VARNAME,
    EVAP_D_COL,
    LAT_VARNAME,
    LON_VARNAME,
    NAME_ATTRIB,
    NAME_ATTRIB_VALUE_DAILY,
    RAIN_D_COL,
    RUNOFF_D_COL,
    SERIES_VARNAME,
    STATION_ID_VARNAME,
    STATION_NAME_VARNAME,
    STATIONS_DIM_NAME,
    TEMPMAX_D_COL,
    TIME_DIM_NAME,
    XR_UNITS_ATTRIB_ID,
)


def set_xr_units(x: xr.DataArray, units: str) -> None:
    """Sets the units attribute of an xr.DataArray. No effect if x is not a dataarray

    Args:
        x (xr.DataArray): data array
        units (str): units descriptor
    """
    if units is None:
        return
    if isinstance(x, xr.DataArray):
        x.attrs[XR_UNITS_ATTRIB_ID] = units


def get_xr_units(x: xr.DataArray) -> str:
    """Gets the units attribute of an xr.DataArray. Empty string if no attribute

    Args:
        x (xr.DataArray): data array
    """
    assert isinstance(x, xr.DataArray)
    if XR_UNITS_ATTRIB_ID not in x.attrs.keys():
        return ""
    else:
        return x.attrs[XR_UNITS_ATTRIB_ID]


# def multiindex_to_monoindex_series(x_multiindex:xr.DataArray):
# y = x_multiindex.year_month_level_0.to_numpy()
# m = x_multiindex.year_month_level_1.to_numpy()
# dates = ['-'.join([str(y[i]), str(m[i]), '01']) for i in range(len(y))]
# time_index = [pd.Timestamp(x) for x in dates]
#     x = mk_xarray_series(
#         data=x_multiindex.values.T,
#         dim_name=STATION_ID_VARNAME,
#         units=None,
#         time_index=time_index,
#         colnames=[i for i in range(x_multiindex.station.shape[0])],
#         fill_miss_func= None,
#     )
#     return x


def to_monoindex_series(x_multiindex: xr.DataArray) -> xr.DataArray:
    """Transform a yearly or monthly time series with possibly multi-indices to a CF time series

    Work in progress; possibly too use case specific
    """
    if len(x_multiindex.station.shape) == 0:  # degenerate array, scalar value
        station_ids = [str(x_multiindex.station.values)]
    else:
        station_ids = x_multiindex.station.values

    vals = x_multiindex.values.T

    if hasattr(x_multiindex, "year"):
        y = x_multiindex.year.to_numpy()
        dates = ["-".join([str(y[i]), "01", "01"]) for i in range(len(y))]
    elif hasattr(x_multiindex, "year_month"):
        # Work around a breaking change with v2022.6.0  https://docs.xarray.dev/en/stable/whats-new.html#v2022-06-0-july-21-2022
        if hasattr(x_multiindex, "time_level_0"):
            y = x_multiindex.time_level_0.to_numpy()
            m = x_multiindex.time_level_1.to_numpy()
        else:
            y = x_multiindex.year_month_level_0.to_numpy()
            m = x_multiindex.year_month_level_1.to_numpy()
        dates = ["-".join([str(y[i]), str(m[i]), "01"]) for i in range(len(y))]

    time_index = [pd.Timestamp(x) for x in dates]
    x = mk_xarray_series(
        data=vals,
        dim_name=STATION_ID_VARNAME,
        units=None,
        time_index=time_index,
        colnames=station_ids,
        fill_miss_func=None,
    )
    return x


def to_pd_series(x: xr.DataArray) -> pd.Series:
    """Converts an xarray multiindex monthly time series to a pandas series

    Very use case specific. Revisit.
    """
    indx = pd.DatetimeIndex(
        [pd.Timestamp(year=m_y[0], month=m_y[1], day=1) for m_y in x.year_month.values]
    )
    return pd.Series(x.values, indx)


def _group_monthly(daily_series: xr.DataArray):
    return daily_series.groupby("year_month")


def _group_yearly(daily_series: xr.DataArray):
    return daily_series.groupby("year")


def sum_monthly(daily_series: xr.DataArray, skipna=False):
    return _group_monthly(daily_series).sum(skipna=skipna)


def max_monthly(daily_series: xr.DataArray, skipna=False):
    return _group_monthly(daily_series).max(skipna=skipna)


def mean_monthly(daily_series: xr.DataArray, skipna=False):
    return _group_monthly(daily_series).mean(skipna=skipna)


def sum_yearly(daily_series: xr.DataArray, skipna=False):
    return _group_yearly(daily_series).sum(skipna=skipna)


def max_yearly(daily_series: xr.DataArray, skipna=False):
    return _group_yearly(daily_series).max(skipna=skipna)


def mean_yearly(daily_series: xr.DataArray, skipna=False):
    return _group_yearly(daily_series).mean(skipna=skipna)


class OzDataProcessing:
    """Class handling the lower-level ingestion of on-disk data

    Data set from [Lerat, J., Thyer, M., McInerney, D., Kavetski, D., Woldemeskel, F., Pickett-Heaps, C., Shin, D., & Feikema, P. (2020). A robust approach for calibrating a daily rainfall-runoff model to monthly streamflow data. Journal of Hydrology, 591. https://doi.org/10.1016/j.jhydrol.2020.125129](https://doi.org/10.1016/j.jhydrol.2020.125129)
    """

    def __init__(self, root_dir: Path, nc_filename: Path = None) -> None:
        self.root_dir = root_dir
        self.sites_csv = root_dir / "sites.csv"
        self.rr_dir = root_dir / "rainfall_runoff"
        self.rr_nc = (
            nc_filename
            if nc_filename is not None
            else (root_dir / "rainfall_runoff_series.nc")
        )
        self.rr_series = None

    def load_daily_data(
        self, discard_cached_nc: bool = False, lazy_loading: bool = False
    ) -> xr.DataArray:
        import json

        if not self.rr_nc.exists() or discard_cached_nc:
            files = [f for f in self.rr_dir.glob("*.csv")]
            cat_ids = [get_catchment_id(fn) for fn in files]
            series = [load_rr_series(fn) for fn in files]
            rr_series = xr.concat(series, pd.Index(cat_ids, name=STATIONS_DIM_NAME))
            sites = data_load(self.sites_csv)
            siteid = [str(x) for x in sites.siteid.values]
            assert set(siteid) == set(rr_series.station.values)
            sites.set_index("siteid", inplace=True)
            xarray_sids = [x for x in rr_series.station.values]
            reorder_indx = [siteid.index(s) for s in xarray_sids]
            sites = sites.iloc[reorder_indx]
            rr_series.coords[LAT_VARNAME] = (
                STATIONS_DIM_NAME,
                sites.lat_outlet_corrected,
            )
            rr_series.coords[LON_VARNAME] = (
                STATIONS_DIM_NAME,
                sites.lon_outlet_corrected,
            )
            for site_attrib in sites.columns:
                rr_series.coords[site_attrib] = (
                    STATIONS_DIM_NAME,
                    sites[site_attrib].values,
                )
            # The 'name' value inherited from the CSV files should not be used as coordigate, as it seems to interfere with at least some xr.merge operations.
            rr_series = rr_series.rename({NAME_ATTRIB: STATION_NAME_VARNAME})
            description_fn = self.rr_dir / "description.json"
            with open(description_fn) as fp:
                attrs = json.load(fp)
            attrs[NAME_ATTRIB] = NAME_ATTRIB_VALUE_DAILY
            rr_series.attrs.update(attrs)
            rr_series = rr_series.to_dataset(name=DAILY_SERIES_VARNAME)
            try:
                rr_series.to_netcdf(self.rr_nc, mode="w")
            except Exception as e:
                os.remove(self.rr_nc)
                raise e
        if lazy_loading:
            rr_series = xr.open_dataset(self.rr_nc)
        else:
            rr_series = xr.load_dataarray(self.rr_nc)
        # Add coordinates helping the transformation to monthly and yearly (note also may be handled with resample)
        # rr_month = rr_series.groupby("time.month").sum()
        a = rr_series["time.year"].to_numpy()
        b = rr_series["time.month"].to_numpy()
        year_month_idx = pd.MultiIndex.from_arrays([a, b])
        rr_series.coords["year_month"] = (TIME_DIM_NAME, year_month_idx)
        year_idx = pd.Index(a)
        rr_series.coords["year"] = (TIME_DIM_NAME, year_idx)

        self.rr_series = rr_series
        return self.rr_series

    def get_monthly(self, skipna_tmax=True) -> xr.DataArray:
        assert self.rr_series is not None
        rr_series = self.rr_series
        rain_mm_mth = sum_monthly(rr_series.sel(series_id=RAIN_D_COL))
        evap_mm_mth = sum_monthly(rr_series.sel(series_id=EVAP_D_COL))
        runoff_mm_mth = sum_monthly(rr_series.sel(series_id=RUNOFF_D_COL))
        tmax_mean_mth = mean_monthly(rr_series.sel(series_id=TEMPMAX_D_COL), skipna=skipna_tmax)

        rain_mm_mth = rain_mm_mth.rename("rain_mm_mth").reset_coords(
            SERIES_VARNAME, drop=True
        )
        evap_mm_mth = evap_mm_mth.rename("evap_mm_mth").reset_coords(
            SERIES_VARNAME, drop=True
        )
        runoff_mm_mth = runoff_mm_mth.rename("runoff_mm_mth").reset_coords(
            SERIES_VARNAME, drop=True
        )
        tmax_mean_mth = tmax_mean_mth.rename("tmax_mean_mth").reset_coords(
            SERIES_VARNAME, drop=True
        )
        series_monthly = xr.merge(
            [rain_mm_mth, evap_mm_mth, runoff_mm_mth, tmax_mean_mth]
        )
        series_monthly.attrs.update({NAME_ATTRIB: "OZDATA monthly"})
        return series_monthly

    def get_yearly(self) -> xr.DataArray:
        assert self.rr_series is not None
        rr_series = self.rr_series
        rain_mm_yr = sum_yearly(rr_series.sel(series_id=RAIN_D_COL))
        evap_mm_yr = sum_yearly(rr_series.sel(series_id=EVAP_D_COL))
        runoff_mm_yr = sum_yearly(rr_series.sel(series_id=RUNOFF_D_COL))
        rain_mm_yr = rain_mm_yr.rename("rain_mm_yr").reset_coords(
            SERIES_VARNAME, drop=True
        )
        evap_mm_yr = evap_mm_yr.rename("evap_mm_yr").reset_coords(
            SERIES_VARNAME, drop=True
        )
        runoff_mm_yr = runoff_mm_yr.rename("runoff_mm_yr").reset_coords(
            SERIES_VARNAME, drop=True
        )
        runoff_coeffs_yearly = runoff_mm_yr / rain_mm_yr
        runoff_coeffs_yearly = runoff_coeffs_yearly.rename("runoff_coeffs_yr")
        series_yearly = xr.merge(
            [rain_mm_yr, evap_mm_yr, runoff_mm_yr, runoff_coeffs_yearly]
        )
        series_yearly.attrs.update({NAME_ATTRIB: "OZDATA annual"})
        return series_yearly


def mk_xarray_series(
    data: Union[np.ndarray, pd.Series],
    dim_name: str = None,
    units: str = None,
    time_index: Optional[Union[List, pd.DatetimeIndex]] = None,
    colnames: Optional[List[str]] = None,
    fill_miss_func: Optional[Callable] = None,
) -> xr.DataArray:
    if len(data.shape) > 2:
        raise NotImplementedError("data must be at most of dimension 2")
    if len(data.shape) > 1 and dim_name is None:
        raise NotImplementedError(
            "data has more than one dimension, so the name of the second dimension 'dim_name' must be provided"
        )
    if time_index is None:
        if not isinstance(data, pd.Series):
            raise NotImplementedError(
                "if time_index is None data must be a pandas Series"
            )
        else:
            time_index = data.index
    if colnames is None and len(data.shape) > 1:
        if not isinstance(data, pd.Series):
            raise NotImplementedError(
                "if colnames is None and data of shape 2, data must be a pandas Series"
            )
        else:
            colnames = data.columns
    if fill_miss_func is not None:
        data = fill_miss_func(data)
    if len(data.shape) > 1:
        x = xr.DataArray(
            data, coords=[time_index, colnames], dims=[TIME_DIM_NAME, dim_name]
        )
    else:
        x = xr.DataArray(data, coords=[time_index], dims=[TIME_DIM_NAME])
    if units is not None:
        set_xr_units(x, units)
    return x


LBL_OBS_RO = "observed runoff"
LBL_RAIN = "rainfall"
LBL_ET = "ET demand"
LBL_AWRA_IN = "AWRA balance in"
LBL_AWRA_RO = "AWRA runoff"

EVAP_MM_MONTH_VAR = "evap_mm_mth"
RAIN_MM_MONTH_VAR = "rain_mm_mth"
RUNOFF_MM_MONTH_VAR = "runoff_mm_mth"

EVAP_YR_MONTH_VAR = "evap_mm_yr"
RAIN_YR_MONTH_VAR = "rain_mm_yr"
RUNOFF_YR_MONTH_VAR = "runoff_mm_yr"

RAIN_KEY = "rainfall"
EVAP_KEY = "evaporation"
RUNOFF_KEY = "runoff"
TMAX_MEAN_KEY = "tmax_mean_mth"


def _mth_var_id_for_name(name: str):
    d = {
        EVAP_KEY: EVAP_MM_MONTH_VAR,
        RAIN_KEY: RAIN_MM_MONTH_VAR,
        RUNOFF_KEY: RUNOFF_MM_MONTH_VAR,
    }
    if name in d.keys():
        return d[name]
    else:
        return name


def _yr_var_id_for_name(name: str):
    d = {
        EVAP_KEY: EVAP_YR_MONTH_VAR,
        RAIN_KEY: RAIN_YR_MONTH_VAR,
        RUNOFF_KEY: RUNOFF_YR_MONTH_VAR,
    }
    if name in d.keys():
        return d[name]
    else:
        return name


class MaskTimeSeries:
    def __init__(self, observed_ts: xr.DataArray) -> None:
        self._mask = np.logical_not(np.isnan(observed_ts.values))

    # WARNING: this is not checking anything about consistent time handling bewteen mask and input!
    def mask(self, x: xr.DataArray):
        if isinstance(x, pd.DataFrame) or isinstance(x, pd.Series):
            x = xr_time_series(x)
        return x.where(self._mask)


class OzDataProvider:
    """A high level facade to access daily 'ozdata' and its derived monthly/yearly datasets

    Data set from [Lerat, J., Thyer, M., McInerney, D., Kavetski, D., Woldemeskel, F., Pickett-Heaps, C., Shin, D., & Feikema, P. (2020). A robust approach for calibrating a daily rainfall-runoff model to monthly streamflow data. Journal of Hydrology, 591. https://doi.org/10.1016/j.jhydrol.2020.125129](https://doi.org/10.1016/j.jhydrol.2020.125129)

    """

    def __init__(self, ds: xr.Dataset) -> None:
        self.ds = ds

    def cumsums_daily(self, station_id: str) -> xr.DataArray:
        """Gets the cumulated sums of a subset of daily series :
            * runoff[mm/d]
            * rain[mm/d]
            * evap[mm/d]
            * AWRA_Runoff[mm/d]
            * AWRA_ETactual[mm/d]

        Args:
            station_id (str): station identifier

        Returns:
            xr.DataArray: Daily cumulated sums
        """
        ds = self.ds
        daily = ds["daily_series"]
        one_station = daily.sel(station=station_id)

        ro = one_station.sel(series_id="runoff[mm/d]")
        rainfall = one_station.sel(series_id="rain[mm/d]")
        evap = one_station.sel(series_id="evap[mm/d]")
        awra_ro = one_station.sel(series_id="AWRA_Runoff[mm/d]")
        awra_et_actual = one_station.sel(series_id="AWRA_ETactual[mm/d]")

        rain_in = rainfall
        rain_in.coords[SERIES_VARNAME] = LBL_RAIN
        evap_in = evap
        evap_in.coords[SERIES_VARNAME] = LBL_ET

        # rain_eff_in = rainfall - evap
        # rain_eff_in.coords[SERIES_VARNAME] = "max effective rainfall"
        # rain_eff_in = np.maximum(rain_eff_in, 0.0)

        awra_rain_eff_in = rainfall - awra_et_actual
        awra_rain_eff_in.coords[SERIES_VARNAME] = LBL_AWRA_IN

        # there can be two masks: one is for cumulative calculations on the
        # largest possible observed runoff span. The other is for display.
        # Unfortunately depending on what we want to do, or rather what we display,
        #  there is a balance
        # between using separate masks or not.
        mask_display = np.logical_not(np.isnan(ro.values))
        # mask_calc = get_largest_mask(ro.values)
        mask_calc = mask_display

        rain_in_masked = rain_in.where(mask_calc)
        evap_in_masked = evap_in.where(mask_calc)
        awra_rain_eff_in_masked = awra_rain_eff_in.where(mask_calc)
        awra_ro_masked = awra_ro.where(mask_calc)
        awra_ro_masked.coords[SERIES_VARNAME] = LBL_AWRA_RO

        ro_cs = ro.cumsum(skipna=True)
        ro_cs.coords[SERIES_VARNAME] = LBL_OBS_RO
        rain_in_cs = rain_in_masked.cumsum(skipna=True)
        evap_in_cs = evap_in_masked.cumsum(skipna=True)
        awra_rain_eff_in_cs = awra_rain_eff_in_masked.cumsum(skipna=True)
        awra_ro_cs = awra_ro_masked.cumsum(skipna=True)

        masked_ro_cs = ro_cs.where(mask_display)
        masked_rain_in_cs = rain_in_cs.where(mask_display)
        masked_evap_in_cs = evap_in_cs.where(mask_display)
        masked_awra_rain_eff_in_cs = awra_rain_eff_in_cs.where(mask_display)
        masked_awra_ro_cs = awra_ro_cs.where(mask_display)

        x = xr.concat(
            [
                masked_ro_cs,
                masked_evap_in_cs,
                masked_rain_in_cs,
                masked_awra_rain_eff_in_cs,
                masked_awra_ro_cs,
            ],
            dim=SERIES_VARNAME,
        )
        return x

    def data_for_station(
        self, station_id: str, as_data_provider=False
    ) -> Union[xr.Dataset, "OzDataProvider"]:
        subset_ds = self.ds.sel({STATIONS_DIM_NAME: station_id})
        if not as_data_provider:
            return subset_ds
        else:
            return OzDataProvider(subset_ds)

    def monthly_data(
        self,
        station_id: str = None,
        variable_name: str = RAIN_KEY,
        cf_time: bool = True,
    ):
        d = self.ds if station_id is None else self.data_for_station(station_id)
        varname = _mth_var_id_for_name(variable_name)
        x = d[varname]
        if cf_time:
            x = to_monoindex_series(x)
        return x

    def yearly_data(
        self,
        station_id: str = None,
        variable_name: str = RAIN_KEY,
        cf_time: bool = True,
    ):
        d = self.ds if station_id is None else self.data_for_station(station_id)
        varname = _yr_var_id_for_name(variable_name)
        x = d[varname]
        if cf_time:
            x = to_monoindex_series(x)
        return x

    def not_suspicious(self):
        not_suspicious = np.equal(self.ds.suspicious.values, "")
        not_suspicious_station_ids = self.ds.station[not_suspicious]
        not_sus_data = self.ds.sel({STATIONS_DIM_NAME: not_suspicious_station_ids})
        return OzDataProvider(not_sus_data)


def load_aus_rr_data(
    aus_rr_root_dir: str = None,
    discard_cached_nc: bool = False,
    nc_cache_filename: str = "rainfall_runoff_series.nc",
    lazy_loading: bool = False,
    do_aggregate: bool = True,
) -> OzDataProvider:
    import os

    if aus_rr_root_dir is None:
        if "AUSRR_DATA_DIR" in os.environ:
            aus_rr_root_dir = os.environ["AUSRR_DATA_DIR"]
        else:
            raise ValueError(
                "Aus RR Data directory not specified, and no env. var. AUSRR_DATA_DIR was found"
            )
    root_dir = Path(aus_rr_root_dir)
    if not root_dir.exists():
        raise FileNotFoundError(
            "Data directory with OZDATA not found: {}".format(aus_rr_root_dir)
        )
    rr_nc = root_dir / nc_cache_filename
    loader = OzDataProcessing(root_dir, rr_nc)
    rr_series = loader.load_daily_data(
        discard_cached_nc=discard_cached_nc, lazy_loading=lazy_loading
    )
    if do_aggregate:
        series_monthly = loader.get_monthly()
        series_yearly = loader.get_yearly()
        ds = xr.merge([rr_series, series_monthly, series_yearly])
    else:
        ds = rr_series
    data_repo = OzDataProvider(ds)
    return data_repo
