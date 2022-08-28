from typing import List, Optional, Sequence
from .conventions import (
    MODELLED_SERIES_LABEL,
    MONTH_TIME_DIM_NAME,
    OBSERVED_SERIES_LABEL,
    SERIES_VARNAME,
    STATIONS_DIM_NAME,
    TIME_DIM_NAME,
)
import pandas as pd
import xarray as xr


def xr_time_series(rr_series: pd.DataFrame) -> xr.DataArray:
    """Creates an xarray time series out of a data frame

    Args:
        rr_series (pd.DataFrame): data frame where one of the columns is named 'time', with values that can be used to construct a pd.Timestamp

    Returns:
        xr.DataArray: a uni or multi-variate time series

    Example:
        >>> x = pd.DataFrame.from_dict({'time': ['2000-01-01', '2001-01-01'], 'rain': [0.0, 1.2], 'evap': [0.9, 0.2]})
        >>> xr_time_series(x)
        <xarray.DataArray (time: 2, series_id: 2)>
        array([[0. , 0.9],
            [1.2, 0.2]])
        Coordinates:
        * time       (time) datetime64[ns] 2000-01-01 2001-01-01
        * series_id  (series_id) object 'rain' 'evap'
    """
    assert TIME_DIM_NAME in rr_series.columns
    indx = [pd.Timestamp(x) for x in rr_series[TIME_DIM_NAME]]
    rr_series = rr_series.drop([TIME_DIM_NAME], axis=1)
    res = xr.DataArray(
        rr_series.values,
        coords={
            TIME_DIM_NAME: pd.DatetimeIndex(indx),
            SERIES_VARNAME: rr_series.columns,
        },
        dims=[TIME_DIM_NAME, SERIES_VARNAME],
    )
    return res


def to_xarray_ts(
    list_series: Sequence[pd.DataFrame], station_ids: Sequence[str]
) -> xr.DataArray:
    """Create a multidimensional time series, multi-variate and multiple gauging stations

    Args:
        list_series (Sequence[pd.DataFrame]): list of pandas time series data frames with a 'time' column
        station_ids (Sequence[str]): list of gauging stations identifiers

    Returns:
        xr.DataArray: multi-dim array of time series

    Example:
        >>> def f(): return pd.DataFrame.from_dict({'time': ['2000-01-01', '2001-01-01'], 'rain': [0.0, 1.2], 'evap': [0.9, 0.2]})
        ...
        >>> f()
                time  rain  evap
        0  2000-01-01   0.0   0.9
        1  2001-01-01   1.2   0.2
        >>> a = f()
        >>> b = f()
        >>> to_xarray_ts([a, b], ['123456', '234567'])
        <xarray.DataArray (station: 2, time: 2, series_id: 2)>
        array([[[0. , 0.9],
                [1.2, 0.2]],

            [[0. , 0.9],
                [1.2, 0.2]]])
        Coordinates:
        * time       (time) datetime64[ns] 2000-01-01 2001-01-01
        * series_id  (series_id) object 'rain' 'evap'
        * station    (station) object '123456' '234567'
    """
    wpb_xr = [xr_time_series(x) for x in list_series]
    rr_series = xr.concat(wpb_xr, pd.Index(station_ids, name=STATIONS_DIM_NAME))
    return rr_series


def as_ts_df(
    series: pd.DataFrame, colnames: List[str] = None, new_colnames: List[str] = None
) -> pd.DataFrame:
    """transform a data frame, or subset of it, to a multivariate time series represented as a data frame

    Args:
        series (pd.DataFrame): input data frame with a column 'time'
        colnames (List[str], optional): Names of the columns to use for time series variates. Defaults to None, in which case expects columns 'obs' and 'pred' present.
        new_colnames (List[str], optional): Column names of the output dataframe. Same length as previous argument colnames. Defaults to None, in which case defaults are used for observation / modelled identification.

    Returns:
        pd.DataFrame: A multivariate dataframe time series indexed by 'time'
    """
    assert TIME_DIM_NAME in series.columns
    if colnames is None:
        colnames = ["obs", "pred"]
    if new_colnames is None:
        new_colnames = [OBSERVED_SERIES_LABEL, MODELLED_SERIES_LABEL]
    x = pd.DataFrame(
        series[colnames].values, index=pd.Index([pd.Timestamp(x) for x in series.time])
    )
    x.columns = new_colnames
    return x


def monthly_timedim(
    rr_series: xr.DataArray, resample_timedim: Optional[str] = "M", rename_timedim=False
) -> xr.DataArray:
    """resample an xarray time series to another time step, monthly by default. Aggregates by sum.

    Args:
        rr_series (xr.DataArray): Input time series, presumably daily
        resample_timedim (Optional[str], optional): target time step. Defaults to "M".
        rename_timedim (bool, optional): If true renames the 'time' dimension to 'month_time'. Defaults to False.

    Returns:
        xr.DataArray: resampled time series
    """
    if resample_timedim is not None:
        # below is a "hack", though rather clean, to transform t end of months time stamps for monthly data...
        rr_series_x = rr_series.resample({TIME_DIM_NAME: resample_timedim}).sum(
            skipna=False
        )
    else:
        rr_series_x = rr_series
    if rename_timedim:
        rr_series_x = rr_series_x.rename({TIME_DIM_NAME: MONTH_TIME_DIM_NAME})
    return rr_series_x


def to_monthly_xr(
    sl_df: Sequence[pd.DataFrame], station_ids: Sequence[str]
) -> xr.DataArray:
    """Builds a multivariate and multilocation time series, and resample to monthly

    Args:
        sl_df (Sequence[pd.DataFrame]): list of pandas time series data frames with a 'time' column
        station_ids (Sequence[str]): list of gauging stations identifiers

    Returns:
        xr.DataArray: monthly multivariate and multilocation time series
    """
    sl_xr = to_xarray_ts(sl_df, station_ids)
    rr_series = xr.concat(sl_xr, pd.Index(station_ids, name=STATIONS_DIM_NAME))
    rr_series_sl = monthly_timedim(rr_series)
    rr_series_sl.coords[SERIES_VARNAME] = [OBSERVED_SERIES_LABEL, MODELLED_SERIES_LABEL]
    return rr_series_sl


class SeriesComparison:
    def __init__(self, results_ds: xr.Dataset) -> None:
        self.results_ds = results_ds

    def results_cumsums(
        self,
        station_id: str,
        model_ids: Sequence[str],
        start: pd.Timestamp = None,
        end: pd.Timestamp = None,
    ):
        one_station = self.results_ds.series.sel(station=station_id)
        one_station = one_station.sel(time=slice(start, end))

        def f(x, series_id, model_id):
            return (
                x.sel(series_id=series_id)
                .sel(model=model_id)
                .squeeze(drop=True)
                .reset_coords(drop=True)
            )

        import numpy as np

        assert len(model_ids) > 0
        ro = f(one_station, series_id=OBSERVED_SERIES_LABEL, model_id=model_ids[0])
        mod_series = [
            f(one_station, series_id=MODELLED_SERIES_LABEL, model_id=m)
            for m in model_ids
        ]

        mask_display = np.logical_not(np.isnan(ro.values))
        mask_calc = mask_display

        mod_series_masked = [m.where(mask_calc) for m in mod_series]
        ro_cs = ro.cumsum(skipna=True)
        mod_series_masked_cs = [m.cumsum(skipna=True) for m in mod_series_masked]

        # finally replace the misleading "flats" in cumulated values resulting from NaNs, with NaNs
        masked_ro_cs = ro_cs.where(mask_display)
        masked_cumulated_series = [m.where(mask_display) for m in mod_series_masked_cs]

        x = xr.concat(([masked_ro_cs] + masked_cumulated_series), dim=SERIES_VARNAME)

        x.coords[SERIES_VARNAME] = [OBSERVED_SERIES_LABEL] + model_ids
        return x
