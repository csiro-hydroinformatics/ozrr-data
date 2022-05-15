from typing import Optional, Sequence
from ozrr.conventions import (
    MODELLED_SERIES_LABEL,
    MONTH_TIME_DIM_NAME,
    OBSERVED_SERIES_LABEL,
    SERIES_VARNAME,
    STATIONS_DIM_NAME,
    TIME_DIM_NAME,
)
import pandas as pd
import xarray as xr


def xr_time_series(rr_series: pd.DataFrame):
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


def to_xarray_ts(list_series: Sequence[pd.DataFrame], station_ids: Sequence[str]):
    wpb_xr = [xr_time_series(x) for x in list_series]
    rr_series = xr.concat(wpb_xr, pd.Index(station_ids, name=STATIONS_DIM_NAME))
    return rr_series


def as_ts_df(series: pd.DataFrame, colnames=None, new_colnames=None) -> pd.DataFrame:
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
):
    if not resample_timedim is None:
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
