"""Define constants for conventions for netCDF names, etc.
"""

# There are inspired by the swift netCDF conventions, and CF (climate and forecasting) conventions

STATIONS_DIM_NAME = "station"
STATION_ID_VARNAME = "station_id"
LEAD_TIME_DIM_NAME = "lead_time"
SERIES_VARNAME = "series_id"
TIME_DIM_NAME = "time"
TIME_DIMNAME = TIME_DIM_NAME

MONTH_TIME_DIM_NAME = "month_time"

NAME_ATTRIB = "name"

NAME_ATTRIB_VALUE_DAILY = "OZDATA daily"

DAILY_SERIES_VARNAME = "daily_series"

# char station_name[str_len,station]
STATION_NAME_VARNAME = "station_name"
# float lat[station]
LAT_VARNAME = "lat"
# float lon[station]
LON_VARNAME = "lon"
# float x[station]
X_VARNAME = "x"
# float y[station]
Y_VARNAME = "y"
# float area[station]
AREA_VARNAME = "area"
# float elevation[station]
ELEVATION_VARNAME = "elevation"

PERIOD_DIM_NAME = "period"
PERIOD_ID_VARNAME = "period_id"
MODEL_DIM_NAME = "model"
METRIC_DIM_NAME = "metric"

OBSERVED_SERIES_LABEL = "Observed"
MODELLED_SERIES_LABEL = "Modelled"

TRAINING_LABEL = "Training"
TESTING_LABEL = "Testing"

# Variable names found in the source data set by Lerat et al.
EVAP_D_COL = "evap[mm/d]"
AWRA_SS_D_COL = "AWRA_SS[mm]"
TEMPMAX_D_COL = "TEMPmax[C]"
AWRA_S0_D_COL = "AWRA_S0[mm]"
TEMPMIN_D_COL = "TEMPmin[C]"
AWRA_ETACTUAL_D_COL = "AWRA_ETactual[mm/d]"
RAIN_D_COL = "rain[mm/d]"
AWRA_RUNOFF_D_COL = "AWRA_Runoff[mm/d]"
STREAMFLOW_D_COL = "streamflow[ML/d]"
RUNOFF_LINEAR_FLAG_D_COL = "runoff_linear_flag"
RUNOFF_D_COL = "runoff[mm/d]"


XR_UNITS_ATTRIB_ID: str = "units"
"""key for the units attribute on xarray DataArray objects"""
