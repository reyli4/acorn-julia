from glob import glob

import numpy as np
import pandas as pd
import xarray as xr
import salem
import cartopy.crs as ccrs
from scipy.optimize import minimize_scalar
from python.utils import nrel_wtk_path


def read_all_wtk(
    vars_to_keep=["wind_speed", "power"], nrel_wtk_path: str = nrel_wtk_path
):
    """
    Read all WTK data from the given path.
    """
    wtk_files = glob(f"{nrel_wtk_path}/techno-economic/met_data/*.nc")

    df_all = []

    # Loop through all
    for file in wtk_files[::100]:
        # Read
        ds = xr.open_dataset(file)

        # Subset to vars
        ds = ds[vars_to_keep]

        # Decode times
        start_time = pd.to_datetime(ds.attrs["start_time"], unit="s", origin="unix")
        time_index = pd.date_range(
            start=start_time, periods=len(ds["time"]), freq="5min"
        )

        # Convert to dataframe
        df = ds.to_dataframe().reset_index()

        # Decode times
        df["datetime"] = time_index.tz_localize(
            "America/New_York", ambiguous="NaT", nonexistent="NaT"
        )
        df["datetime"] = df["datetime"].dt.tz_convert("UTC")

        # Add lon/lat
        df["wtk_lon"] = ds.attrs["longitude"]
        df["wtk_lat"] = ds.attrs["latitude"]

        # Resample to hourly
        df = df.set_index("datetime").resample("1h").mean()

        # Tidy
        df = df.drop(columns=["time"])
        df = df.rename(columns={"wind_speed": "wtk_ws", "power": "wtk_power"})

        # Append to all
        df_all.append(df)

    return pd.concat(df_all)


def prepare_wind_data(
    climate_paths,
    wind_vars,
    lat_name="lat",
    lon_name="lon",
    curvilinear=False,
    min_lat=39,  # approx NYS
    max_lat=45,  # approx NYS
    min_lon=-80,  # approx NYS
    max_lon=-71,  # approx NYS
):
    """
    Gather input data for solar power generation.
    """
    # Get bounds for NYS
    if curvilinear:
        # Get CRS
        ds_tmp = salem.open_wrf_dataset(climate_paths[0])
        ds_crs = ccrs.Projection(ds_tmp.pyproj_srs)
        # Get bounds
        x_min, y_min = ds_crs.transform_point(
            min_lon, min_lat, src_crs=ccrs.PlateCarree()
        )
        x_max, y_max = ds_crs.transform_point(
            max_lon, max_lat, src_crs=ccrs.PlateCarree()
        )
    else:
        x_min, y_min = min_lon, min_lat
        x_max, y_max = max_lon, max_lat

    # Read climate data
    ds = []
    for file in np.sort(climate_paths):
        # Read WRF output
        ds_tmp = salem.open_wrf_dataset(file)[wind_vars]
        # Subset to NYS for quicker processing
        ds_tmp = ds_tmp.sel(
            {lat_name: slice(y_min, y_max), lon_name: slice(x_min, x_max)}
        )

        # Append
        ds.append(ds_tmp)
    ds = xr.concat(ds, dim="time")

    # Subset to WTK data only
    ds = ds.sel(time=slice("2007-01-01", "2013-12-31"))
    assert len(ds.time) > 0, "No data found"

    # Get lat/lons from NYS solar sites
    df_wtk = read_all_wtk()
    wtk_latlons = df_wtk[["wtk_lat", "wtk_lon"]].value_counts().index.to_numpy()

    # Loop through lat/lons
    df_all = []
    for wtk_lat, wtk_lon in wtk_latlons:
        # Select climate data point
        if curvilinear:
            x, y = ds_crs.transform_point(
                np.round(wtk_lon, 2), np.round(wtk_lat, 2), src_crs=ccrs.PlateCarree()
            )
        else:
            x, y = wtk_lon, wtk_lat

        ds_sel = ds.sel({lon_name: x, lat_name: y}, method="nearest")

        # Take only the wind data
        # df = ds_sel[[wind_vars, "time"]].to_dataframe().reset_index()
        df = ds_sel.to_dataframe().reset_index()

        # Add info
        df["wtk_lat"] = wtk_lat
        df["wtk_lon"] = wtk_lon
        df["datetime"] = pd.to_datetime(df["time"])
        df["datetime"] = df["time"].dt.tz_localize("UTC")
        # df = df.rename(columns={lat_name: "ds_lat", lon_name: "ds_lon"})
        df = df.drop(columns="time")

        # Append
        df_all.append(df)

    # Combine all
    df_all = pd.concat(df_all, ignore_index=True)

    # Merge
    df_all = pd.merge(
        df_all, df_wtk.reset_index(), on=["datetime", "wtk_lat", "wtk_lon"]
    )

    # Drop duplicates
    df_all = (
        df_all.set_index(["wtk_lat", "wtk_lon", "datetime"]).sort_index().reset_index()
    ).drop_duplicates()

    # Add datetime info
    df_all["month"] = df_all["datetime"].dt.month
    df_all["dayofyear"] = df_all["datetime"].dt.dayofyear
    df_all["hour"] = df_all["datetime"].dt.hour

    # Return
    return df_all


def get_stability_coefficients(
    df,
    ws_climate_10m,
    ws_hubheight,
    groupby_cols=["month", "hour"],
):
    """
    Calculates stability coefficients from climate data.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing climate data
    ws_climate : str
        Name of the climate wind speed variable
    ws_hubheight : float
        Hub height of the wind turbine
    lookup_cols : list
        Columns to use for the lookup table
    """

    # Objective function
    def _objective(alpha, df, ws_climate, ws_hubheight):
        # Inferred hubheight windspeed
        df["ws_climate_hubheight"] = df[ws_climate_10m] * (100 / 10) ** alpha

        # Calculate RMSE
        rmse = np.sqrt(np.mean((df["ws_climate_hubheight"] - df[ws_hubheight]) ** 2))

        # Return
        return rmse

    # Optimize for each group
    res = df.groupby(groupby_cols).apply(
        lambda x: minimize_scalar(
            _objective,
            bounds=(0.01, 1.0),
            args=(x, ws_climate_10m, ws_hubheight),
            method="bounded",
        ).x
    )

    # Return
    return pd.DataFrame(res).rename(columns={0: "alpha"})
