from glob import glob

import cartopy.crs as ccrs
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import salem
import xarray as xr
from scipy.optimize import minimize_scalar

from python.utils import nrel_sind_path, project_path, zone_names


def read_all_sind():
    """
    Read all SIND data from the given path.
    """
    sind_files = glob(f"{nrel_sind_path}/ny-pv-2006/Actual_*.csv")

    df_all = []

    # Loop through all
    for file in sind_files:
        # Read
        df = pd.read_csv(file)

        # Convert to UTC
        df["datetime"] = pd.to_datetime(df["LocalTime"], format="%m/%d/%y %H:%M")
        df["datetime"] = df["datetime"].dt.tz_localize(
            "America/New_York", ambiguous="NaT", nonexistent="NaT"
        )
        df["datetime"] = df["datetime"].dt.tz_convert("UTC")

        # Resample to hourly
        df = df.set_index("datetime")
        df = df.resample("h").mean(numeric_only=True)

        # Add lat/lon
        lat, lon = file.split("_")[1], file.split("_")[2]
        df["sind_lat"] = float(lat)
        df["sind_lon"] = float(lon)

        # Add system type
        df["solar_type"] = file.split("_")[4]

        # Fix naming
        df = df.rename(columns={"Power(MW)": "actual_power_MW"})

        # Normalize
        power_rating = float(file.split("_")[5].replace("MW", ""))
        df["actual_power_norm"] = df["actual_power_MW"] / power_rating

        # Append to all
        df_all.append(df)

    return pd.concat(df_all)


def calculate_solar_power(
    Geff,  # Incident solar radiation (W/m²)
    Ta,  # Ambient air temperature (°C)
    Pg_star=1.0,  # Rated capacity (MW) - set to 1 MW in the paper
    G_star=1000.0,  # Reference solar radiation (W/m²) - standard value
    Tc_star=25.0,  # Reference cell temperature (°C) - standard value
    beta=0.45,  # Temperature loss coefficient (%/°C) - from the paper
    NOCT=46.0,  # Nominal operating cell temperature (°C) - from the paper
):
    """
    Calculate solar power generation based on Perpiñan et al. model.
    Reference: https://onlinelibrary.wiley.com/doi/abs/10.1002/pip.728
    See also:

    Parameters:
    -----------
    Geff : float or array
        Incident solar radiation (W/m²)
    Ta : float or array
        Ambient air temperature (°C)
    Pg_star : float, optional
        Rated capacity (MW), default is 1.0 MW
    G_star : float, optional
        Reference solar radiation (W/m²), default is 1000 W/m²
    Tc_star : float, optional
        Reference cell temperature (°C), default is 25°C
    beta : float, optional
        Temperature loss coefficient (%/°C), default is 0.45%/°C
    NOCT : float, optional
        Nominal operating cell temperature (°C), default is 46°C

    Returns:
    --------
    P_DC : float or array
        Generated solar power (MW)
    """
    # Calculate the conversion parameter CT (Equation 13)
    CT = (NOCT - 20) / (0.8 * G_star)

    # Calculate cell temperature (Equation 12)
    Tc = Ta + CT * Geff

    # Calculate efficiency ratio (Equation 11)
    efficiency_ratio = 1 - (beta / 100) * (Tc - Tc_star)

    # Calculate DC power output (Equation 10)
    P_DC = Pg_star * (Geff / G_star) * efficiency_ratio

    return P_DC


def prepare_solar_data(
    climate_path,
    temperature_var,
    shortwave_var,
    sind_site_type=None,
    lat_name="lat",
    lon_name="lon",
    curvilinear=False,
):
    """
    Gather input data for solar power generation.
    """
    # Read climate data
    ds = []
    for file in np.sort(glob(climate_path)):
        ds.append(salem.open_wrf_dataset(file)[[temperature_var, shortwave_var]])
    ds = xr.concat(ds, dim="time")

    # Subset to 2006 only
    ds = ds.sel(time=slice("2006-01-01", "2006-12-31"))
    assert len(ds.time) > 0, f"No 2006 data found for {climate_path}"

    # Get lat/lons from NYS solar sites
    df_sind = read_all_sind()
    sind_latlons = df_sind[["sind_lat", "sind_lon"]].value_counts().index.to_numpy()

    # Get CRS
    if curvilinear:
        ds_crs = ccrs.Projection(ds.pyproj_srs)

    # Loop through lat/lons
    df_all = []
    for sind_lat, sind_lon in sind_latlons:
        # Select climate data point
        if curvilinear:
            x, y = ds_crs.transform_point(
                np.round(sind_lon, 2), np.round(sind_lat, 2), src_crs=ccrs.PlateCarree()
            )
        else:
            x, y = sind_lon, sind_lat

        ds_sel = ds.sel({lon_name: x, lat_name: y}, method="nearest")

        # Take only the shortwave and temperature data
        df = (
            ds_sel[[shortwave_var, temperature_var, "time"]]
            .to_dataframe()
            .reset_index()
        )

        # Add info
        df["sind_lat"] = sind_lat
        df["sind_lon"] = sind_lon
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
        df_all, df_sind.reset_index(), on=["datetime", "sind_lat", "sind_lon"]
    )

    # Subset to site type
    if sind_site_type is not None:
        df_all = df_all[df_all["solar_type"] == sind_site_type]

    # Drop duplicates
    df_all = (
        df_all.set_index(["sind_lat", "sind_lon", "datetime"])
        .sort_index()
        .reset_index()
    ).drop_duplicates()

    # Add datetime info
    df_all["month"] = df_all["datetime"].dt.month
    df_all["dayofyear"] = df_all["datetime"].dt.dayofyear
    df_all["hour"] = df_all["datetime"].dt.hour

    # Return
    return df_all


def get_solar_correction_factors(
    df,
    temperature_var,
    shortwave_var,
    beta,
    lookup_cols=["dayofyear", "hour"],
):
    """
    Calculates solar power correction factors from climate data.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing climate data
    lookup_cols : list
        Columns to use for the lookup table
    """
    # Calculate solar power
    df["sim_power_norm"] = calculate_solar_power(
        df[shortwave_var], df[temperature_var], beta=beta
    )

    # Create lookup table: average bias by doy and hour
    df["bias"] = df["actual_power_norm"] - df["sim_power_norm"]
    correction_lookup = (
        df.groupby(lookup_cols)["bias"].mean().to_frame(name="bias_correction")
    )

    # Apply correction
    df = pd.merge(df, correction_lookup.reset_index(), on=lookup_cols)
    df["sim_power_norm_corrected"] = df["sim_power_norm"] + df["bias_correction"]
    # Set negative values to zero
    df["sim_power_norm_corrected"] = df["sim_power_norm_corrected"].clip(lower=0.0)

    # Return
    return df


def optimize_beta(
    df, temperature_var, shortwave_var, lookup_cols=["dayofyear", "hour"]
):
    """
    Objective function for the beta optimization.
    """

    # Objective function
    def _objective(beta, df, temperature_var, shortwave_var, lookup_cols):
        # Calculate solar power (with bias correction if specified)
        if lookup_cols is not None:
            df = get_solar_correction_factors(
                df, temperature_var, shortwave_var, beta, lookup_cols=lookup_cols
            )
        else:
            df["sim_power_norm_corrected"] = calculate_solar_power(
                df[shortwave_var], df[temperature_var], beta=beta
            )

        # Calculate RMSE
        rmse = np.sqrt(
            np.mean((df["sim_power_norm_corrected"] - df["actual_power_norm"]) ** 2)
        )

        # Return
        return rmse

    # Optimize
    res = minimize_scalar(
        _objective,
        bounds=(0.01, 5.0),
        args=(df, temperature_var, shortwave_var, lookup_cols),
        method="bounded",
    )

    # Return
    return res.x


def merge_to_zones(
    df,
    nyiso_zone_shp_path: str = f"{project_path}/data/nyiso/shapefiles/NYISO_Load_Zone_Dissolved.shp",
):
    """
    Merge the dataframe to the NYISO zones.
    """
    # Read NYISO zones
    nyiso_gdf = gpd.read_file(nyiso_zone_shp_path)

    # Merge
    df_gdf = gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df.sind_lon, df.sind_lat), crs="EPSG:4326"
    )

    # Merge
    df_gdf = gpd.sjoin(df_gdf, nyiso_gdf, how="inner", predicate="within")

    # Return
    return df_gdf


def plot_solar_correction_fit(
    df, x_col, y_col, x_name=None, y_name=None, daily=False, zonal=False
):
    """
    Plot the solar correction fit.
    """
    # For nicer plotting
    months = {
        1: "Jan",
        2: "Feb",
        3: "Mar",
        4: "Apr",
        5: "May",
        6: "Jun",
        7: "Jul",
        8: "Aug",
        9: "Sep",
        10: "Oct",
        11: "Nov",
        12: "Dec",
    }

    # Daily averages if selected
    if daily:
        df = (
            df.groupby([df["datetime"].dt.date, "sind_lat", "sind_lon"])
            .mean(numeric_only=True)
            .reset_index()
        )

    # Zonal averages if selected
    if zonal:
        df = merge_to_zones(df)

    # Loop through counter variables
    if zonal:
        counter_var = "ZONE"
    else:
        counter_var = "month"

    # Plot
    if zonal:
        n_zones = len(df[counter_var].unique())
        n_cols = 3
        n_rows = int(np.ceil(n_zones / n_cols))
    else:
        n_cols = 3
        n_rows = 4
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(10, n_rows * 2.5))
    axs = axs.flatten()

    # Loop through counter variables
    for idc, counter in enumerate(df[counter_var].unique()):
        df_counter = df[df[counter_var] == counter]
        df_counter.plot(
            y=y_col,
            x=x_col,
            kind="scatter",
            s=3,
            ax=axs[idc],
            alpha=0.5,
        )
        # Add fit info
        r2 = (
            np.corrcoef(df_counter.dropna()[x_col], df_counter.dropna()[y_col])[0, 1]
            ** 2
        )
        rmse = np.sqrt(
            np.mean((df_counter.dropna()[x_col] - df_counter.dropna()[y_col]) ** 2)
        )
        # Add 1:1 line
        axs[idc].plot(
            [0, 1], [0, 1], transform=axs[idc].transAxes, ls="--", color="black"
        )
        # Tidy
        if zonal:
            counter_names = zone_names
        else:
            counter_names = months

        axs[idc].set_title(
            f"{counter_names[counter]} (R$^2$: {r2:.2f}, RMSE: {rmse:.2f})"
        )
        axs[idc].grid(alpha=0.5)
        axs[idc].set_xlabel("")
        axs[idc].set_ylabel("")

    fig.supxlabel(x_name if x_name is not None else x_col)
    fig.supylabel(y_name if y_name is not None else y_col)

    plt.tight_layout()
    plt.show()
