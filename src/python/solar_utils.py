from glob import glob

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import salem
import xarray as xr

from python.utils import nrel_sind_path


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


def get_solar_correction_factors(
    climate_path,
    temperature_var,
    shortwave_var,
    sind_site_type,
    lat_name="lat",
    lon_name="lon",
    curvilinear=False,
    lookup_cols=["dayofyear", "hour"],
):
    """
    Calculates solar power correction factors from climate data.

    Parameters:
    -----------
    climate_path : str
        Path pattern to the climate data
    temperature_var : str
        Name of the temperature variable (must be degC)
    shortwave_var : str
        Name of the shortwave variable (must be W/m2)
    sind_site_type : str
        Type of SIND site
    lat_name : str
        Name of the latitude variable
    lon_name : str
        Name of the longitude variable
    curvilinear : bool
        Whether the grid is curvilinear
    lookup_cols : list
        Columns to use for the lookup table
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

        # Estimate solar generation
        df = (
            calculate_solar_power(ds_sel[shortwave_var], ds_sel[temperature_var])
            .to_dataframe(name="sim_power_norm")
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
    df_all = (
        df_all[df_all["solar_type"] == sind_site_type]
        .set_index(["sind_lat", "sind_lon", "datetime"])
        .sort_index()
        .reset_index()
    )

    # Add datetime info
    df_all["month"] = df_all["datetime"].dt.month
    df_all["dayofyear"] = df_all["datetime"].dt.dayofyear
    df_all["hour"] = df_all["datetime"].dt.hour

    # Create lookup table: average bias by month and hour
    df_all["bias"] = df_all["actual_power_norm"] - df_all["sim_power_norm"]
    correction_lookup = (
        df_all.groupby(lookup_cols)["bias"].mean().to_frame(name="bias_correction")
    )

    # Apply correction
    df_all = pd.merge(df_all, correction_lookup.reset_index(), on=lookup_cols)
    df_all["sim_power_norm_corrected"] = (
        df_all["sim_power_norm"] + df_all["bias_correction"]
    )
    # Set negative values to zero
    df_all["sim_power_norm_corrected"] = df_all["sim_power_norm_corrected"].clip(
        lower=0.0
    )

    # Return
    return df_all


def plot_solar_correction_fit(df, x_col, y_col, x_name=None, y_name=None):
    """
    Plot the solar correction fit.
    """
    # For nicer plotting
    months = [
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
    ]

    # Plot
    fig, axs = plt.subplots(4, 3, figsize=(10, 10))
    axs = axs.flatten()

    # Loop through months
    for idm in range(12):
        month = idm + 1
        df_month = df[df["month"] == month]
        df_month.plot(
            y=y_col,
            x=x_col,
            kind="scatter",
            s=3,
            ax=axs[idm],
            alpha=0.5,
        )
        # Add fit info
        r2 = np.corrcoef(df_month.dropna()[x_col], df_month.dropna()[y_col])[0, 1] ** 2
        rmse = np.sqrt(
            np.mean((df_month.dropna()[x_col] - df_month.dropna()[y_col]) ** 2)
        )
        # Add 1:1 line
        axs[idm].plot(
            [0, 1], [0, 1], transform=axs[idm].transAxes, ls="--", color="black"
        )
        # Tidy
        axs[idm].set_title(f"{months[idm]} (R$^2$: {r2:.2f}, RMSE: {rmse:.2f})")
        axs[idm].grid(alpha=0.5)
        axs[idm].set_xlabel("")
        axs[idm].set_ylabel("")

    fig.supxlabel(x_name if x_name is not None else x_col)
    fig.supylabel(y_name if y_name is not None else y_col)

    plt.tight_layout()
    plt.show()
