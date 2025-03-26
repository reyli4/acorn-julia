import geopandas as gpd
import pandas as pd
import salem

from python.utils import project_path


def tgw_to_zones(
    tgw_file_path: str,
    tgw_vars: list[str],
    nyiso_zone_shp_path: str = f"{project_path}/data/nyiso/shapefiles/NYISO_Load_Zone_Dissolved.shp",
) -> pd.DataFrame:
    """
    Converts TGW output variables to NYISO load zone (area weighted) averages.

    Code based on: https://github.com/IMMM-SFA/im3components/blob/main/im3components/wrf_to_tell/wrf_tell_counties.py
    """

    # Read TGW
    tgw = salem.open_wrf_dataset(tgw_file_path)[tgw_vars]
    tgw = tgw.where((tgw.lat > 40) & (tgw.lon > -80), drop=True)  # subset NYS

    tgw_crs = tgw.pyproj_srs
    geometry = tgw.salem.grid.to_geometry().geometry

    # Read NYISO zones
    gdf = gpd.read_file(nyiso_zone_shp_path).to_crs(tgw_crs)

    # Get the intersection df
    tgw_df_single = tgw.isel(time=0).to_dataframe().reset_index(drop=True)
    tgw_df_single = gpd.GeoDataFrame(tgw_df_single, geometry=geometry).set_crs(tgw_crs)
    intersection_single = gpd.overlay(gdf, tgw_df_single, how="intersection")

    # Convert TGW to dataframe
    tgw_df = tgw.to_dataframe().reset_index()

    # Merge
    intersection = pd.merge(
        tgw_df[["time", "lat", "lon"] + tgw_vars],
        intersection_single[["ZONE", "lat", "lon", "geometry"]],
        how="inner",
        on=["lat", "lon"],
    )
    intersection = gpd.GeoDataFrame(intersection, geometry=intersection["geometry"])

    # Area weighting
    intersection["area"] = intersection.area
    intersection["weight"] = intersection["area"] / intersection[
        ["ZONE", "time", "area"]
    ].groupby(["ZONE", "time"]).area.transform("sum")

    # Compute area-weighted average
    out = (
        intersection[tgw_vars]
        .multiply(intersection["weight"], axis="index")
        .join(intersection[["ZONE", "time"]])
        .groupby(["ZONE", "time"])
        .sum()
    )

    return out.rename(columns={"ZONE": "zone"})
