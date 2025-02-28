import salem
import geopandas as gpd
import pandas as pd

from python_utils import project_path


def tgw_to_zones(
    tgw_file_path: str,
    tgw_vars: list[str],
    nyiso_zone_shp_path: str = f"{project_path}/data/nyiso/shapefiles/NYISO_Load_Zone_Dissolved.shp",
) -> pd.DataFrame:
    """
    Converts TGW output variables to NYISO load zone (weighted) averages.

    Code based on: https://github.com/IMMM-SFA/im3components/blob/main/im3components/wrf_to_tell/wrf_tell_counties.py
    """

    # Read TGW
    tgw = salem.open_wrf_dataset(tgw_file_path)
    tgw_crs = tgw.pyproj_srs

    # Read NYISO zones
    gdf = gpd.read_file(nyiso_zone_shp_path).to_crs(tgw_crs)

    # Loop through times
    out = []
    for itime in range(len(tgw["time"])):
        # Convert TGW to dataframe
        tgw_df = tgw[tgw_vars].isel(time=itime).to_dataframe().reset_index(drop=True)
        tgw_df = gpd.GeoDataFrame(tgw_df, geometry=tgw.salem.grid.to_geometry().geometry).set_crs(
            tgw_crs
        )
        tgw_df["cell_index"] = tgw_df.index.values

        # Get intersection
        intersection = gpd.overlay(gdf, tgw_df, how="intersection")

        # Area weighting
        intersection["area"] = intersection.area
        intersection["weight"] = intersection["area"] / intersection[["ZONE", "area"]].groupby(
            "ZONE"
        ).area.transform("sum")

        # Compute average
        out.append(
            intersection[tgw_vars]
            .multiply(intersection["weight"], axis="index")
            .join(intersection[["ZONE", "time"]])
            .groupby(["ZONE", "time"])
            .sum()
        )

    return pd.concat(out)
