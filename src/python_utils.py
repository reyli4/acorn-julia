import pandas as pd

#############
# Paths
#############
tgw_path = "/home/shared/vs498_0001/im3_hyperfacets_tgw"
nyiso_path = "/home/fs01/dcl257/data/nyiso"
nrel_sind_path = "/home/fs01/dcl257/data/nrel-sind"
nrel_wtk_path = "/home/fs01/dcl257/data/nrel-wtk"


###################
# Data processing
###################
def _preprocess_tgw(ds):
    # Get filename
    filename = ds.encoding["source"]
    # Generate datetime index
    start_time = pd.to_datetime(filename.split("hourly_")[1][:-3], format="%Y-%m-%d_%H_%M_%S")
    time_index = pd.date_range(start=start_time, periods=len(ds["Times"]), freq="h")
    # Update
    ds["Time"] = time_index
    return ds
