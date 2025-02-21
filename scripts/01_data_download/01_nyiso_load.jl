using Downloads
using Dates
using ZipFile
using DataFrames
using CSV
using Dates
using Statistics
using Plots

"""
This script downloads and processes historical load data from NYISO. The data is
downloaded in CSV format and processed to aggregate the data into hourly averages
and store the results in a CSV file. This script is self contained and does not 
require any web-based interactions. 

Assumptions:
- Load zones seem to change over time, from NYC_LongIsland to NYC and LongIsland. We assume
  that the NYC_LongIsland zone can be split into NYC and LongIsland, using a ratio of 2.5:1.
"""
#########################################################
# Output dir
data_folder = "/home/fs01/dcl257/data/nyiso/historical_load"
#########################################################

###########################
## Preliminaries
###########################
# Define project root directory
base_dir = dirname(dirname(@__FILE__))

# Define the base URL
base_url = "http://mis.nyiso.com/public/csv/pal/"

# Create the relevant folders
isdir(data_folder) || mkpath(data_folder)
isdir("$data_folder/zipped") || mkpath("$data_folder/zipped")
isdir("$data_folder/extracted") || mkpath("$data_folder/extracted")
isdir("$data_folder/combined") || mkpath("$data_folder/combined")

###########################
## User-defined functions
###########################
# Unzip a zipped file
function unzip_file(zip_path, extract_path)
    mkpath(extract_path)  # ensure the extraction directory exists
    zarchive = ZipFile.Reader(zip_path)
    for f in zarchive.files
        # Create full path for the file
        full_path = joinpath(extract_path, f.name)
        # Write the file
        write(full_path, read(f))
    end
    close(zarchive)
end

# Function to process and aggregate CSV files
function process_load_file(file_path::String)
    # Define the mapping for zones -> zone numbers
    name_mapping = Dict(
        "WEST" => "A",
        "GENESE" => "B",
        "CENTRL" => "C",
        "NORTH" => "D",
        "MHK VL" => "E",
        "CAPITL" => "F",
        "HUD VL" => "G",
        "MILLWD" => "H",
        "DUNWOD" => "I",
        "N.Y.C." => "J",
        "LONGIL" => "K",
        "N.Y.C._LONGIL" => "NYC_Long_Island",  # Placeholder for handling separately
    )

    # Read the CSV file
    df = CSV.read(file_path, DataFrame)

    # Replace missing values in the Load column with 0.0 before processing
    df[!, "Load"] = coalesce.(df[!, "Load"], 0.0)

    # Parse the "Time Stamp" column
    df[!, "Time Stamp"] = DateTime.(df[!, "Time Stamp"], dateformat"m/d/yyyy HH:MM:SS")
    df[!, "Hourly Time"] = floor.(df[!, "Time Stamp"], Hour)

    # Rename the "Name" column based on the mapping and handle NYC/Long Island split
    transformed_rows = DataFrame(HourlyTime=DateTime[], Zone=String[], Load_MW=Float64[])

    for row in eachrow(df)
        zone_name = row.Name
        load_value = row.Load

        if zone_name == "N.Y.C._LONGIL"
            # Split into two rows with the specified ratio (2.5:1)
            nyc_load = (2.5 / 3.5) * load_value
            longisland_load = (1 / 3.5) * load_value

            push!(transformed_rows, (row."Hourly Time", "J", nyc_load))
            push!(transformed_rows, (row."Hourly Time", "K", longisland_load))
        else
            # Map the name and add it to the transformed rows
            mapped_name = name_mapping[zone_name]
            push!(transformed_rows, (row."Hourly Time", mapped_name, load_value))
        end
    end

    # Calculate hourly averages and group by "Hourly Time" and "Name"
    df_hourly = combine(groupby(transformed_rows, ["HourlyTime", "Zone"]), :Load_MW => mean => :Load_MW)

    return df_hourly
end

###########################
## Download and unzip data
###########################
# Loop over each year and month
for year in 2002:2023
    for month in 1:12
        # Use the first day of the month as the date format for URL (e.g., YYYYMM01)
        date_str = Dates.format(Date(year, month, 1), "yyyymm01")
        file_url = base_url * date_str * "pal_csv.zip"
        zip_file_path = "$data_folder/zipped/$date_str.zip"
        extracted_folder_path = "$data_folder/extracted/"

        # Check if the zip file or extracted folder already exists
        if isfile(zip_file_path) || isdir(extracted_folder_path)
            println("Data for $date_str already exists. Skipping download.")
            continue
        end

        try
            # Download the zip file
            println("Downloading data for $date_str...")
            println(file_url)
            Downloads.download(file_url, zip_file_path)

            # Unzip the file
            println("Unzipping data for $date_str...")
            unzip_file(zip_file_path, extracted_folder_path)
        catch e
            println("Failed to download or unzip data for $date_str: $e")
            # Optionally, remove the zip file if it failed
            isfile(zip_file_path) && rm(zip_file_path)
        end
    end
end

############################
## Combine the data files
############################
combined_df = DataFrame(HourlyTime=DateTime[], Zone=String[], Load_MW=Float64[])

# Loop over each day
for year in 2002:2023
    for month in 1:12
        # Use the first day of the month as the date format for URL (e.g., YYYYMM01)
        folder_str = Dates.format(Date(year, month, 1), "yyyymm01")
        # List all files in the folder
        files = readdir("$data_folder/extracted/$folder_str")
        # Loop through and process each file
        for file in files
            processed_df = process_load_file("$data_folder/extracted/$folder_str/$file")
            combined_df = vcat(combined_df, processed_df)
        end
    end
end

# Store the combined dataframe
combined_file_path = "$data_folder/combined/historical_load.csv"
CSV.write(combined_file_path, combined_df)

#######################################
# Plot the historical loads to check
#####################################
# Load the combined CSV file
df = CSV.read("$data_folder/combined/historical_load.csv", DataFrame)

# Group by each hour and calculate the total load across all zones
df = combine(groupby(df, "HourlyTime"), :Load_MW => (x -> sum(x) / 1000) => :Load_GW)

# Add new columns for day, month, and hour
df[!, "Year"] = year.(df.HourlyTime)
df[!, "Month"] = month.(df.HourlyTime)
df[!, "Day"] = day.(df.HourlyTime)
df[!, "Hour"] = hour.(df.HourlyTime)

# Plot 2016 to compare to Kenji's paper SI
df_2016 = filter!(row -> row.Year == 2016, df)
plot(df_2016.HourlyTime, df_2016.Load_GW, ylabel="Total Load (GW)", linewidth=1, legend=false)

# Calculate percentiles for each day, month, and hour
percentiles = combine(groupby(df, ["Day", "Month", "Hour"]),
    :Load_GW => minimum => :Min_Load_GW,
    :Load_GW => (x -> quantile(x, 0.01)) => :q01_Load_GW,
    :Load_GW => median => :Median_Load_GW,
    :Load_GW => (x -> quantile(x, 0.99)) => :q99_Load_GW,
    :Load_GW => maximum => :Max_Load_GW
)

# Sort
sort!(percentiles, [:Month, :Day, :Hour])

# Plot
plot([percentiles.Min_Load_GW percentiles.Median_Load_GW percentiles.Max_Load_GW],
    label=["Min." "Median" "Max."],
    xlabel="Hour of Year",
    ylabel="Total Load (GW)",
    linewidth=1,
    legend=:topleft)