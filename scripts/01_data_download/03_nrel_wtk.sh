#!/bin/bash
#SBATCH --nodes=1                      # Number of nodes
#SBATCH --ntasks-per-node=1            # Number of tasks per node
#SBATCH --mem=3GB                      # Memory per node (adjust as needed)
#SBATCH --time=100:00:00                # Maximum run time (adjust as needed)

echo "Job started on `hostname` at `date`"

############################################
# Save path
cd /home/fs01/dcl257/data/nrel-wtk
############################################

# Download all years
base_url="https://nrel-pds-wtk.s3.amazonaws.com/conus/v1.0.0"

for year in {2007..2013}; do
    url="${base_url}/${year}/wtk_conus_${year}_100m.h5"
    
    echo "Starting download for year ${year}..."
    
    until wget -c --tries=0 --retry-connrefused --timeout=30 --read-timeout=30 --waitretry=30 "$url"
    do
        echo "Download failed for ${year}. Retrying in 60 seconds..."
        sleep 60
    done
    
    echo "Successfully downloaded data for ${year}"
done

echo "All downloads completed!"

echo "Job Ended at `date`"
