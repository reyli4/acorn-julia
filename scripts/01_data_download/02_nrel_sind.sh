#!/bin/bash

###############################################
# Specify output dir
OUT_DIR="/home/fs01/dcl257/data/nrel-sind"
###############################################

# Download
wget https://www.nrel.gov/grid/assets/downloads/ny-pv-2006.zip -P $OUT_DIR

# Unzip
unzip $OUT_DIR/ny-pv-2006.zip -d $OUT_DIR/ny-pv-2006