#!/usr/bin/env julia

"""
Simple test script to verify seasonal storage implementation
Tests the seasonal storage functionality with the low_RE_high_elec_iter0 scenario
"""

# Add project path
push!(LOAD_PATH, "/workspace/src/julia")

# Include the ACORN model
include("/workspace/src/julia/utils.jl")
include("/workspace/src/julia/acorn.jl")

println("="^60)
println("TESTING SEASONAL STORAGE IMPLEMENTATION")
println("="^60)

# Test parameters
run_name = "low_RE_high_elec_iter0"
climate_scenario = "historical_1980_2019"
sim_year = 2019
branchprop_name = "boyuan"
busprop_name = "boyuan"
if_lim_name = "boyuan"
save_name = "seasonal_storage_test"

println("Test parameters:")
println("  Run name: $run_name")
println("  Climate scenario: $climate_scenario")
println("  Simulation year: $sim_year")
println("  Save name: $save_name")
println()

# Check if seasonal storage assignment file exists
seasonal_storage_path = "/workspace/runs/$run_name/inputs/seasonal_storage_assignment.csv"
if isfile(seasonal_storage_path)
    println("✅ Seasonal storage assignment file found: $seasonal_storage_path")
    
    # Read and display the seasonal storage data
    using CSV, DataFrames
    seasonal_storage = CSV.read(seasonal_storage_path, DataFrame)
    println("\nSeasonal storage configuration:")
    println(seasonal_storage)
    println("\nTotal seasonal storage units: $(nrow(seasonal_storage))")
    println("Total power capacity: $(sum(seasonal_storage.charge_capacity_MW)) MW")
    println("Total energy capacity: $(sum(seasonal_storage.storage_capacity_mwh)) MWh")
else
    println("❌ Seasonal storage assignment file not found: $seasonal_storage_path")
    println("Please create the seasonal storage assignment file first.")
    exit(1)
end

println("\n" * "="^60)
println("RUNNING ACORN WITH SEASONAL STORAGE")
println("="^60)

try
    # Run ACORN with seasonal storage
    println("Starting ACORN simulation with seasonal storage...")
    
    run_acorn(
        run_name,
        climate_scenario,
        sim_year,
        branchprop_name,
        busprop_name,
        if_lim_name,
        save_name;
        exclude_external_zones = true,
        include_new_hvdc = false,
        storage_eff = 0.75,
        seasonal_storage_eff = 0.95
    )
    
    println("✅ ACORN simulation completed successfully!")
    
    # Check if seasonal storage output files were created
    output_dir = "/workspace/runs/$run_name/outputs/$climate_scenario/$save_name"
    
    seasonal_charge_file = "$output_dir/seasonal_charge_$sim_year.csv"
    seasonal_discharge_file = "$output_dir/seasonal_discharge_$sim_year.csv"
    seasonal_batt_state_file = "$output_dir/seasonal_batt_state_$sim_year.csv"
    
    if isfile(seasonal_charge_file)
        println("✅ Seasonal charge output created: $seasonal_charge_file")
    else
        println("❌ Seasonal charge output not found: $seasonal_charge_file")
    end
    
    if isfile(seasonal_discharge_file)
        println("✅ Seasonal discharge output created: $seasonal_discharge_file")
    else
        println("❌ Seasonal discharge output not found: $seasonal_discharge_file")
    end
    
    if isfile(seasonal_batt_state_file)
        println("✅ Seasonal battery state output created: $seasonal_batt_state_file")
    else
        println("❌ Seasonal battery state output not found: $seasonal_batt_state_file")
    end
    
    println("\n" * "="^60)
    println("SEASONAL STORAGE TEST COMPLETED SUCCESSFULLY!")
    println("="^60)
    println("Next steps:")
    println("1. Analyze the seasonal storage outputs")
    println("2. Compare with baseline scenario (no seasonal storage)")
    println("3. Run portfolio sweep analysis")
    println("4. Calculate marginal benefits by zone")
    
catch e
    println("❌ Error running ACORN with seasonal storage:")
    println("Error: $e")
    println("\nPlease check:")
    println("1. All required input files exist")
    println("2. Seasonal storage assignment file is properly formatted")
    println("3. ACORN model is correctly implemented")
end