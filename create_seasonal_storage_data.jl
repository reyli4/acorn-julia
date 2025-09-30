#!/usr/bin/env julia

"""
Create seasonal storage assignment files for ACORN model
Based on professor's notes: focus on zones A, B, D (near renewable resources)
"""

using CSV
using DataFrames

# Project paths
project_path = abspath(joinpath(@__DIR__))
data_path = joinpath(project_path, "data")
runs_path = joinpath(project_path, "runs")

function create_seasonal_storage_assignment()
    """
    Create seasonal storage assignment files for zones A, B, D
    Based on professor's notes: "usually zone near renewable"
    """
    
    # Read bus properties to get zone mapping
    bus_prop_path = joinpath(data_path, "grid", "bus_prop_boyuan.csv")
    bus_prop = CSV.read(bus_prop_path, DataFrame)
    
    # Focus zones as mentioned in professor's notes
    focus_zones = ["A", "B", "D"]
    
    # Get buses in focus zones
    zone_buses = Dict()
    for zone in focus_zones
        zone_buses[zone] = bus_prop[bus_prop.ZONE .== zone, :BUS_I]
        println("Zone $zone: $(length(zone_buses[zone])) buses")
        println("  Bus IDs: $(zone_buses[zone][1:min(5, length(zone_buses[zone]))])...")  # Show first 5
    end
    
    # Create seasonal storage assignments for each zone
    all_seasonal_storage = DataFrame()
    
    for zone in focus_zones
        println("\nCreating seasonal storage for Zone $zone...")
        
        # Select representative buses in each zone (not all buses need storage)
        zone_bus_list = zone_buses[zone]
        
        # For seasonal storage, we want fewer, larger installations
        # Select 2-3 representative buses per zone
        if length(zone_bus_list) >= 3
            selected_buses = zone_bus_list[1:3:end][1:3]  # Take 3 evenly spaced
        else
            selected_buses = zone_bus_list[1:min(2, length(zone_bus_list))]  # Take first 2 if less than 3
        end
        
        println("  Selected buses: $selected_buses")
        
        # Seasonal storage characteristics (much larger than regular storage)
        seasonal_storage_data = DataFrame()
        
        for (i, bus_id) in enumerate(selected_buses)
            # Seasonal storage has different characteristics:
            # - Much larger energy capacity (seasonal scale)
            # - Lower power-to-energy ratio (weeks to months)
            # - Higher efficiency (95% vs 75%)
            
            # Power capacity (MW) - charging/discharging rate
            if zone == "A"  # Zone A gets larger capacity
                charge_capacity_MW = 500 + (i-1) * 200  # 500, 700, 900 MW
            elseif zone == "B"
                charge_capacity_MW = 300 + (i-1) * 150  # 300, 450, 600 MW  
            else  # Zone D
                charge_capacity_MW = 400 + (i-1) * 100  # 400, 500, 600 MW
            end
            
            # Energy capacity (MWh) - much larger for seasonal storage
            # Power-to-energy ratio of ~1 week (168 hours) for seasonal storage
            storage_capacity_mwh = charge_capacity_MW * 168  # 1 week at full power
            
            # Add to DataFrame
            new_row = DataFrame(
                bus_id = bus_id,
                charge_capacity_MW = charge_capacity_MW,
                storage_capacity_mwh = storage_capacity_mwh
            )
            
            if isempty(seasonal_storage_data)
                seasonal_storage_data = new_row
            else
                seasonal_storage_data = vcat(seasonal_storage_data, new_row)
            end
            
            println("    Bus $bus_id: $charge_capacity_MW MW, $storage_capacity_mwh MWh")
        end
        
        # Save zone-specific file
        run_dir = joinpath(runs_path, "low_RE_high_elec_iter0")
        inputs_dir = joinpath(run_dir, "inputs")
        mkpath(inputs_dir)
        
        zone_file = joinpath(inputs_dir, "seasonal_storage_assignment_zone_$(zone).csv")
        CSV.write(zone_file, seasonal_storage_data)
        println("  Saved: $zone_file")
        
        # Add to combined DataFrame
        if isempty(all_seasonal_storage)
            all_seasonal_storage = seasonal_storage_data
        else
            all_seasonal_storage = vcat(all_seasonal_storage, seasonal_storage_data)
        end
    end
    
    # Save combined seasonal storage assignment
    run_dir = joinpath(runs_path, "low_RE_high_elec_iter0")
    inputs_dir = joinpath(run_dir, "inputs")
    combined_output = joinpath(inputs_dir, "seasonal_storage_assignment.csv")
    CSV.write(combined_output, all_seasonal_storage)
    println("\nCombined seasonal storage saved: $combined_output")
    println("Total seasonal storage units: $(nrow(all_seasonal_storage))")
    println("Total power capacity: $(sum(all_seasonal_storage.charge_capacity_MW)) MW")
    println("Total energy capacity: $(sum(all_seasonal_storage.storage_capacity_mwh)) MWh")
    
    return all_seasonal_storage
end

function create_zone_specific_assignments()
    """
    Create zone-specific seasonal storage assignments for portfolio analysis
    """
    println("\n" * "="^50)
    println("CREATING ZONE-SPECIFIC SEASONAL STORAGE ASSIGNMENTS")
    println("="^50)
    
    # Read bus properties
    bus_prop_path = joinpath(data_path, "grid", "bus_prop_boyuan.csv")
    bus_prop = CSV.read(bus_prop_path, DataFrame)
    
    focus_zones = ["A", "B", "D"]
    
    for zone in focus_zones
        println("\nCreating seasonal storage assignment for Zone $zone only...")
        
        # Get buses in this zone
        zone_buses = bus_prop[bus_prop.ZONE .== zone, :BUS_I]
        
        # Select 2-3 representative buses
        if length(zone_buses) >= 3
            selected_buses = zone_buses[1:3:end][1:3]
        else
            selected_buses = zone_buses[1:min(2, length(zone_buses))]
        end
        
        # Create seasonal storage data for this zone only
        seasonal_storage_data = DataFrame()
        
        for (i, bus_id) in enumerate(selected_buses)
            # Zone-specific capacity
            if zone == "A"
                charge_capacity_MW = 800 + (i-1) * 200  # Larger for Zone A
            elseif zone == "B"
                charge_capacity_MW = 500 + (i-1) * 150
            else  # Zone D
                charge_capacity_MW = 600 + (i-1) * 100
            
            storage_capacity_mwh = charge_capacity_MW * 168  # 1 week
            
            new_row = DataFrame(
                bus_id = bus_id,
                charge_capacity_MW = charge_capacity_MW,
                storage_capacity_mwh = storage_capacity_mwh
            )
            
            if isempty(seasonal_storage_data)
                seasonal_storage_data = new_row
            else
                seasonal_storage_data = vcat(seasonal_storage_data, new_row)
            end
        end
        
        # Save zone-specific file
        run_dir = joinpath(runs_path, "low_RE_high_elec_iter0")
        inputs_dir = joinpath(run_dir, "inputs")
        
        zone_file = joinpath(inputs_dir, "seasonal_storage_assignment_zone_$(zone)_only.csv")
        CSV.write(zone_file, seasonal_storage_data)
        println("  Zone $zone seasonal storage: $zone_file")
        println("  Capacity: $(sum(seasonal_storage_data.charge_capacity_MW)) MW, $(sum(seasonal_storage_data.storage_capacity_mwh)) MWh")
    end
end

# Main execution
println("SEASONAL STORAGE DATA GENERATION")
println("Based on professor's notes: focus on zones A, B, D")
println("="^50)

# Create main seasonal storage assignment
combined_df = create_seasonal_storage_assignment()

# Create zone-specific assignments for portfolio analysis
create_zone_specific_assignments()

println("\n" * "="^50)
println("SEASONAL STORAGE DATA GENERATION COMPLETE")
println("="^50)
println("Files created:")
println("  - seasonal_storage_assignment.csv (all zones)")
println("  - seasonal_storage_assignment_zone_A_only.csv")
println("  - seasonal_storage_assignment_zone_B_only.csv") 
println("  - seasonal_storage_assignment_zone_D_only.csv")
println("\nNext steps:")
println("1. Run ACORN with seasonal storage enabled")
println("2. Compare baseline vs seasonal storage scenarios")
println("3. Analyze marginal benefits by zone")