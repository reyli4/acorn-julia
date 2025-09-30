#!/usr/bin/env python3
"""
Create seasonal storage assignment files for ACORN model
Based on professor's notes: focus on zones A, B, D (near renewable resources)
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Project paths
project_path = Path(__file__).parent
data_path = project_path / "data"
runs_path = project_path / "runs"

def create_seasonal_storage_assignment():
    """
    Create seasonal storage assignment files for zones A, B, D
    Based on professor's notes: "usually zone near renewable"
    """
    
    # Read bus properties to get zone mapping
    bus_prop_path = data_path / "grid" / "bus_prop_boyuan.csv"
    bus_prop = pd.read_csv(bus_prop_path)
    
    # Focus zones as mentioned in professor's notes
    focus_zones = ["A", "B", "D"]
    
    # Get buses in focus zones
    zone_buses = {}
    for zone in focus_zones:
        zone_buses[zone] = bus_prop[bus_prop["ZONE"] == zone]["BUS_I"].tolist()
        print(f"Zone {zone}: {len(zone_buses[zone])} buses")
        print(f"  Bus IDs: {zone_buses[zone][:5]}...")  # Show first 5
    
    # Create seasonal storage assignments for each zone
    for zone in focus_zones:
        print(f"\nCreating seasonal storage for Zone {zone}...")
        
        # Select representative buses in each zone (not all buses need storage)
        zone_bus_list = zone_buses[zone]
        
        # For seasonal storage, we want fewer, larger installations
        # Select 2-3 representative buses per zone
        if len(zone_bus_list) >= 3:
            selected_buses = zone_bus_list[::len(zone_bus_list)//3][:3]  # Take 3 evenly spaced
        else:
            selected_buses = zone_bus_list[:2]  # Take first 2 if less than 3
        
        print(f"  Selected buses: {selected_buses}")
        
        # Seasonal storage characteristics (much larger than regular storage)
        seasonal_storage_data = []
        
        for i, bus_id in enumerate(selected_buses):
            # Seasonal storage has different characteristics:
            # - Much larger energy capacity (seasonal scale)
            # - Lower power-to-energy ratio (weeks to months)
            # - Higher efficiency (95% vs 75%)
            
            # Power capacity (MW) - charging/discharging rate
            if zone == "A":  # Zone A gets larger capacity
                charge_capacity_MW = 500 + i * 200  # 500, 700, 900 MW
            elif zone == "B":
                charge_capacity_MW = 300 + i * 150  # 300, 450, 600 MW  
            else:  # Zone D
                charge_capacity_MW = 400 + i * 100  # 400, 500, 600 MW
            
            # Energy capacity (MWh) - much larger for seasonal storage
            # Power-to-energy ratio of ~1 week (168 hours) for seasonal storage
            storage_capacity_mwh = charge_capacity_MW * 168  # 1 week at full power
            
            seasonal_storage_data.append({
                'bus_id': bus_id,
                'charge_capacity_MW': charge_capacity_MW,
                'storage_capacity_mwh': storage_capacity_mwh
            })
            
            print(f"    Bus {bus_id}: {charge_capacity_MW} MW, {storage_capacity_mwh} MWh")
        
        # Create DataFrame and save
        seasonal_storage_df = pd.DataFrame(seasonal_storage_data)
        
        # Save to run directory
        run_dir = runs_path / "low_RE_high_elec_iter0"
        run_dir.mkdir(exist_ok=True)
        
        inputs_dir = run_dir / "inputs"
        inputs_dir.mkdir(exist_ok=True)
        
        output_file = inputs_dir / f"seasonal_storage_assignment_zone_{zone}.csv"
        seasonal_storage_df.to_csv(output_file, index=False)
        print(f"  Saved: {output_file}")
        
        # Also create a combined file for all zones
        if zone == focus_zones[0]:  # First zone
            combined_df = seasonal_storage_df.copy()
        else:
            combined_df = pd.concat([combined_df, seasonal_storage_df], ignore_index=True)
    
    # Save combined seasonal storage assignment
    combined_output = runs_path / "low_RE_high_elec_iter0" / "inputs" / "seasonal_storage_assignment.csv"
    combined_df.to_csv(combined_output, index=False)
    print(f"\nCombined seasonal storage saved: {combined_output}")
    print(f"Total seasonal storage units: {len(combined_df)}")
    print(f"Total power capacity: {combined_df['charge_capacity_MW'].sum()} MW")
    print(f"Total energy capacity: {combined_df['storage_capacity_mwh'].sum()} MWh")
    
    return combined_df

def create_zone_specific_assignments():
    """
    Create zone-specific seasonal storage assignments for portfolio analysis
    """
    print("\n" + "="*50)
    print("CREATING ZONE-SPECIFIC SEASONAL STORAGE ASSIGNMENTS")
    print("="*50)
    
    # Read bus properties
    bus_prop_path = data_path / "grid" / "bus_prop_boyuan.csv"
    bus_prop = pd.read_csv(bus_prop_path)
    
    focus_zones = ["A", "B", "D"]
    
    for zone in focus_zones:
        print(f"\nCreating seasonal storage assignment for Zone {zone} only...")
        
        # Get buses in this zone
        zone_buses = bus_prop[bus_prop["ZONE"] == zone]["BUS_I"].tolist()
        
        # Select 2-3 representative buses
        if len(zone_buses) >= 3:
            selected_buses = zone_buses[::len(zone_buses)//3][:3]
        else:
            selected_buses = zone_buses[:2]
        
        # Create seasonal storage data for this zone only
        seasonal_storage_data = []
        
        for i, bus_id in enumerate(selected_buses):
            # Zone-specific capacity
            if zone == "A":
                charge_capacity_MW = 800 + i * 200  # Larger for Zone A
            elif zone == "B":
                charge_capacity_MW = 500 + i * 150
            else:  # Zone D
                charge_capacity_MW = 600 + i * 100
            
            storage_capacity_mwh = charge_capacity_MW * 168  # 1 week
            
            seasonal_storage_data.append({
                'bus_id': bus_id,
                'charge_capacity_MW': charge_capacity_MW,
                'storage_capacity_mwh': storage_capacity_mwh
            })
        
        # Save zone-specific file
        zone_df = pd.DataFrame(seasonal_storage_data)
        run_dir = runs_path / "low_RE_high_elec_iter0"
        inputs_dir = run_dir / "inputs"
        
        zone_file = inputs_dir / f"seasonal_storage_assignment_zone_{zone}_only.csv"
        zone_df.to_csv(zone_file, index=False)
        print(f"  Zone {zone} seasonal storage: {zone_file}")
        print(f"  Capacity: {zone_df['charge_capacity_MW'].sum()} MW, {zone_df['storage_capacity_mwh'].sum()} MWh")

if __name__ == "__main__":
    print("SEASONAL STORAGE DATA GENERATION")
    print("Based on professor's notes: focus on zones A, B, D")
    print("="*50)
    
    # Create main seasonal storage assignment
    combined_df = create_seasonal_storage_assignment()
    
    # Create zone-specific assignments for portfolio analysis
    create_zone_specific_assignments()
    
    print("\n" + "="*50)
    print("SEASONAL STORAGE DATA GENERATION COMPLETE")
    print("="*50)
    print("Files created:")
    print("  - seasonal_storage_assignment.csv (all zones)")
    print("  - seasonal_storage_assignment_zone_A_only.csv")
    print("  - seasonal_storage_assignment_zone_B_only.csv") 
    print("  - seasonal_storage_assignment_zone_D_only.csv")
    print("\nNext steps:")
    print("1. Run ACORN with seasonal storage enabled")
    print("2. Compare baseline vs seasonal storage scenarios")
    print("3. Analyze marginal benefits by zone")