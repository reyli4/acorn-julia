# Seasonal Storage Implementation Summary

## Overview
This document summarizes the implementation of seasonal storage technology into the ACORN model pipeline. Seasonal storage differs from regular battery storage in that it can store energy for much longer periods (seasonal timescales) with higher efficiency and larger capacity.

## Files Modified

### 1. `/workspace/src/julia/acorn.jl`
**Key Changes:**
- Added `seasonal_storage_eff=0.95` parameter (higher efficiency than regular storage)
- Added seasonal storage data reading from `seasonal_storage_assignment.csv`
- Added seasonal storage variables: `seasonal_charge`, `seasonal_discharge`, `seasonal_batt_state`
- Updated node balance constraints to include seasonal storage
- Added seasonal storage constraints with higher efficiency
- Updated objective function to include seasonal storage costs
- Added seasonal storage results extraction and output

### 2. `/workspace/src/julia/utils.jl`
**Key Changes:**
- Added `add_seasonal_storage_generators()` function
- Added `create_seasonal_storage_assignment()` function
- Seasonal storage generators have different characteristics:
  - Higher efficiency (95% vs 75%)
  - Lower operating costs
  - Different generation type ("SeasonalStorage")

### 3. `/workspace/src/python/prepare_inputs.py`
**Key Changes:**
- Updated `resource_mapping()` to recognize "seasonal_storage" resources
- Added `generate_seasonal_storage_sites()` function
- Seasonal storage sites have different characteristics:
  - Much larger energy capacity (seasonal scale)
  - Different power-to-energy ratios
  - Typically located near renewable resources

### 4. `/workspace/runs/*/construct_inputs.ipynb`
**Key Changes:**
- Added seasonal storage section to generate seasonal storage sites
- Maps GenX seasonal storage resources to NYISO zones
- Creates seasonal storage assignment files
- Handles both power capacity (EndCap) and energy capacity (EndEnergyCap)

## Data Structure

### Seasonal Storage Assignment File
**File:** `seasonal_storage_assignment.csv`
**Columns:**
- `bus_id`: Bus ID where seasonal storage is located
- `charge_capacity_MW`: Maximum charging/discharging power (MW)
- `storage_capacity_mwh`: Maximum energy storage capacity (MWh)

### Output Files
**New output files for seasonal storage:**
- `seasonal_charge_<year>.csv`: Seasonal storage charging power by time
- `seasonal_discharge_<year>.csv`: Seasonal storage discharging power by time  
- `seasonal_batt_state_<year>.csv`: Seasonal storage energy state by time

## Key Differences from Regular Storage

| Characteristic | Regular Storage | Seasonal Storage |
|----------------|----------------|------------------|
| Efficiency | 75% | 95% |
| Energy Capacity | Daily scale (hours) | Seasonal scale (months) |
| Power-to-Energy Ratio | High (4-8 hours) | Low (weeks to months) |
| Cost Structure | Higher operating cost | Lower operating cost |
| Use Case | Daily balancing | Seasonal balancing |

## Usage

### 1. GenX Integration
To use seasonal storage, ensure your GenX outputs include seasonal storage resources:
```python
# In GenX results, seasonal storage should be labeled as:
# Resource: "seasonal_storage"
# EndCap: Power capacity (MW)
# EndEnergyCap: Energy capacity (MWh)
```

### 2. Running ACORN with Seasonal Storage
```julia
# The model will automatically detect seasonal storage if the file exists:
# runs/<run_name>/inputs/seasonal_storage_assignment.csv

run_acorn(
    "your_run_name",
    "climate_scenario", 
    2019,
    "branchprop_name",
    "busprop_name", 
    "if_lim_name",
    "save_name";
    seasonal_storage_eff=0.95  # Higher efficiency for seasonal storage
)
```

### 3. Portfolio Sweep Implementation
For the portfolio sweep analysis mentioned in your notes:

**Core portfolio sweep (no stress):**
- Baseline → Seasonal Storage
- Compare marginal benefit of seasonal storage by zone
- Focus on zones A, B, D (near renewable resources)

**Stress sweep:**
- {Baseline, Seasonal} × Stress × Selected Zones
- Isolate how seasonal storage performs under adverse conditions
- Test heatwave, cold snap scenarios

## Testing

A test script has been created at `/workspace/test_seasonal_storage.jl` to verify the implementation.

## Next Steps

1. **Data Generation**: Create seasonal storage assignment files for your specific scenarios
2. **Zone Selection**: Focus on zones A, B, D as mentioned in your notes
3. **Portfolio Analysis**: Implement the portfolio sweep framework
4. **Stress Testing**: Test seasonal storage under various climate stress scenarios
5. **Validation**: Compare results with baseline scenarios to quantify benefits

## Notes from Meeting

Based on your professor's notes:
- **Fluctuation**: Battery constant charging/discharging creates disincentive for near-term use
- **Small unit selection**: Focus on zones near renewable resources (A, B, D)
- **Base case**: Compare electricity production with seasonal storage vs. baseline
- **Portfolio factors**: P (Portfolio), S (Stress), Z (Zone) analysis framework

The implementation supports this analysis framework by providing:
- Separate seasonal storage technology
- Zone-specific deployment
- Stress scenario testing capability
- Portfolio comparison functionality