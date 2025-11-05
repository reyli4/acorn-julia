# Professor's Notes Testing Workflow

## 🎯 **Complete Testing Workflow Based on Professor's Notes**

### **Professor's Notes Requirements:**
1. **Fluctuation:** Battery constant charging/discharging disincentive
2. **Small unit:** Select small zone, see if used too often, usually zone near renewable (A, B, D)
3. **Zone A appropriate**
4. **Base case:** Electricity produced, compare to seasonal storage run

## 📋 **Step-by-Step Testing Workflow**

### **Step 1: Generate Seasonal Storage Data** 🔄

**File:** `/workspace/runs/low_RE_high_elec_iter0/construct_inputs.ipynb`

**Current code (lines 1204-1220):**
```python
# Seasonal Storage
# Generate seasonal storage sites per zone
df_genX_seasonal_storage = pu.map_genX_zones_to_nyiso(df_genX[df_genX['Resource'] == 'seasonal_storage'].copy(), genX_zone_col='Zone')
df_genX_seasonal_storage = pi.generate_seasonal_storage_sites(df_genX_seasonal_storage,
                                                             sites_per_zone = 1,
                                                             columns_to_scale = ['EndCap', 'EndEnergyCap'])

# Assign to random bus
df_genX_seasonal_storage = pu.nearest_neighbor_lat_lon(df_genX_seasonal_storage, PV_bus_only=False)

# Store seasonal storage assignment
(
    df_genX_seasonal_storage[['bus_id', 'EndCap', 'EndEnergyCap']]
    .groupby('bus_id').sum()
    .rename(columns={'EndCap':'charge_capacity_MW',
                     'EndEnergyCap':'storage_capacity_mwh'})
    .to_csv(f"{run_path}/inputs/seasonal_storage_assignment.csv")
)
```

**To test different zones/buses, modify these parameters:**

#### **A. Test Different Zones (A, B, D focus)**
```python
# Modify the zone mapping to focus on specific zones
# For Zone A only:
df_genX_seasonal_storage = pu.map_genX_zones_to_nyiso(
    df_genX[df_genX['Resource'] == 'seasonal_storage'].copy(), 
    genX_zone_col='Zone'
)
# Filter for Zone A only
df_genX_seasonal_storage = df_genX_seasonal_storage[df_genX_seasonal_storage['Zone'] == 'A']

# For Zone B only:
df_genX_seasonal_storage = df_genX_seasonal_storage[df_genX_seasonal_storage['Zone'] == 'B']

# For Zone D only:
df_genX_seasonal_storage = df_genX_seasonal_storage[df_genX_seasonal_storage['Zone'] == 'D']
```

#### **B. Test Different Bus Assignments**
```python
# Instead of random bus assignment, specify specific buses
# For Zone A buses (54, 55, 56):
df_genX_seasonal_storage['bus_id'] = [54, 55, 56]  # Zone A buses

# For Zone B buses (52, 53):
df_genX_seasonal_storage['bus_id'] = [52, 53]  # Zone B buses

# For Zone D buses (48, 49):
df_genX_seasonal_storage['bus_id'] = [48, 49]  # Zone D buses
```

#### **C. Test Different Capacities**
```python
# Modify the scaling parameters
df_genX_seasonal_storage = pi.generate_seasonal_storage_sites(
    df_genX_seasonal_storage,
    sites_per_zone = 2,  # More sites per zone
    columns_to_scale = ['EndCap', 'EndEnergyCap']
)

# Or manually set capacities
df_genX_seasonal_storage['EndCap'] = [500, 700, 900]  # Power capacity (MW)
df_genX_seasonal_storage['EndEnergyCap'] = [84000, 117600, 151200]  # Energy capacity (MWh)
```

### **Step 2: Run ACORN with Seasonal Storage** 🔄

**File:** `/workspace/runs/low_RE_high_elec_iter0/run_acorn.sh`

**Current command:**
```bash
bash run_acorn.sh
```

**To test different scenarios, modify the save_name:**
```bash
# For Zone A only test:
julia scripts/04_run_acorn.jl --project-dir /workspace --run-dir /workspace/runs/low_RE_high_elec_iter0 --if_lim_name boyuan --exclude_external_zones 1 --include_new_hvdc 0 --save_name zone_a_test

# For Zone B only test:
julia scripts/04_run_acorn.jl --project-dir /workspace --run-dir /workspace/runs/low_RE_high_elec_iter0 --if_lim_name boyuan --exclude_external_zones 1 --include_new_hvdc 0 --save_name zone_b_test

# For Zone D only test:
julia scripts/04_run_acorn.jl --project-dir /workspace --run-dir /workspace/runs/low_RE_high_elec_iter0 --if_lim_name boyuan --exclude_external_zones 1 --include_new_hvdc 0 --save_name zone_d_test
```

### **Step 3: Test Professor's Notes Requirements** 📊

**File:** `/workspace/run_seasonal_storage_scenarios.jl`

**Current code (lines 20-90):**
```julia
function main()
    # Scenario parameters
    run_name = "low_RE_high_elec_iter0"  # Your scenario
    climate_scenario = "historical_1980_2019"
    sim_year = 2019
    branchprop_name = "boyuan"
    busprop_name = "boyuan"
    if_lim_name = "boyuan"
    save_name = "seasonal_storage_scenarios"
    
    # Focus zones as mentioned in professor's notes
    focus_zones = ["A", "B", "D"]  # "usually zone near renewable"
```

**To test different zones, modify focus_zones:**
```julia
# Test Zone A only:
focus_zones = ["A"]

# Test Zone B only:
focus_zones = ["B"]

# Test Zone D only:
focus_zones = ["D"]

# Test all zones:
focus_zones = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K"]
```

### **Step 4: Analyze Results Based on Professor's Notes** 🔍

**File:** `/workspace/src/julia/scenario_framework.jl`

#### **A. Fluctuation Analysis (Lines 284-294 in acorn.jl)**
```julia
# Usage frequency penalty - discourage constant charging/discharging
usage_frequency_penalty = 0.1  # Penalty factor for over-usage
@variable(model, usage_penalty[1:length(storage_bus_ids)])
for i in 1:length(storage_bus_ids)
    @constraint(model, usage_penalty[i] >= sum(battery_usage[i, :]) - 0.5 * nt)  # Penalty if usage > 50% of time
end
```

**To test different penalty levels:**
```julia
# More aggressive penalty:
usage_frequency_penalty = 0.2  # Higher penalty

# Less aggressive penalty:
usage_frequency_penalty = 0.05  # Lower penalty

# Different usage threshold:
@constraint(model, usage_penalty[i] >= sum(battery_usage[i, :]) - 0.3 * nt)  # Penalty if usage > 30% of time
```

#### **B. Small Unit Analysis (Lines 273-282)**
```julia
function calculate_usage_frequency(results)
    total_usage = results[:seasonal][:total_storage_usage]
    total_capacity = results[:seasonal][:total_storage_capacity]
    return total_usage / total_capacity
end
```

**To test different usage thresholds:**
```julia
# Check if usage > 50% (over-usage)
over_usage = usage_frequency > 0.5

# Check if usage > 30% (more sensitive)
over_usage = usage_frequency > 0.3

# Check if usage > 70% (less sensitive)
over_usage = usage_frequency > 0.7
```

#### **C. Zone A Appropriateness (Lines 227-264)**
```julia
function run_zone_a_focus_analysis(...)
    zone_a_results = Dict(
        :renewable_proximity = "High",  # Zone A is near renewable resources
        :storage_appropriateness = "High",  # Professor noted "Zone A appropriate"
        :usage_frequency = calculate_usage_frequency(zone_a_results),
        :marginal_benefit = zone_a_results[:marginal_benefit]
    )
end
```

**To test different appropriateness criteria:**
```julia
# More strict appropriateness:
:storage_appropriateness = usage_frequency < 0.3 ? "High" : "Low"

# Less strict appropriateness:
:storage_appropriateness = usage_frequency < 0.7 ? "High" : "Low"
```

#### **D. Base Case Comparison (Lines 147-163)**
```julia
function calculate_marginal_benefit(baseline_results, seasonal_results)
    marginal_benefit = Dict(
        :load_shedding_reduction = baseline_load_shedding - seasonal_load_shedding,
        :cost_reduction = baseline_cost - seasonal_cost,
        :renewable_utilization = calculate_renewable_utilization(seasonal_results),
        :storage_utilization = calculate_storage_utilization(seasonal_results)
    )
    return marginal_benefit
end
```

## 🧪 **Testing Scenarios Based on Professor's Notes**

### **Scenario 1: Zone A Focus Test**
```python
# In construct_inputs.ipynb:
df_genX_seasonal_storage = df_genX_seasonal_storage[df_genX_seasonal_storage['Zone'] == 'A']
df_genX_seasonal_storage['bus_id'] = [54, 55, 56]  # Zone A buses
```

### **Scenario 2: Small Unit Analysis Test**
```python
# Test with smaller capacities:
df_genX_seasonal_storage['EndCap'] = [100, 150, 200]  # Smaller power capacity
df_genX_seasonal_storage['EndEnergyCap'] = [16800, 25200, 33600]  # Smaller energy capacity
```

### **Scenario 3: Usage Frequency Test**
```julia
# In acorn.jl, modify the penalty threshold:
@constraint(model, usage_penalty[i] >= sum(battery_usage[i, :]) - 0.3 * nt)  # More sensitive
```

### **Scenario 4: Base Case Comparison Test**
```bash
# Run baseline (no seasonal storage):
julia scripts/04_run_acorn.jl --save_name baseline_test

# Run seasonal storage:
julia scripts/04_run_acorn.jl --save_name seasonal_test

# Compare results:
julia run_seasonal_storage_scenarios.jl
```

## 📊 **Expected Results for Each Test**

### **Zone A Test:**
- ✅ **Highest renewable proximity**
- ✅ **Best appropriateness rating**
- ✅ **Largest marginal benefits**

### **Small Unit Test:**
- ✅ **Usage frequency analysis**
- ✅ **Over-usage detection**
- ✅ **Penalty application**

### **Base Case Comparison:**
- ✅ **Load shedding reduction**
- ✅ **Cost reduction**
- ✅ **Renewable utilization improvement**

## 🚀 **Complete Testing Command Sequence**

```bash
# 1. Generate data for specific zone
cd /workspace/runs/low_RE_high_elec_iter0
jupyter notebook construct_inputs.ipynb
# Modify the seasonal storage section for your test case
# Run all cells

# 2. Run ACORN with seasonal storage
bash run_acorn.sh

# 3. Run portfolio analysis
cd /workspace
julia run_seasonal_storage_scenarios.jl

# 4. Check results
ls -la /workspace/runs/low_RE_high_elec_iter0/outputs/historical_1980_2019/nyiso_only/seasonal_*
```

This workflow allows you to test all aspects of your professor's notes systematically! 🎉