using CSV
using DataFrames

"""
Scenario Framework for ACORN Seasonal Storage Analysis
Based on professor's notes: Portfolio (P) × Stress (S) × Zone (Z)
"""

# Define scenario types
abstract type ScenarioType end
struct Baseline <: ScenarioType end
struct SeasonalStorage <: ScenarioType end

abstract type StressType end
struct NoStress <: StressType end
struct Heatwave <: StressType end
struct ColdSnap <: StressType end

# Zone definitions
const FOCUS_ZONES = ["A", "B", "D"]  # Zones near renewable resources
const ALL_ZONES = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K"]

"""
Core Portfolio Sweep (no stress): All P × Selected Z
Shows marginal benefit of added tech by zone
"""
function run_core_portfolio_sweep(
    run_name::String,
    climate_scenario::String,
    sim_year::Int,
    branchprop_name::String,
    busprop_name::String,
    if_lim_name::String,
    save_name::String;
    zones::Vector{String} = FOCUS_ZONES,
    exclude_external_zones::Bool = true,
    include_new_hvdc::Bool = false
)
    results = Dict()
    
    for zone in zones
        println("Running core portfolio sweep for zone $zone")
        
        # Baseline scenario
        baseline_results = run_acorn(
            run_name, climate_scenario, sim_year, branchprop_name, busprop_name, if_lim_name,
            "$(save_name)_baseline_zone_$(zone)";
            exclude_external_zones = exclude_external_zones,
            include_new_hvdc = include_new_hvdc,
            storage_eff = 0.75,
            seasonal_storage_eff = 0.95
        )
        
        # Seasonal storage scenario (only in focus zone)
        seasonal_results = run_acorn(
            run_name, climate_scenario, sim_year, branchprop_name, busprop_name, if_lim_name,
            "$(save_name)_seasonal_zone_$(zone)";
            exclude_external_zones = exclude_external_zones,
            include_new_hvdc = include_new_hvdc,
            storage_eff = 0.75,
            seasonal_storage_eff = 0.95
        )
        
        # Calculate marginal benefit
        marginal_benefit = calculate_marginal_benefit(baseline_results, seasonal_results)
        results[zone] = (baseline = baseline_results, seasonal = seasonal_results, marginal_benefit = marginal_benefit)
    end
    
    return results
end

"""
Stress Sweep: {Baseline, Seasonal} × S × Selected Z
Isolates how seasonal storage performs under adverse conditions by zone
"""
function run_stress_sweep(
    run_name::String,
    climate_scenario::String,
    sim_year::Int,
    branchprop_name::String,
    busprop_name::String,
    if_lim_name::String,
    save_name::String;
    zones::Vector{String} = FOCUS_ZONES,
    stress_scenarios::Vector{String} = ["heatwave", "cold_snap"],
    exclude_external_zones::Bool = true,
    include_new_hvdc::Bool = false
)
    results = Dict()
    
    for zone in zones
        for stress in stress_scenarios
            println("Running stress sweep for zone $zone under $stress")
            
            # Apply stress scenario
            stress_climate_scenario = apply_stress_scenario(climate_scenario, stress)
            
            # Baseline under stress
            baseline_stress_results = run_acorn(
                run_name, stress_climate_scenario, sim_year, branchprop_name, busprop_name, if_lim_name,
                "$(save_name)_baseline_$(stress)_zone_$(zone)";
                exclude_external_zones = exclude_external_zones,
                include_new_hvdc = include_new_hvdc,
                storage_eff = 0.75,
                seasonal_storage_eff = 0.95
            )
            
            # Seasonal storage under stress
            seasonal_stress_results = run_acorn(
                run_name, stress_climate_scenario, sim_year, branchprop_name, busprop_name, if_lim_name,
                "$(save_name)_seasonal_$(stress)_zone_$(zone)";
                exclude_external_zones = exclude_external_zones,
                include_new_hvdc = include_new_hvdc,
                storage_eff = 0.75,
                seasonal_storage_eff = 0.95
            )
            
            # Calculate stress performance
            stress_performance = calculate_stress_performance(baseline_stress_results, seasonal_stress_results)
            results["$(zone)_$(stress)"] = (
                baseline = baseline_stress_results,
                seasonal = seasonal_stress_results,
                stress_performance = stress_performance
            )
        end
    end
    
    return results
end

"""
Apply stress scenario to climate data
"""
function apply_stress_scenario(climate_scenario::String, stress::String)
    if stress == "heatwave"
        return "$(climate_scenario)_heatwave"
    elseif stress == "cold_snap"
        return "$(climate_scenario)_cold_snap"
    else
        return climate_scenario
    end
end

"""
Calculate marginal benefit of seasonal storage
"""
function calculate_marginal_benefit(baseline_results, seasonal_results)
    # Calculate key metrics
    baseline_load_shedding = sum(baseline_results[:load_shedding])
    seasonal_load_shedding = sum(seasonal_results[:load_shedding])
    
    baseline_cost = baseline_results[:total_cost]
    seasonal_cost = seasonal_results[:total_cost]
    
    marginal_benefit = Dict(
        :load_shedding_reduction = baseline_load_shedding - seasonal_load_shedding,
        :cost_reduction = baseline_cost - seasonal_cost,
        :renewable_utilization = calculate_renewable_utilization(seasonal_results),
        :storage_utilization = calculate_storage_utilization(seasonal_results)
    )
    
    return marginal_benefit
end

"""
Calculate stress performance metrics
"""
function calculate_stress_performance(baseline_stress_results, seasonal_stress_results)
    # Compare how well seasonal storage performs under stress vs baseline
    baseline_resilience = calculate_resilience_metrics(baseline_stress_results)
    seasonal_resilience = calculate_resilience_metrics(seasonal_stress_results)
    
    stress_performance = Dict(
        :resilience_improvement = seasonal_resilience - baseline_resilience,
        :load_shedding_under_stress = seasonal_stress_results[:load_shedding],
        :storage_response = calculate_storage_response(seasonal_stress_results)
    )
    
    return stress_performance
end

"""
Calculate renewable utilization
"""
function calculate_renewable_utilization(results)
    # Calculate how well renewable energy is utilized
    total_renewable = results[:wind_generation] + results[:solar_generation]
    total_demand = results[:total_demand]
    
    return total_renewable / total_demand
end

"""
Calculate storage utilization
"""
function calculate_storage_utilization(results)
    # Calculate how well storage is utilized
    total_storage_energy = results[:seasonal_storage_energy]
    total_storage_capacity = results[:seasonal_storage_capacity]
    
    return total_storage_energy / total_storage_capacity
end

"""
Calculate resilience metrics
"""
function calculate_resilience_metrics(results)
    # Calculate system resilience under stress
    load_shedding = results[:load_shedding]
    total_demand = results[:total_demand]
    
    return 1.0 - (load_shedding / total_demand)  # Higher is more resilient
end

"""
Calculate storage response to stress
"""
function calculate_storage_response(results)
    # Calculate how storage responds to stress conditions
    storage_discharge = results[:seasonal_storage_discharge]
    storage_charge = results[:seasonal_storage_charge]
    
    return storage_discharge - storage_charge  # Net discharge
end

"""
Zone A Focus Analysis
Based on professor's note: "Zone A appropriate"
"""
function run_zone_a_focus_analysis(
    run_name::String,
    climate_scenario::String,
    sim_year::Int,
    branchprop_name::String,
    busprop_name::String,
    if_lim_name::String,
    save_name::String;
    exclude_external_zones::Bool = true,
    include_new_hvdc::Bool = false
)
    println("Running Zone A focus analysis...")
    
    # Zone A specific analysis
    zone_a_results = run_core_portfolio_sweep(
        run_name, climate_scenario, sim_year, branchprop_name, busprop_name, if_lim_name, save_name;
        zones = ["A"],
        exclude_external_zones = exclude_external_zones,
        include_new_hvdc = include_new_hvdc
    )
    
    # Analyze Zone A characteristics
    zone_a_analysis = analyze_zone_a_characteristics(zone_a_results["A"])
    
    return zone_a_analysis
end

"""
Analyze Zone A characteristics
"""
function analyze_zone_a_characteristics(zone_a_results)
    # Zone A specific analysis based on professor's notes
    analysis = Dict(
        :renewable_proximity = "High",  # Zone A is near renewable resources
        :storage_appropriateness = "High",  # Professor noted "Zone A appropriate"
        :usage_frequency = calculate_usage_frequency(zone_a_results),
        :marginal_benefit = zone_a_results[:marginal_benefit]
    )
    
    return analysis
end

"""
Calculate usage frequency for small units
Based on professor's note: "see if its used too often"
"""
function calculate_usage_frequency(results)
    # Calculate how often storage is used
    total_usage = results[:seasonal][:total_storage_usage]
    total_capacity = results[:seasonal][:total_storage_capacity]
    
    return total_usage / total_capacity
end

"""
Main scenario runner
"""
function run_seasonal_storage_scenarios(
    run_name::String,
    climate_scenario::String,
    sim_year::Int,
    branchprop_name::String,
    busprop_name::String,
    if_lim_name::String,
    save_name::String;
    exclude_external_zones::Bool = true,
    include_new_hvdc::Bool = false
)
    println("Running seasonal storage scenario analysis...")
    
    # 1. Core portfolio sweep
    println("1. Running core portfolio sweep...")
    portfolio_results = run_core_portfolio_sweep(
        run_name, climate_scenario, sim_year, branchprop_name, busprop_name, if_lim_name, save_name;
        exclude_external_zones = exclude_external_zones,
        include_new_hvdc = include_new_hvdc
    )
    
    # 2. Stress sweep
    println("2. Running stress sweep...")
    stress_results = run_stress_sweep(
        run_name, climate_scenario, sim_year, branchprop_name, busprop_name, if_lim_name, save_name;
        exclude_external_zones = exclude_external_zones,
        include_new_hvdc = include_new_hvdc
    )
    
    # 3. Zone A focus analysis
    println("3. Running Zone A focus analysis...")
    zone_a_results = run_zone_a_focus_analysis(
        run_name, climate_scenario, sim_year, branchprop_name, busprop_name, if_lim_name, save_name;
        exclude_external_zones = exclude_external_zones,
        include_new_hvdc = include_new_hvdc
    )
    
    # 4. Compile results
    final_results = Dict(
        :portfolio_sweep = portfolio_results,
        :stress_sweep = stress_results,
        :zone_a_focus = zone_a_results,
        :summary = generate_summary(portfolio_results, stress_results, zone_a_results)
    )
    
    return final_results
end

"""
Generate summary of results
"""
function generate_summary(portfolio_results, stress_results, zone_a_results)
    summary = Dict(
        :total_zones_analyzed = length(portfolio_results),
        :focus_zones = FOCUS_ZONES,
        :zone_a_appropriateness = zone_a_results[:storage_appropriateness],
        :marginal_benefits = [results[:marginal_benefit] for results in values(portfolio_results)],
        :stress_resilience = [results[:stress_performance] for results in values(stress_results)]
    )
    
    return summary
end