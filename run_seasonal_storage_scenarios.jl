#!/usr/bin/env julia

# Seasonal Storage Scenario Runner
# Implements the Portfolio × Stress × Zone framework from professor's notes

using CSV
using DataFrames

# Include the scenario framework
include("src/julia/scenario_framework.jl")

"""
Main function to run all seasonal storage scenarios
Based on professor's notes:
- Portfolio (P): Baseline → Seasonal Storage
- Stress (S): (heatwave, cold snap, etc.)
- Zone (Z): NYISO zones (A…K) or curated subset
"""

function main()
    println("=== Seasonal Storage Scenario Analysis ===")
    println("Based on professor's research framework")
    println()
    
    # Scenario parameters
    run_name = "seasonal_storage_analysis"
    climate_scenario = "historical_1980_2019"
    sim_year = 2019
    branchprop_name = "boyuan"
    busprop_name = "boyuan"
    if_lim_name = "boyuan"
    save_name = "seasonal_storage_scenarios"
    
    # Focus zones as mentioned in professor's notes
    focus_zones = ["A", "B", "D"]  # "usually zone near renewable"
    
    println("1. FLUCTUATION ANALYSIS")
    println("   - Battery degradation/disincentive modeling implemented")
    println("   - Usage frequency tracking for small units")
    println("   - Near-term vs future storage optimization")
    println()
    
    println("2. SMALL UNIT ANALYSIS")
    println("   - Zone selection: $(focus_zones)")
    println("   - Usage frequency analysis: 'see if its used too often'")
    println("   - Zone A focus: 'Zone A appropriate'")
    println()
    
    println("3. SCENARIO FRAMEWORK")
    println("   - Portfolio (P): Baseline → Seasonal Storage")
    println("   - Stress (S): heatwave, cold snap scenarios")
    println("   - Zone (Z): Focus on zones A, B, D")
    println()
    
    # Run the complete scenario analysis
    try
        results = run_seasonal_storage_scenarios(
            run_name, climate_scenario, sim_year, branchprop_name, busprop_name, if_lim_name, save_name;
            exclude_external_zones = true,
            include_new_hvdc = false
        )
        
        println("=== RESULTS SUMMARY ===")
        println("Portfolio Sweep Results:")
        for (zone, zone_results) in results[:portfolio_sweep]
            println("  Zone $zone:")
            println("    Marginal Benefit: $(zone_results[:marginal_benefit])")
        end
        
        println("\nStress Sweep Results:")
        for (scenario, stress_results) in results[:stress_sweep]
            println("  $scenario:")
            println("    Stress Performance: $(stress_results[:stress_performance])")
        end
        
        println("\nZone A Focus Analysis:")
        println("  $(results[:zone_a_focus])")
        
        println("\n=== ANALYSIS COMPLETE ===")
        println("Key findings:")
        println("- Fluctuation disincentive: $(results[:summary][:fluctuation_analysis])")
        println("- Usage frequency: $(results[:summary][:usage_frequency])")
        println("- Zone A appropriateness: $(results[:zone_a_focus][:storage_appropriateness])")
        println("- Marginal benefits by zone: $(results[:summary][:marginal_benefits])")
        
    catch e
        println("Error running scenarios: $e")
        println("Please check your input data and configuration.")
    end
end

# Run the analysis
main()