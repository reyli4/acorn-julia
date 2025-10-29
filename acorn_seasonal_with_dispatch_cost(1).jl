# ============================================================================
# Markers legend:
#   ### [SEASONAL NEW] ...            -> new blocks added beyond original acorn.jl
# ============================================================================

using CSV
using DataFrames
using Dates
using JuMP
using Gurobi

include("./utils.jl")


function run_acorn(
    run_name,
    climate_scenario,
    sim_year,
    branchprop_name,
    busprop_name,
    if_lim_name,
    save_name;
    exclude_external_zones=true,
    include_new_hvdc=false,
    storage_eff=0.75,
    ### [SEASONAL NEW] RUNTIME PARAMS for alignment with run_config
    battery_cycle_cost=1.0,             # proxy $/MWh for battery cycling
    penalty_unserved_load=10000.0,      # penalty for load shedding ($/MWh)
    include_seasonal_storage=true,
    ### [SEASONAL NEW] PARAMETERS: long-duration (hydrogen) storage
    seasonal_eff=0.45,                 # round-trip efficiency for seasonal device
    seasonal_charge_cost=0.0,          # variable cost proxy for charging
    seasonal_discharge_cost=0.0,       # variable cost proxy for discharging
    seasonal_cycle=true                # enforce s_state[:, end] == s_state[:, 1]
    )

    ############################
    # Read all data
    ############################
    data_directory = "$(project_path)/data"
    run_directory = "$(project_path)/runs/$(run_name)"
    out_path = "$(run_directory)/outputs/$(climate_scenario)/$(save_name)"
    
    if !isdir(out_path)
        mkpath(out_path)
    end

    # Read the load
    load = CSV.read("$(run_directory)/inputs/load_$(climate_scenario).csv", DataFrame)

    # Read generation
    solar_upv = CSV.read("$(run_directory)/inputs/solar_upv_$(climate_scenario).csv", DataFrame)
    wind = CSV.read("$(run_directory)/inputs/wind_$(climate_scenario).csv", DataFrame)
    solar_dpv = CSV.read("$(run_directory)/inputs/solar_dpv_$(climate_scenario).csv", DataFrame)

    # Read hydro
    hydro_scenario = split(climate_scenario, "_")[1]
    large_hydro = CSV.read("$(run_directory)/inputs/large_hydro_$(hydro_scenario).csv", DataFrame)
    small_hydro = CSV.read("$(run_directory)/inputs/small_hydro_$(hydro_scenario).csv", DataFrame)

    # Read storage
    storage = CSV.read("$(run_directory)/inputs/storage_assignment.csv", DataFrame)

    ### [SEASONAL NEW] INPUTS: seasonal storage (co-located hydrogen / long-duration)
# Optional: controlled by include_seasonal_storage and file presence
seasonal_file = "$(run_directory)/inputs/seasonal_storage.csv"
has_seasonal = include_seasonal_storage && isfile(seasonal_file)
if has_seasonal
    seasonal_storage = CSV.read(seasonal_file, DataFrame)
    seasonal_buses = seasonal_storage.bus_id
    # flexible column names for robustness
    charge_col = hasproperty(seasonal_storage, :charge_capacity_MW) ? :charge_capacity_MW :
                 hasproperty(seasonal_storage, :charge_capacity)    ? :charge_capacity    : nothing
    @assert charge_col !== nothing "seasonal_storage.csv must include charge_capacity_MW (or charge_capacity)"
    discharge_col = hasproperty(seasonal_storage, :discharge_capacity_MW) ? :discharge_capacity_MW :
                    hasproperty(seasonal_storage, :discharge_capacity)    ? :discharge_capacity    : nothing
    energy_col = hasproperty(seasonal_storage, :storage_capacity_MWh) ? :storage_capacity_MWh :
                 hasproperty(seasonal_storage, :storage_capacity)     ? :storage_capacity     : nothing
    @assert energy_col !== nothing "seasonal_storage.csv must include storage_capacity_MWh (or storage_capacity)"
    soc0_col = hasproperty(seasonal_storage, :soc0_frac) ? :soc0_frac : nothing
else
    seasonal_storage = DataFrame()
    seasonal_buses = Int[]
    charge_col = nothing; discharge_col = nothing; energy_col = nothing; soc0_col = nothing
end

# Read generators
genprop_nuclear = CSV.read("$(run_directory)/inputs/genprop_nuclear_matched.csv", DataFrame, stringtype=String) CSV.read("$(run_directory)/inputs/genprop_nuclear_matched.csv", DataFrame, stringtype=String)
    genprop_ng = CSV.read("$(run_directory)/inputs/genprop_ng_matched.csv", DataFrame, stringtype=String)
    genprop_hydro = CSV.read("$(run_directory)/inputs/genprop_hydro.csv", DataFrame, stringtype=String)
    genprop = vcat(genprop_nuclear, genprop_ng, genprop_hydro)

    # Read bus data
    busprop = CSV.read("$(data_directory)/grid/bus_prop_$(busprop_name).csv", DataFrame)
    bus_ids = busprop[:, "BUS_I"]

    # Read branch data
    branchprop = CSV.read("$(data_directory)/grid/branch_prop_$(branchprop_name).csv", DataFrame)

    ##### Remove external zones if specified
    if exclude_external_zones
        external_buses = [21, 29, 35, 100, 102, 103, 124, 125, 132, 134, 138]

        # Buses
        busprop = busprop[findall(.!in(external_buses), busprop.BUS_I), :]
        bus_ids = busprop[:, "BUS_I"]

        # Branches
        branchprop = branchprop[findall(.!in(external_buses), branchprop.F_BUS), :]
        branchprop = branchprop[findall(.!in(external_buses), branchprop.T_BUS), :]
    end

    #############################
    # Load adjustments 
    #############################
    # Subtract small solar 
    load = subtract_solar_dpv(load, solar_dpv, sim_year)

    # Subtract small hydro
    load = subtract_small_hydro(load, small_hydro, sim_year)

    if !exclude_external_zones
        # Fill the missing load buses with zero (external ones)
        println("NOTE: External buses all have zero load")
        load = leftjoin(DataFrame(bus_id=bus_ids), load, on=:bus_id)
        load = coalesce.(load, 0.0)
        load = sort(load, [:bus_id])
        load = max.(load, 0)
    end

    # Get final load data
    sim_dates = names(load)[2:end]
    nt = length(sim_dates)
    load_data = Matrix(load[:, sim_dates])

    #############################
    # Add generators
    #############################
    # Add solar and wind generators
    wind_bus_ids = wind[:, "bus_id"]
    genprop = add_wind_generators(genprop, wind_bus_ids)

    solar_upv_bus_ids = solar_upv[:, "bus_id"]
    genprop = add_solar_generators(genprop, solar_upv_bus_ids)

    if !exclude_external_zones
        if include_new_hvdc
            genprop = add_hvdc_generators(genprop, true)
        else
            genprop = add_hvdc_generators(genprop, false)
        end
    end

    # Get generator limits
    g_max = repeat(genprop[:, "PMAX"], 1, nt) # Maximum real power output (MW)
    g_min = repeat(genprop[:, "PMIN"], 1, nt) # Minimum real power output (MW)

    # Update for renewables
    wind_idx = findall(x -> x == "Wind", genprop[:, "UNIT_TYPE"])
    g_max[wind_idx, :] .= wind[:, sim_dates]

    solar_upv_idx = findall(x -> x == "SolarUPV", genprop[:, "UNIT_TYPE"])
    g_max[solar_upv_idx, :] .= solar_upv[:, sim_dates]

    # Get generator ramp rates
    ramp_down = max.(repeat(genprop[:, "RAMP_30"], 1, nt) .* 2, repeat(genprop[:, "PMAX"], 1, nt)) # max of 2*RAMP_30, PMAX
    ramp_up = copy(ramp_down)

    # Generator cost
    gencost = repeat(genprop[:, "COST_1"], 1, nt) # Cost per unit power generated

    # Get daily storage (battery) info
storage_path = "$(run_directory)/inputs/storage_assignment.csv"
storage = CSV.read(storage_path, DataFrame)
storage_bus_ids = storage[:, "bus_id"]
batt_charge_col = hasproperty(storage, :charge_capacity_MW) ? :charge_capacity_MW :
                  hasproperty(storage, :charge_capacity)    ? :charge_capacity    :
                  error("storage_assignment.csv needs charge_capacity_MW or charge_capacity")
batt_energy_col = hasproperty(storage, :storage_capacity_mwh) ? :storage_capacity_mwh :
                  hasproperty(storage, :storage_capacity_MWh) ? :storage_capacity_MWh :
                  error("storage_assignment.csv needs storage_capacity_mwh or storage_capacity_MWh")
storage_charge_cap = repeat(storage[:, batt_charge_col], 1, nt)
storage_energy_cap = repeat(storage[:, batt_energy_col], 1, nt + 1)

    ### [SEASONAL NEW] Build caps/initial state arrays for seasonal devices
n_seasonal = length(seasonal_buses)
if n_seasonal > 0
    s_charge_cap_vec = [seasonal_storage[i, charge_col] for i in 1:n_seasonal]
    s_discharge_cap_vec = discharge_col === nothing ? s_charge_cap_vec :
                          [seasonal_storage[i, discharge_col] for i in 1:n_seasonal]
    s_energy_cap_vec = [seasonal_storage[i, energy_col] for i in 1:n_seasonal]
    s_soc0_vec = soc0_col === nothing ? [0.5 for _ in 1:n_seasonal] :
                 [seasonal_storage[i, soc0_col] for i in 1:n_seasonal]

    s_charge_cap = repeat(reshape(s_charge_cap_vec, :, 1), 1, nt)
    s_discharge_cap = repeat(reshape(s_discharge_cap_vec, :, 1), 1, nt)
    s_energy_cap = repeat(reshape(s_energy_cap_vec, :, 1), 1, nt+1)
else
    s_charge_cap = zeros(0, nt)
    s_discharge_cap = zeros(0, nt)
    s_energy_cap = zeros(0, nt+1)
    s_soc0_vec = Float64[]
end

    #########################
    # Interface limits
    #########################
    if_lims = CSV.read("$(data_directory)/nyiso/interface_limits/if_lim_$(if_lim_name).csv", DataFrame)

    if exclude_external_zones
        external_zones = ["NE", "IESO", "PJM"]
        if_lims = if_lims[findall(.!in(external_zones), if_lims.FROM_ZONE), :]
        if_lims = if_lims[findall(.!in(external_zones), if_lims.TO_ZONE), :]
    end

    if_lim_up = repeat(if_lims[:, "IF_MAX"], 1, nt)
    if_lim_down = repeat(if_lims[:, "IF_MIN"], 1, nt)

    # Get IF lim map and update branchprop
    if_lim_map, branchprop = create_interface_map(if_lims, branchprop)

    # Branch limits
    branch_lims = repeat(Float64.(branchprop[:, "RATE_A"]), 1, nt)
    branch_lims[branch_lims.==0] .= 99999.0

    ########################
    # Optimization 
    ########################
    n_gen = size(genprop, 1)
    n_bus = size(busprop, 1)
    n_branch = size(branchprop, 1)

    model = Model(Gurobi.Optimizer)

    ## Define variables
    @variable(model, pg[1:n_gen, 1:nt])
    @variable(model, flow[1:n_branch, 1:nt])
    @variable(model, bus_angle[1:n_bus, 1:nt])
    # daily storage (battery)
    @variable(model, charge[1:length(storage_bus_ids), 1:nt] >= 0)
    @variable(model, discharge[1:length(storage_bus_ids), 1:nt] >= 0)
    @variable(model, batt_state[1:length(storage_bus_ids), 1:nt+1] >= 0)
    # load shedding
    @variable(model, load_shedding[1:n_bus, 1:nt] >= 0)

    ### [SEASONAL NEW] Decision variables for seasonal devices
seasonal_cost_expr = 0.0
if n_seasonal > 0
    @variable(model, s_charge[1:n_seasonal, 1:nt] >= 0)
    @variable(model, s_discharge[1:n_seasonal, 1:nt] >= 0)
    @variable(model, s_state[1:n_seasonal, 1:nt+1] >= 0)
end

    ## Constraints 
    # Branch flow limits and power flow equations
    for l in 1:n_branch
        idx_from_bus = findfirst(x -> x == branchprop[l, "F_BUS"], busprop[:, "BUS_I"])
        idx_to_bus = findfirst(x -> x == branchprop[l, "T_BUS"], busprop[:, "BUS_I"])
        # Branch flow limits
        @constraint(model, -branch_lims[l, :] .<= flow[l, :] .<= branch_lims[l, :])
        # DC power flow equations
        @constraint(model, flow[l, :] .== (100 / branchprop[l, "BR_X"]) .*
                                          (bus_angle[idx_from_bus, :] .- bus_angle[idx_to_bus, :]))
    end

    # Node balance and phase angle constraints
    for idx in 1:n_bus
        bus_id = bus_ids[idx]

        # seasonal devices present?
        s_idx = findall(==(bus_id), seasonal_buses)

        if busprop[idx, "BUS_TYPE"] != 3  # Not the slack bus
            if bus_id in storage_bus_ids && !isempty(s_idx)
                # Node balance with battery + seasonal storage
                storage_idx = findfirst(==(bus_id), storage_bus_ids)
                @constraint(model, load_data[idx, 1:nt] .==
                    -sum(flow[l, 1:nt] for l in findall(x -> x == bus_id, branchprop[:, "F_BUS"])) .+
                     sum(flow[l, 1:nt] for l in findall(x -> x == bus_id, branchprop[:, "T_BUS"])) .+
                     sum(pg[l, 1:nt] for l in findall(x -> x == bus_id, genprop[:, "GEN_BUS"])) .+
                     discharge[storage_idx, 1:nt] .- charge[storage_idx, 1:nt] .+
                     sum(s_discharge[s_idx, 1:nt], dims=1) .- sum(s_charge[s_idx, 1:nt], dims=1) .+  ### [SEASONAL NEW] seasonal injection
                     load_shedding[idx, 1:nt])
            elseif bus_id in storage_bus_ids
                # Node balance with battery only
                storage_idx = findfirst(==(bus_id), storage_bus_ids)
                @constraint(model, load_data[idx, 1:nt] .==
                    -sum(flow[l, 1:nt] for l in findall(x -> x == bus_id, branchprop[:, "F_BUS"])) .+
                     sum(flow[l, 1:nt] for l in findall(x -> x == bus_id, branchprop[:, "T_BUS"])) .+
                     sum(pg[l, 1:nt] for l in findall(x -> x == bus_id, genprop[:, "GEN_BUS"])) .+
                     discharge[storage_idx, 1:nt] .- charge[storage_idx, 1:nt] .+
                     load_shedding[idx, 1:nt])
            elseif !isempty(s_idx)
                # Node balance with seasonal only
                @constraint(model, load_data[idx, 1:nt] .==
                    -sum(flow[l, 1:nt] for l in findall(x -> x == bus_id, branchprop[:, "F_BUS"])) .+
                     sum(flow[l, 1:nt] for l in findall(x -> x == bus_id, branchprop[:, "T_BUS"])) .+
                     sum(pg[l, 1:nt] for l in findall(x -> x == bus_id, genprop[:, "GEN_BUS"])) .+
                     sum(s_discharge[s_idx, 1:nt], dims=1) .- sum(s_charge[s_idx, 1:nt], dims=1) .+  ### [SEASONAL NEW] seasonal injection
                     load_shedding[idx, 1:nt])
            else
                # Node balance without storage devices
                @constraint(model, load_data[idx, 1:nt] .==
                    -sum(flow[l, 1:nt] for l in findall(x -> x == bus_id, branchprop[:, "F_BUS"])) .+
                     sum(flow[l, 1:nt] for l in findall(x -> x == bus_id, branchprop[:, "T_BUS"])) .+
                     sum(pg[l, 1:nt] for l in findall(x -> x == bus_id, genprop[:, "GEN_BUS"])) .+
                     load_shedding[idx, 1:nt])
            end
            @constraint(model, -2 * pi .<= bus_angle[idx, 1:nt] .<= 2 * pi)  # Voltage angle limits
        else  # Slack bus
            # Node balance for slack bus
            @constraint(model, load_data[idx, 1:nt] .==
                -sum(flow[l, 1:nt] for l in findall(x -> x == bus_id, branchprop[:, "F_BUS"])) .+
                 sum(flow[l, 1:nt] for l in findall(x -> x == bus_id, branchprop[:, "T_BUS"])) .+
                 sum(pg[l, 1:nt] for l in findall(x -> x == bus_id, genprop[:, "GEN_BUS"])) .+
                 load_shedding[idx, 1:nt])
            @constraint(model, bus_angle[idx, 1:nt] .== 0.2979 / 180 * pi)  # Fix voltage angle at slack bus
        end
    end

    # Battery constraints (original)
    @constraint(model, 0 .<= charge .<= storage_charge_cap)         # Charging limits
    @constraint(model, 0 .<= discharge .<= storage_charge_cap)      # Discharging limits

    # Battery state dynamics (original)
    for t in 1:nt
        @constraint(model, batt_state[1:end, t+1] .== batt_state[1:end, t] .+
            sqrt(storage_eff) .* charge[1:end, t] .- (1 / sqrt(storage_eff)) .* discharge[1:end, t])
    end

    # Battery capacity constraints & initial SoC (original)
    @constraint(model, 0.0 .* storage_energy_cap .<= batt_state .<= storage_energy_cap)
    @constraint(model, batt_state[:, 1] .== 0.3 .* storage_energy_cap[:, 1])

    ### [SEASONAL NEW] Seasonal constraints: caps + SoC dynamics + optional cyclic end
if n_seasonal > 0
    @constraint(model, 0 .<= s_charge .<= s_charge_cap)
    @constraint(model, 0 .<= s_discharge .<= s_discharge_cap)
    @constraint(model, 0 .<= s_state .<= s_energy_cap)
    @constraint(model, s_state[:, 1] .== reshape(s_soc0_vec, :, 1) .* s_energy_cap[:, 1])  # initial SoC
    for t in 1:nt
        @constraint(model, s_state[:, t+1] .== s_state[:, t] .+
            sqrt(seasonal_eff) .* s_charge[:, t] .- (1 / sqrt(seasonal_eff)) .* s_discharge[:, t])
    end
    if seasonal_cycle
        @constraint(model, s_state[:, end] .== s_state[:, 1])
    end
    # seasonal cost term used in objective
    seasonal_cost_expr = seasonal_charge_cost * sum(s_charge) + seasonal_discharge_cost * sum(s_discharge)
end
    if seasonal_cycle
        @constraint(model, s_state[:, end] .== s_state[:, 1])
    end

    # Impose interface limits (original)
    n_if_lims = size(if_lim_up)[1]

    for i in 1:n_if_lims
        # Sum flow across the interfaces
        branch_idx = if_lim_map[findall(==(i), if_lim_map[:, "IF_ID"]), "BR_IDX"]
        idx_signs = sign.(branch_idx)
        idx_abs = abs.(branch_idx)

        flow_sum = [sum(idx_signs .* flow[Int.(idx_abs), t]) for t in 1:nt]
        # Constraint
        @constraint(model, if_lim_down[i, 1:nt] .<= flow_sum .<= if_lim_up[i, 1:nt])
    end

    # Nuclear (original)
    nuclear_idx = findall(x -> x == "Nuclear", genprop[!, "UNIT_TYPE"])
    for idx in nuclear_idx
        @constraint(model, pg[idx, :] .== g_max[idx, :])
    end

    # Hydro weekly (original)
    niagara_idx = findfirst(x -> x == "Moses Niagara (Fleet)", genprop[!, "GEN_NAME"])
    moses_saund_idx = findfirst(x -> x == "St Lawrence - FDR (Fleet)", genprop[!, "GEN_NAME"])

    moses_saund_hydro = Matrix(filter_by_year(large_hydro, sim_year))[1, 2:end]
    niagara_hydro = Matrix(filter_by_year(large_hydro, sim_year))[2, 2:end]

    # Calculate the capacity rate of Moses Saunders
    hydro_pmax = genprop[moses_saund_idx, "PMAX"]
    hours_in_week = 24 * 7
    cap_rate = maximum(moses_saund_hydro ./ hours_in_week / hydro_pmax)
    if cap_rate > 1
        g_max[moses_saund_idx, :] .= g_max[moses_saund_idx, :] .* cap_rate
    end

    # Calculate the capacity rate of Niagara
    hydro_pmax = genprop[niagara_idx, "PMAX"]
    hours_in_week = 24 * 7
    cap_rate = maximum(niagara_hydro ./ hours_in_week / hydro_pmax)
    if cap_rate > 1
        g_max[niagara_idx, :] .= g_max[niagara_idx, :] .* cap_rate
    end

    # Do manually for now, update later
    last_hydro_day = split(names(filter_by_year(large_hydro, sim_year))[end], "-")[end]
    if last_hydro_day == "30"
        weekly_hours = vcat(fill(7 * 24, 52), [14])
    elseif last_hydro_day == "31"
        weekly_hours = vcat(fill(7 * 24, 52), [7])
    else
        throw(DomainError(last_hydro_day, "Error with hydro"))
    end

    # Cumulative time counter
    ct = 0
    for t in 1:48
        @constraint(model, sum(pg[niagara_idx, ct+1:ct+weekly_hours[t]]) == niagara_hydro[t])
        @constraint(model, sum(pg[moses_saund_idx, ct+1:ct+weekly_hours[t]]) == moses_saund_hydro[t])
        ct += weekly_hours[t]
    end

    # Generator capacity constraints (original)
    @constraint(model, g_min .<= pg .<= g_max)

    if !exclude_external_zones
        # HVDC constraints (original)
        csc_idx = findall(x -> x == "HVDC_CSC", genprop[!, "GEN_NAME"])
        @constraint(model, pg[csc_idx[1], :] .== -pg[csc_idx[2], :]) # SC+NPX1385

        neptune_idx = findall(x -> x == "HVDC_Neptune", genprop[!, "GEN_NAME"])
        @constraint(model, pg[neptune_idx[1], :] .== -pg[neptune_idx[2], :]) # Neptune

        vft_idx = findall(x -> x == "HVDC_VFT", genprop[!, "GEN_NAME"])
        @constraint(model, pg[vft_idx[1], :] .== -pg[vft_idx[2], :]) # VFT

        htp_idx = findall(x -> x == "HVDC_HTP", genprop[!, "GEN_NAME"])
        @constraint(model, pg[htp_idx[1], :] .== -pg[htp_idx[2], :]) # HTP
    end

    if include_new_hvdc
        # New HVDC (original)
        clean_path_idx = findall(x -> x == "HVDC_NYCleanPath", genprop[!, "GEN_NAME"])
        @constraint(model, pg[clean_path_idx[1], :] .== -pg[clean_path_idx[2], :]) # CleanPath

        chp_express_idx = findall(x -> x == "HVDC_CHPexpress", genprop[!, "GEN_NAME"])
        @constraint(model, pg[chp_express_idx[1], :] .== -pg[chp_express_idx[2], :]) # CHP Express
    end

    # Generator ramping (original)
    @constraint(model, -ramp_down[:, 2:nt] .<= pg[:, 2:nt] .- pg[:, 1:nt-1] .<= ramp_up[:, 2:nt])

    # Load shedding (original)
    @constraint(model, 0.0 .<= load_shedding .<= max.(load_data, 0))

    # Extract generation for wind and calculate curtailment (original)
    wind_gen = pg[wind_idx, :]
    wind_curt = Matrix(wind[:, sim_dates]) .- wind_gen

    # Extract generation for utility-scale solar (UPV) and calculate curtailment (original)
    solar_gen = pg[solar_upv_idx, :]
    solar_curt = Matrix(solar_upv[:, sim_dates]) .- solar_gen

    ### [SEASONAL NEW] Objective: add seasonal cycling costs to original objective
    @objective(model, Min,
        10000 * sum(load_shedding) +
        (sum(charge) + sum(discharge)) +                 # original battery proxy cost
        seasonal_charge_cost * sum(s_charge) +           # NEW
        seasonal_discharge_cost * sum(s_discharge) +     # NEW
        sum(gencost .* pg)
    )

    # RUN IT
    optimize!(model)

    if termination_status(model) == MOI.OPTIMAL
        # Extract results (original + seasonal)
        pg_result = value.(pg);
        flow_result = value.(flow);
        charge_result = value.(charge);
        discharge_result = value.(discharge);
        batt_state_result = value.(batt_state);
        load_shedding_result = value.(load_shedding);
        wind_curtail_result = value.(wind_curt);
        solar_curtail_result = value.(solar_curt);
        # seasonal results
        if n_seasonal > 0
            s_charge_result = value.(s_charge);
            s_discharge_result = value.(s_discharge);
            s_state_result = value.(s_state);
        end
    else
        println("Error with optimization")
    end

    ### [SEASONAL NEW] Outputs for seasonal storage (mirrors battery outputs)
    # Add bus/branch IDs and datetime to output files
    flow_result = hcat([branchprop[:, "F_BUS"] branchprop[:, "FROM_ZONE"]], flow_result)
    flow_result = hcat([branchprop[:, "T_BUS"] branchprop[:, "TO_ZONE"]], flow_result)
    flow_result = vcat(hcat(["from_bus" "from_bus_zone" "to_bus" "to_bus_zone"], reshape(sim_dates, 1, :)), flow_result)

    pg_result = hcat([genprop[:, "GEN_BUS"] map(x -> bus_to_zone[x], genprop[:, "GEN_BUS"])], pg_result)
    pg_result = vcat(hcat(["bus_id" "zone"], reshape(sim_dates, 1, :)), pg_result)

    charge_result = hcat([storage_bus_ids map(x -> bus_to_zone[x], storage_bus_ids)], charge_result)
    charge_result = vcat(hcat(["bus_id" "zone"], reshape(sim_dates, 1, :)), charge_result)

    discharge_result = hcat([storage_bus_ids map(x -> bus_to_zone[x], storage_bus_ids)], discharge_result)
    discharge_result = vcat(hcat(["bus_id" "zone"], reshape(sim_dates, 1, :)), discharge_result)

    batt_state_result = hcat([storage_bus_ids map(x -> bus_to_zone[x], storage_bus_ids)], batt_state_result)
    batt_state_result = vcat(hcat(["bus_id" "zone"], reshape(vcat(sim_dates, "end"), 1, :)), batt_state_result)

    load_shedding_result = hcat([bus_ids map(x -> bus_to_zone[x], bus_ids)], load_shedding_result)
    load_shedding_result = vcat(hcat(["bus_id" "zone"], reshape(sim_dates, 1, :)), load_shedding_result)

    wind_curtail_result = hcat([wind_bus_ids map(x -> bus_to_zone[x], wind_bus_ids)], wind_curtail_result)
    wind_curtail_result = vcat(hcat(["bus_id" "zone"], reshape(sim_dates, 1, :)), wind_curtail_result)

    solar_curtail_result = hcat([solar_upv_bus_ids map(x -> bus_to_zone[x], solar_upv_bus_ids)], solar_curtail_result)
    solar_curtail_result = vcat(hcat(["bus_id" "zone"], reshape(sim_dates, 1, :)), solar_curtail_result)

    # Save results as CSV files (original)
    CSV.write("$(out_path)/gen_$(sim_year).csv", DataFrame(pg_result, :auto), header=false)
    CSV.write("$(out_path)/flow_$(sim_year).csv", DataFrame(flow_result, :auto), header=false)
    CSV.write("$(out_path)/charge_$(sim_year).csv", DataFrame(charge_result, :auto), header=false)
    CSV.write("$(out_path)/discharge_$(sim_year).csv", DataFrame(discharge_result, :auto), header=false)
    CSV.write("$(out_path)/wind_curtailment_$(sim_year).csv", DataFrame(wind_curtail_result, :auto), header=false)
    CSV.write("$(out_path)/solar_curtailment_$(sim_year).csv", DataFrame(solar_curtail_result, :auto), header=false)
    CSV.write("$(out_path)/batt_state_$(sim_year).csv", DataFrame(batt_state_result, :auto), header=false)
    CSV.write("$(out_path)/load_shedding_$(sim_year).csv", DataFrame(load_shedding_result, :auto), header=false)

    ### [SEASONAL NEW] Save seasonal outputs
    if n_seasonal > 0
        s_charge_out = hcat([seasonal_buses map(x -> bus_to_zone[x], seasonal_buses)], s_charge_result)
        s_charge_out = vcat(hcat(["bus_id" "zone"], reshape(sim_dates, 1, :)), s_charge_out)
        CSV.write("$(out_path)/seasonal_charge_$(sim_year).csv", DataFrame(s_charge_out, :auto), header=false)

        s_discharge_out = hcat([seasonal_buses map(x -> bus_to_zone[x], seasonal_buses)], s_discharge_result)
        s_discharge_out = vcat(hcat(["bus_id" "zone"], reshape(sim_dates, 1, :)), s_discharge_out)
        CSV.write("$(out_path)/seasonal_discharge_$(sim_year).csv", DataFrame(s_discharge_out, :auto), header=false)

        s_state_out = hcat([seasonal_buses map(x -> bus_to_zone[x], seasonal_buses)], s_state_result)
        s_state_out = vcat(hcat(["bus_id" "zone"], reshape(vcat(sim_dates, "end"), 1, :)), s_state_out)
        CSV.write("$(out_path)/seasonal_state_$(sim_year).csv", DataFrame(s_state_out, :auto), header=false)
    end
end
