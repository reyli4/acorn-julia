using JuMP
using Gurobi
using CSV
using DataFrames
using MAT
using LinearAlgebra

include("./utils.jl")

##############################
# Set data directories
##############################
data_dir = joinpath(dirname(@__DIR__), "data")
tmp_data_dir = joinpath(dirname(@__DIR__), "data_tmp")

##############################
# MAIN
##############################
function run_model(scenario, year, gen_prop_name, branch_prop_name, bus_prop_name, out_path)
    # Define constants
    batt_duration = 8
    storage_eff = 0.85 # Efficiency for general storage
    gilboa_eff = 0.75 # Efficiency for specific storage (e.g., Gilboa)
    nt = 8760

    n_if_lims = 15

    # Set flags
    newHVDC = true
    HydroCon = true
    tranRating = true
    networkcon = true

    # Read grid data
    gen_prop = CSV.read("$(data_dir)/grid/gen_prop_$(gen_prop_name).csv", DataFrame, header=true)
    bus_prop = CSV.read("$(data_dir)/grid/bus_prop_$(bus_prop_name).csv", DataFrame, header=true)
    branch_prop = CSV.read("$(data_dir)/grid/branch_prop_$(branch_prop_name).csv", DataFrame, header=true)

    bus_ids = bus_prop[:, 1]

    # Get scaling factors
    cc_scenario, bd_rate, ev_rate, wind_scalar, solar_scalar, batt_scalar = read_scaling_factors(scenario)

    ############## Load ####################
    load = get_load(cc_scenario, year, ev_rate, bd_rate, bus_ids)
    load = subtract_small_hydro(load, bus_ids)
    load = subtract_solar_dpv(load, bus_ids, cc_scenario, year, solar_scalar)

    ############## Supply ##############
    # Read hydro
    niagra_hydro, moses_saund_hydro = get_hydro(cc_scenario, year)

    # Add utility solar generators
    solar_upv_gen, solar_upv_bus_ids = get_solar_upv(cc_scenario, year, solar_scalar)
    gen_prop = add_upv_generators(gen_prop, solar_upv_bus_ids)

    # Add wind generators to the model
    wind_gen, wind_bus_ids = get_wind(year, wind_scalar)
    gen_prop = add_wind_generators(gen_prop, wind_bus_ids)

    # Additional generators for this study (UPDATE THIS!!)
    cleanpath1 = similar(gen_prop, 1)
    cleanpath1[1, :] = reshape([36, 0, 0, 100, -100, 1, 100, 1, 1300, -1300, zeros(11)..., "NYCleanPath"], 1, :) # NY Clean Path

    cleanpath2 = similar(gen_prop, 1)
    cleanpath2[1, :] = reshape([48, 0, 0, 100, -100, 1, 100, 1, 1300, -1300, zeros(11)..., "NYCleanPath"], 1, :) # NY Clean Path

    CHPexpress1 = similar(gen_prop, 1)
    CHPexpress1[1, :] = reshape([15, 0, 0, 100, -100, 1, 100, 1, 1250, -1250, zeros(11)..., "CHPexpress"], 1, :) # Champlain Hudson Power Express

    CHPexpress2 = similar(gen_prop, 1)
    CHPexpress2[1, :] = reshape([48, 0, 0, 100, -100, 1, 100, 1, 1250, -1250, zeros(11)..., "CHPexpress"], 1, :) # Champlain Hudson Power Express

    HQgen = similar(gen_prop, 1)
    HQgen[1, :] = reshape([15, 0, 0, 100, -100, 1, 100, 1, 1250, -1250, zeros(11)..., "HQ"], 1, :) # HydroQuebec

    gen_prop = vcat(gen_prop, HQgen, cleanpath1, cleanpath2, CHPexpress1, CHPexpress2)

    # Get generator limits
    g_max = repeat(gen_prop[:, "PMAX"], 1, nt) # Maximum real power output (MW)
    g_min = repeat(gen_prop[:, "PMIN"], 1, nt) # Minimum real power output (MW)

    # Get generator ramp rates
    ramp_down = max.(repeat(gen_prop[:, "RAMP_30"], 1, nt) .* 2, repeat(gen_prop[:, "PMAX"], 1, nt)) # max of RAMP_30, PMAX??
    ramp_up = copy(ramp_down)

    # Note in the original model there is cost info: skipping for now
    # since it's not used in the 2040 analysis

    ############## Grid ##############
    # Transmission interface limits
    if_lim_up, if_lim_dn, if_lim_map = get_if_lims(year, n_if_lims)

    # Branch limits
    branch_lims = repeat(branch_prop[:, "RATE_A"], 1, nt)
    branch_lims[branch_lims.==0] .= 99999

    # Storage
    charge_cap, storage_cap, storage_bus_ids = get_storage(batt_scalar, batt_duration)

    ########## Optimization ##############
    n_gen = size(gen_prop, 1)
    println(n_gen)
    n_bus = size(bus_prop, 1)
    println(n_bus)
    n_branch = size(branch_prop, 1)
    println(n_branch)

    model = Model(Gurobi.Optimizer)

    ## Define variables
    @variable(model, pg[1:n_gen, 1:nt])
    @variable(model, flow[1:n_branch, 1:nt])
    @variable(model, bus_angle[1:n_bus, 1:nt])
    @variable(model, charge[1:length(storage_bus_ids), 1:nt])
    @variable(model, discharge[1:length(storage_bus_ids), 1:nt])
    @variable(model, batt_state[1:length(storage_bus_ids), 1:nt+1])
    @variable(model, load_shedding[1:n_bus, 1:nt])

    ## Constraints 
    # Branch flow limits and power flow equations
    ################ UPDATE THIS -> need to get idx from F_BUS, T_BUS
    for l in 1:n_branch
        @constraint(model, -branch_lims[l, :] .<= flow[l, :] .<= branch_lims[l, :])  # Branch flow limits
        @constraint(model, flow[l, :] .== (100 / branch_prop[l, "BR_X"]) .*  # DC power flow equations
                                          (bus_angle[Int(branch_prop[l, "F_BUS"]), :] .-
                                           bus_angle[Int(branch_prop[l, "T_BUS"]), :]))
    end

    # Node balance and phase angle constraints
    for idx in 1:n_bus
        if bus_prop[idx, "BUS_TYPE"] != 3  # Not the slack bus
            if idx in storage_bus_ids
                # Node balance with storage devices
                storage_idx = findfirst(==(idx), storage_bus_ids)
                @constraint(model, loads[idx, :] .==
                                   -sum(flow[l, :] for l in findall(x -> x == idx, branch_prop[:, "F_BUS"])) .+
                                   sum(flow[l, :] for l in findall(x -> x == idx, branch_prop[:, "T_BUS"])) .+
                                   sum(pg[l, :] for l in findall(x -> x == idx, gen_prop[:, "GEN_BUS"])) .+
                                   discharge[storage_idx, :] .-
                                   charge[storage_idx, :] .+
                                   load_shedding[idx, :])
            else
                # Node balance without storage devices
                @constraint(model, loads[idx, :] .==
                                   -sum(flow[l, :] for l in findall(x -> x == idx, branch_prop[:, "F_BUS"])) .+
                                   sum(flow[l, :] for l in findall(x -> x == idx, branch_prop[:, "T_BUS"])) .+
                                   sum(pg[l, :] for l in findall(x -> x == idx, gen_prop[:, "GEN_BUS"])) .+
                                   load_shedding[idx, :])
            end
            @constraint(model, -2 * pi .<= bus_angle[idx, :] .<= 2 * pi)  # Voltage angle limits
        else  # Slack bus
            # Node balance for slack bus
            @constraint(model, loads[idx, :] .==
                               -sum(flow[l, :] for l in findall(x -> x == idx, branch_prop[:, "F_BUS"])) .+
                               sum(flow[l, :] for l in findall(x -> x == idx, branch_prop[:, "T_BUS"])) .+
                               sum(pg[l, :] for l in findall(x -> x == idx, gen_prop[:, "GEN_BUS"])) .+
                               load_shedding[idx, :])
            @constraint(model, bus_angle[idx, :] .== 0.2979 / 180 * pi)  # Fix voltage angle at slack bus
        end
    end

    # Storage constraints
    @constraint(model, 0 .<= charge .<= charge_cap)         # Charging limits
    @constraint(model, 0 .<= discharge .<= charge_cap)      # Discharging limits

    # Battery state dynamics for all time steps
    for t in 1:nt
        # Battery state dynamics for all but the last storage bus
        @constraint(model, batt_state[1:end-1, t+1] .== batt_state[1:end-1, t] .+ sqrt(storage_eff) .* charge[1:end-1, t] .- (1 / sqrt(eff)) .* discharge[1:end-1, t])

        # Battery state dynamics Gilboa
        @constraint(model, batt_state[end, t+1] .== batt_state[end, t] .+ sqrt(gilboa_eff) .* charge[end, t] .- (1 / sqrt(effGilboa)) .* discharge[end, t])
    end

    # Battery capacity constraints
    @constraint(model, 0.0 .* storage_cap .<= batt_state .<= storage_cap)

    # Initial battery state (assuming 30% of capacity)
    @constraint(model, batt_state[:, 1] .== 0.3 .* storage_cap[:, 1])

    # Interface flow constraints
    # Internal limits set to infinity if desired
    if !networkcon
        if_lim_dn[1:12, :] .= -Inf
        if_lim_up[1:12, :] .= Inf
    end

    # Impose interface limits
    for i in 1:n_if_lims
        # Sum flow across the interfaces
        idx = if_lim_map[findall(==(i), if_lim_map[:, "IF_ID"]), "BUS_ID"]
        idx_signs = sign.(idx)
        idx_abs = abs.(idx)
        flow_sum = [sum(idx_signs .* flow[Int.(idx_abs), t]) for t in 1:nt]
        # Constraint
        @constraint(model, if_lim_dn[i, 1:nt-1] .<= flow_sum .<= if_lim_up[i, 1:nt-1])
    end

    # Nuclear generators always fully dispatch
    nuclear_idx = findall(x -> x == "Nuclear", gen_prop[!, "UNIT_TYPE"])
    for idx in nuclear_idx
        @constraint(model, pg[idx, :] .== g_max[idx, :])
    end

    # Hydro generators always fully dispatch
    niagra_idx = findfirst(x -> x == "Niagra", gen_prop[!, "UNIT_TYPE"])
    moses_saund_idx = findfirst(x -> x == "MosesSaunders", gen_prop[!, "UNIT_TYPE"])
    if HydroCon
        # Load the 'qm_to_numdays.csv' file into a DataFrame
        dayofqm = CSV.read("$(tmp_data_dir)/qm_to_numdays.csv", DataFrame)
        nhours = dayofqm.Days .* 24  # Convert days to hours

        # Calculate the capacity rate of Moses Saunders
        hydro_pmax = gen_prop[moses_saund_idx, "PMAX"]
        cap_rate = maximum(moses_saund_hydro ./ nhours / hydro_pmax)
        if cap_rate > 1
            g_max[moses_saund_idx, :] .= g_max[moses_saund_idx, :] .* cap_rate
        end

        # Cumulative time counter
        ct = 0
        for t in 1:48
            # Add constraints for generator power sum based on nyhy and moses_saund_hydro
            @constraint(model, sum(pg[niagra_idx, ct+1:ct+nhours[t]]) == niagra_hydro[t])
            @constraint(model, sum(pg[moses_saund_idx, ct+1:ct+nhours[t]]) == moses_saund_hydro[t])
            ct += nhours[t]
        end
    end

    # Generator capacity constraints
    @constraint(model, g_min .<= pg .<= g_max)

    # HVDC lines (modelled as two dummy generators on each side of the lines)
    @constraint(model, pg[end-12, :] .== -pg[end-8, :])
    @constraint(model, pg[end-11, :] .== -pg[end-7, :])
    @constraint(model, pg[end-10, :] .== -pg[end-6, :])
    @constraint(model, pg[end-9, :] .== -pg[end-5, :])
    @constraint(model, pg[end-3, :] .== -pg[end-2, :])
    @constraint(model, pg[end-1, :] .== -pg[end, :])
    @constraint(model, pg[end-4, :] .== -pg[end-1, :])

    if !newHVDC
        @constraint(model, pg[end-3, :] .== 0)
        @constraint(model, pg[end-1, :] .== 0)
    end

    # Generator ramping constraints
    @constraint(model, -ramp_down[:, 2:nt] .<= pg[:, 2:nt] .- pg[:, 1:nt-1] .<= ramp_up[:, 2:nt])

    # Load shedding constraints
    @constraint(model, 0.0 .<= load_shedding .<= max.(loads, 0))

    # Extract generation for wind and calculate curtailment
    wind_idx = findall(x -> x == "Wind", gen_prop[!, "GEN_TYPE"])
    wg = pg[wind_indices, :]
    wc = wind_gen .- wg

    # Extract generation for utility-scale solar (UPV) and calculate curtailment
    solar_idx = findall(x -> x == "SolarUPV", gen_prop[!, "GEN_TYPE"])
    sg = pg[solar_idx, :]
    sc = solar_upv_gen .- sg

    # Objective function: Minimize load shedding and storage operation costs
    @objective(model, Min, sum(load_shedding) + 0.05 * (sum(charge) + sum(discharge)))

    # SOLVE
    optimize!(model)

    # Check if the solver found an optimal solution
    if termination_status(model) == MOI.OPTIMAL
        # Extract results
        pg_result = value.(pg)
        flow_result = value.(flow)
        angle_result = value.(bus_angle) .* (180 / pi)  # Convert angles to degrees;
        charge_result = value.(charge)
        discharge_result = value.(discharge)
        batt_state_result = value.(batt_state)
        load_shedding_result = value.(load_shedding)
        wind_curtail_result = value.(wc)
        solar_curtail_result = value.(sc)
    else
        println("Optimization did not find an optimal solution for year $(year).")
    end

    # Save results to files
    if !isdir(out_path)
        mkdir(out_path)
    end

    # Save results as CSV files
    CSV.write("$(out_path)/gen_$(year).csv", DataFrame(pg_result, :auto), header=false)
    CSV.write("$(out_path)/flow_$(year).csv", DataFrame(flow_result, :auto), header=false)
    CSV.write("$(out_path)/charge_$(year).csv", DataFrame(charge_result, :auto), header=false)
    CSV.write("$(out_path)/disch_$(year).csv", DataFrame(discharge_result, :auto), header=false)
    CSV.write("$(out_path)/wind_curtailment_$(year).csv", DataFrame(wind_curtail_result, :auto), header=false)
    CSV.write("$(out_path)/solar_curtailment_$(year).csv", DataFrame(solar_curtail_result, :auto), header=false)
    CSV.write("$(out_path)/batt_state_$(year).csv", DataFrame(batt_state_result, :auto), header=false)
    CSV.write("$(out_path)/load_shedding_$(year).csv", DataFrame(load_shedding_result, :auto), header=false)
end
;