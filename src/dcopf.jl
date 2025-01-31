using JuMP
using Gurobi
using CSV
using DataFrames
using MAT
using LinearAlgebra

##############################
# Set data directories
##############################
data_dir = joinpath(@__DIR__, "data")
tmp_data_dir = joinpath(@__DIR__, "data_tmp")

##############################
# MAIN
##############################
function run_model(scenario, year, gen_prop, branch_prop, bus_prop)
    # Define constants
    battduration = 8

    # Set flags
    newHVDC = true
    HydroCon = true
    tranRating = true
    networkcon = true

    # Read grid data
    gen_prop = CSV.read("$(tmp_data_dir)/grid/gen_prop_$(gen_prop).csv", DataFrame, header=true)
    bus_prop = CSV.read("$(tmp_data_dir)/grid/bus_prop_$(bus_prop).csv", DataFrame, header=true)
    branch_prop = CSV.read("$(tmp_data_dir)/grid/branch_prop_$(branch_prop).csv", DataFrame, header=true)

    bus_ids = bus_prop[:, 1]
    n_gen = size(gen_prop, 1)
    n_bus = size(bus_prop, 1)
    n_branch = size(branch_prop, 1)

    # Get scaling factors
    cc_scenario, bd_rate, ev_rate, wind_cap, solar_cap, batt_cap = read_scaling_factors(scenario)

    ############## Load ####################
    load = get_load(cc_scenario, year, ev_rate, bd_rate, bus_ids)
    load = subtract_small_hydro(load)
    load = subtract_solar_dpv(cc_scenario, year, load, solar_cap)

    ############## Supply ##############
    # Add utility solar generators
    gen_prop = add_upv_generators(gen_prop, solar_bus_ids)

    # Add wind generators to the model
    gen_prop = add_wind_generators(gen_prop, wind_bus_ids)

    # Additional generators for this study (UPDATE THIS!!)
    cleanpath1 = [36, 0, 0, 100, -100, 1, 100, 1, 1300, -1300, zeros(11)...]
    cleanpath2 = [48, 0, 0, 100, -100, 1, 100, 1, 1300, -1300, zeros(11)...]
    CHPexpress1 = [15, 0, 0, 100, -100, 1, 100, 1, 1250, -1250, zeros(11)...]
    CHPexpress2 = [48, 0, 0, 100, -100, 1, 100, 1, 1250, -1250, zeros(11)...]
    HQgen = [15, 0, 0, 100, -100, 1, 100, 1, 1250, -1250, zeros(11)...]
    gen_prop = vcat(gen_prop, reshape(HQgen, 1, :))
    gen_prop = vcat(gen_prop, reshape(cleanpath1, 1, :))
    gen_prop = vcat(gen_prop, reshape(cleanpath2, 1, :))
    gen_prop = vcat(gen_prop, reshape(CHPexpress1, 1, :))
    gen_prop = vcat(gen_prop, reshape(CHPexpress2, 1, :))

    # Get solar UPV generation
    solar_upv, solar_upv_busid = get_solar_upv(cc_scenario, year, solar_cap)

    # Get wind generation
    wind, wind_bus_ids = get_wind(cc_scenario, year, wind_cap)

    # Get generator limits
    Gmax, Gmin = get_gen_limits(gen_prop, solar_upv, wind)

    # Read hydro
    niagra_hydro, moses_saund_hydro = get_hydro(cc_scenario, year)

    # Note in the original model there is cost info: skipping for now
    # since it't not used in the 2040 analysis



    ##################################################
    ##################################################
    # Transmission interface limits
    if_scenario = 0 # DIFFERENT FROM MAIN??
    iflimup = Matrix(CSV.read("Data/Iflim/iflimup_$(year)_$(if_scenario).csv", DataFrame, header=false))
    iflimdn = Matrix(CSV.read("Data/Iflim/iflimdn_$(year)_$(if_scenario).csv", DataFrame, header=false))
    iflimup[9, :] .= iflimup[9, :] ./ 8750 .* 8450

    # Add storage data
    Storage = Matrix(CSV.read("Data/StorageData/StorageAssignment.csv", DataFrame, header=false))
    Storagebus = Storage[:, 1]
    Storagebus_indices = [findfirst(==(id), mpcreduced["bus"][:, 1]) for id in Storagebus]

    # Ramp rates
    Rdn = max.(repeat(mpc["gen"][:, 19], 1, nt) .* 2, repeat(mpc["gen"][:, 9], 1, nt))
    Rup = copy(Rdn)

    # Define constraints
    # Branch limits and power flow equations
    lineC = repeat(mpc["branch"][:, 6], 1, nt)
    lineC[lineC.==0] .= Inf

    # Define optimization variables
    model = Model(Gurobi.Optimizer)

    # Define variables using updated ngen
    @variable(model, pg[1:ngen, 1:nt])
    @variable(model, flow[1:nbranch, 1:nt])
    @variable(model, bus_angle[1:nbus, 1:nt])
    @variable(model, charge[1:length(Storagebus_indices), 1:nt])
    @variable(model, discharge[1:length(Storagebus_indices), 1:nt])
    @variable(model, battstate[1:length(Storagebus_indices), 1:nt+1])
    @variable(model, loadshedding[1:nbus, 1:nt])

    # Branch flow limits and power flow equations
    for l in 1:nbranch
        @constraint(model, -lineC[l, :] .<= flow[l, :] .<= lineC[l, :])  # Branch flow limits
        @constraint(model, flow[l, :] .== (100 / mpc["branch"][l, 4]) .*  # DC power flow equations
                                          (bus_angle[Int(mpc["branch"][l, 1]), :] .-
                                           bus_angle[Int(mpc["branch"][l, 2]), :]))
    end

    nconstraints = (nbranch * nt * 2)

    # Node balance and phase angle constraints
    for i in 1:nbus
        if mpc["bus"][i, 2] != 3  # Not the slack bus
            if i in Storagebus_indices
                # Node balance with storage devices
                storage_idx = findfirst(==(i), Storagebus_indices)
                @constraint(model, loads[i, :] .==
                                   -sum(flow[l, :] for l in findall(x -> x == i, mpc["branch"][:, 1])) .+
                                   sum(flow[l, :] for l in findall(x -> x == i, mpc["branch"][:, 2])) .+
                                   sum(pg[l, :] for l in findall(x -> x == i, mpc["gen"][:, 1])) .+
                                   discharge[storage_idx, :] .-
                                   charge[storage_idx, :] .+
                                   loadshedding[i, :])
            else
                # Node balance without storage devices
                @constraint(model, loads[i, :] .==
                                   -sum(flow[l, :] for l in findall(x -> x == i, mpc["branch"][:, 1])) .+
                                   sum(flow[l, :] for l in findall(x -> x == i, mpc["branch"][:, 2])) .+
                                   sum(pg[l, :] for l in findall(x -> x == i, mpc["gen"][:, 1])) .+
                                   loadshedding[i, :])
            end
            @constraint(model, -2 * pi .<= bus_angle[i, :] .<= 2 * pi)  # Voltage angle limits
        else  # Slack bus
            # Node balance for slack bus
            @constraint(model, loads[i, :] .==
                               -sum(flow[l, :] for l in findall(x -> x == i, mpc["branch"][:, 1])) .+
                               sum(flow[l, :] for l in findall(x -> x == i, mpc["branch"][:, 2])) .+
                               sum(pg[l, :] for l in findall(x -> x == i, mpc["gen"][:, 1])) .+
                               loadshedding[i, :])
            @constraint(model, bus_angle[i, :] .== 0.2979 / 180 * pi)  # Fix voltage angle at slack bus
        end
    end

    nconstraints += (nbus * nt * 2)

    # Storage constraints
    eff = 0.85 # Efficiency for general storage
    effGilboa = 0.75 # Efficiency for specific storage (e.g., Gilboa)
    Chargecap = batt_cap .* repeat(Storage[:, 2], 1, nt)
    storagecap = batt_cap .* battduration .* repeat(Storage[1:end-1, 2], 1, nt + 1)
    storagecap = vcat(storagecap, batt_cap * 12 * repeat(Storage[end:end, 2], 1, nt + 1))  # Adjust for last storage

    @constraint(model, 0 .<= charge .<= Chargecap)         # Charging limits
    @constraint(model, 0 .<= discharge .<= Chargecap)      # Discharging limits

    nconstraints += (size(charge, 1) * nt * 2)

    # Battery state dynamics for all time steps
    for t in 1:nt
        # Battery state dynamics for all but the last storage bus
        @constraint(model, battstate[1:end-1, t+1] .== battstate[1:end-1, t] .+ sqrt(eff) .* charge[1:end-1, t] .- (1 / sqrt(eff)) .* discharge[1:end-1, t])

        # Battery state dynamics for the last storage bus with effGilboa
        @constraint(model, battstate[end, t+1] .== battstate[end, t] .+ sqrt(effGilboa) .* charge[end, t] .- (1 / sqrt(effGilboa)) .* discharge[end, t])
    end

    # Battery capacity constraints
    @constraint(model, 0.0 .* storagecap .<= battstate .<= storagecap)

    # Initial battery state (assuming 30% of capacity)
    @constraint(model, battstate[:, 1] .== 0.3 .* storagecap[:, 1])

    nconstraints += (size(charge, 1) * nt * 2) + (size(charge, 1) * 2)

    # Interface flow constraints (if applicable)
    ifmap = mpc["if"]["map"]

    if tranRating != 0
        if !networkcon
            iflimdn[1:12, :] .= -Inf
            iflimup[1:12, :] .= Inf
        end
        for i in 1:15
            idx = ifmap[findall(==(i), ifmap[:, 1]), 2]
            idx_signs = sign.(idx)
            idx_abs = abs.(idx)
            flow_sum = [sum(idx_signs .* flow[Int.(idx_abs), t]) for t in 1:nt]
            @constraint(model, iflimdn[i, starttime:starttime+nt-1] .<= flow_sum .<= iflimup[i, starttime:starttime+nt-1])
        end
    else
        # Set iflimdn and iflimup based on mpc["if"]["lims"]
        iflimdn .= mpc["if"]["lims"][:, 2] * ones(1, nt)
        iflimup .= mpc["if"]["lims"][:, 3] * ones(1, nt)
        if !networkcon
            iflimdn[1:12, :] .= -Inf
            iflimup[1:12, :] .= Inf
        end
        for i in 1:15
            idx = ifmap[findall(==(i), ifmap[:, 1]), 2]
            idx_signs = sign.(idx)
            idx_abs = abs.(idx)
            flow_sum = [sum(idx_signs .* flow[Int.(idx_abs), t]) for t in 1:nt]
            @constraint(model, iflimdn[i, starttime:starttime+nt-1] .<= flow_sum .<= iflimup[i, starttime:starttime+nt-1])
        end
    end

    nconstraints += (15 * nt)

    # Nuclear generators always fully dispatch
    nuclear_ids = findall(x -> x == "Nuclear", mpc["genfuel"])
    for id in nuclear_ids
        @constraint(model, pg[id, :] .== Gmax[id, :])
    end

    nconstraints += (length(nuclear_ids) * nt)

    if HydroCon
        # Load the 'qm_to_numdays.csv' file into a DataFrame
        dayofqm = CSV.read("Data/qm_to_numdays.csv", DataFrame)
        nhours = dayofqm.Days .* 24  # Convert days to hours

        # Calculate the capacity rate based on mshy and generator capacity for gen(5)
        caprate = maximum(mshy ./ nhours / mpc["gen"][5, 9])
        if caprate > 1
            Gmax[5, :] .= Gmax[5, :] .* caprate
        end

        # Cumulative time counter
        ct = 0
        for i in 1:48
            # Add constraints for generator power sum based on nyhy and mshy
            @constraint(model, sum(pg[4, ct+1:ct+nhours[i]]) == nyhy[i])
            @constraint(model, sum(pg[5, ct+1:ct+nhours[i]]) == mshy[i])
            ct += nhours[i]
        end
    end

    nconstraints += (2 * 48)

    # Generator capacity constraints
    @constraint(model, Gmin .<= pg .<= Gmax)

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

    nconstraints += (length(pg)) + (7 * nt)

    # Generator ramping constraints
    @constraint(model, -Rdn[:, 2:nt] .<= pg[:, 2:nt] .- pg[:, 1:nt-1] .<= Rup[:, 2:nt])

    # Load shedding constraints
    @constraint(model, 0.0 .<= loadshedding .<= max.(loads, 0))

    nconstraints += (length(loadshedding)) + (length(pg[:, 2:nt]))

    # Extract generation for wind and calculate curtailment
    wind_indices = findall(x -> x == "Wind", mpcreduced["genfuel"])
    wg = pg[wind_indices, :]
    wc = Wind .- wg

    # Extract generation for utility-scale solar (UPV) and calculate curtailment
    solar_indices = findall(x -> x == "SolarUPV", mpcreduced["genfuel"])
    sg = pg[solar_indices, :]
    sc = SolarUPV .- sg

    # Objective function: Minimize load shedding and storage operation costs
    @objective(model, Min, sum(loadshedding) + 0.05 * (sum(charge) + sum(discharge)))

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
        battstate_result = value.(battstate)
        loadshedding_result = value.(loadshedding)
        curtailwind_result = value.(wc)
        curtailsolar_result = value.(sc)
    else
        println("Optimization did not find an optimal solution for year $(year).")
    end

    # Save results to files
    if !isdir(directory_path)
        mkdir(directory_path)
    end

    # Save results as CSV files
    CSV.write("$(directory_path)/gen_$(year).csv", DataFrame(pg_result, :auto), header=false)
    CSV.write("$(directory_path)/flow_$(year).csv", DataFrame(flow_result, :auto), header=false)
    CSV.write("$(directory_path)/charge_$(year).csv", DataFrame(charge_result, :auto), header=false)
    CSV.write("$(directory_path)/disch_$(year).csv", DataFrame(discharge_result, :auto), header=false)
    CSV.write("$(directory_path)/wc_$(year).csv", DataFrame(curtailwind_result, :auto), header=false)
    CSV.write("$(directory_path)/sc_$(year).csv", DataFrame(curtailsolar_result, :auto), header=false)
    CSV.write("$(directory_path)/battstate_$(year).csv", DataFrame(battstate_result, :auto), header=false)
    CSV.write("$(directory_path)/loadshed_$(year).csv", DataFrame(loadshedding_result, :auto), header=false)
end
;