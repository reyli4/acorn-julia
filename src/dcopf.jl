using JuMP
using Gurobi
using CSV
using DataFrames
using MAT
using LinearAlgebra

##############################
# Set data directories
##############################
data_dir = joinpath(@__DIR__, "..", "data")
tmp_data_dir = joinpath(@__DIR__, "..", "data_tmp")

#################################
# Read uncertain scaling factors
#################################
function read_scaling_factors(scenario)
    # Get uncertain factors
    DU_f = Matrix(CSV.read("$(tmp_data_dir)/DU_factors_v3_300.csv", DataFrame, header=false))
    DU_f = DU_f[sortperm(DU_f[:, 7]), :] # sort by 7th column = CC scenario number

    if scenario == 0
        # bd_rateAE = 0.92
        # bd_rateFI = 0.92
        # bd_rateJK = 0.92
        # ev_rateAE = 0.9
        # ev_rateFI = 0.9
        # ev_rateJK = 0.9
        bd_rate = 0.92
        ev_rate = 0.9
        wind_cap = 1
        solar_cap = 1
        batt_cap = 1
    else
        cc_scenario = string(Int(DU_f[scenario, 7]))

        bd_rate = DU_f[scenario, 2]
        # bd_rateAE = bd_rate
        # bd_rateFI = bd_rate
        # bd_rateJK = bd_rate

        ev_rate = DU_f[scenario, 3]
        # ev_rateAE = ev_rate
        # ev_rateFI = ev_rate
        # ev_rateJK = ev_rate

        wind_cap = DU_f[scenario, 4]
        solar_cap = DU_f[scenario, 5]
        batt_cap = DU_f[scenario, 6]
    end

    return cc_scenario, bd_rate, ev_rate, wind_cap, solar_cap, batt_cap
end


##############################
# Getting the load data
##############################
function get_load(cc_scenario, year, ev_rate, bd_rate)
    """
    Reads and sums the four kinds of load (base, commerical, residential, EV)
    """
    # Base load
    base_load = Matrix(CSV.read("$(tmp_data_dir)/load/BaseLoad/Scenario$(cc_scenario)/simload_$(year).csv", DataFrame, header=false))
    @assert size(base_load)[1] == (365 * 24) "Base load is incorrect size"

    # EV load, only for certain buses
    ev_load = CSV.read("$(tmp_data_dir)/load/EVload/EVload_Bus.csv", DataFrame, header=false)
    @assert size(ev_load)[1] == (365 * 24) + 1 "EV load is incorrect size"
    ev_load_busids = ev_load[:, 1]

    # Residential load, for certain buses
    res_load = CSV.read("$(tmp_data_dir)/load/ResLoad/Scenario$(cc_scenario)/ResLoad_Bus_$(year).csv", DataFrame, header=false)
    @assert size(res_load)[1] == (365 * 24) + 1 "Residential load is incorrect size"
    res_load_busids = res_load[:, 1]

    # Commerical load, for certain buses
    com_load = CSV.read("$(tmp_data_dir)/load/ComLoad/Scenario$(cc_scenario)/ComLoad_Bus_$(year).csv", DataFrame, header=false)
    @assert size(com_load)[1] == (365 * 24) + 1 "Commercial load is incorrect size"
    com_load_busids = com_load[:, 1]

    # Total load
    total_load = copy(base_load)

    # Add EV load
    for i in eachindex(ev_load_busids)
        bus_idx = findfirst(==(ev_load_busids[i]), total_load[:, 1])
        total_load[bus_idx, :] .+= (ev_load[i, 2:end] * ev_rate)
    end

    # Add residential load
    for i in eachindex(res_load_busids)
        bus_idx = findfirst(==(res_load_busids[i]), total_load[:, 1])
        total_load[bus_idx, :] .+= (res_load[i, 2:end] * bd_rate)
    end

    # Add commercial load
    for i in eachindex(com_load_busids)
        bus_idx = findfirst(==(com_load_busids[i]), total_load[:, 1])
        total_load[bus_idx, :] .+= (com_load[i, 2:end] * bd_rate)
    end

    return total_load
end

############################################################
# Adding wind and solar to generator matrix
############################################################
function add_upv_generators(gen_prop, solar_bus_ids)
    # Solar generator info
    solargen = zeros(length(solar_bus_ids), 21)

    solargen[:, 1] .= solar_bus_ids # Bus number
    solargen[:, 2] .= 0 # Pg
    solargen[:, 3] .= 0 # Qg
    solargen[:, 4] .= 9999 # Qmax
    solargen[:, 5] .= -9999 # Qmin
    solargen[:, 6] .= 1 # Vg
    solargen[:, 7] .= 100 # mBase
    solargen[:, 8] .= 1 # status
    solargen[:, 9] .= 0 # Pmax
    solargen[:, 10] .= 0 # Pmin
    solargen[:, 11] .= 0 # Pc1
    solargen[:, 12] .= 0 # Pc2
    solargen[:, 13] .= 0 # Qc1min
    solargen[:, 14] .= 0 # Qc1max
    solargen[:, 15] .= 0 # Qc2min
    solargen[:, 16] .= 0 # Qc2max
    solargen[:, 17] .= Inf # ramp rate for load following/AGC
    solargen[:, 18] .= Inf # ramp rate for 10 minute reserves
    solargen[:, 19] .= Inf # ramp rate for 30 minute reserves
    solargen[:, 20] .= 0 # ramp rate for reactive power
    solargen[:, 21] .= 0 # area participation factor

    # Append to gen_prop
    return vcat(gen_prop, solargen)
end

function add_wind_generators(gen_prop, wind_bus_ids)
    # Wind generator info
    windgen = zeros(length(wind_bus_ids), 21)

    windgen[:, 1] .= wind_bus_ids # Bus number
    windgen[:, 2] .= 0 # Pg
    windgen[:, 3] .= 0 # Qg
    windgen[:, 4] .= 9999 # Qmax
    windgen[:, 5] .= -9999 # Qmin
    windgen[:, 6] .= 1 # Vg
    windgen[:, 7] .= 100 # mBase
    windgen[:, 8] .= 1 # status
    windgen[:, 9] .= 0 # Pmax
    windgen[:, 10] .= 0 # Pmin
    windgen[:, 11] .= 0 # Pc1
    windgen[:, 12] .= 0 # Pc2
    windgen[:, 13] .= 0 # Qc1min
    windgen[:, 14] .= 0 # Qc1max
    windgen[:, 15] .= 0 # Qc2min
    windgen[:, 16] .= 0 # Qc2max
    windgen[:, 17] .= Inf # ramp rate for load following/AGC
    windgen[:, 18] .= Inf # ramp rate for 10 minute reserves
    windgen[:, 19] .= Inf # ramp rate for 30 minute reserves
    windgen[:, 20] .= 0 # ramp rate for reactive power
    windgen[:, 21] .= 0 # area participation factor

    # Append to gen_prop
    return vcat(gen_prop, windgen)
end

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
    gen_prop = CSV.Read("$(tmp_data_dir)/grid/gen_prop_$(gen_prop).csv", DataFrame, header=true)
    branch_prop = CSV.Read("$(tmp_data_dir)/grid/branch_prop_$(branch_prop).csv", DataFrame, header=true)
    bus_prop = CSV.Read("$(tmp_data_dir)/grid/bus_prop_$(bus_prop).csv", DataFrame, header=true)

    # Get scaling factors
    cc_scenario, bd_rate, ev_rate, wind_cap, solar_cap, batt_cap = read_scaling_factors(scenario)

    ############## Load ####################
    load = get_load(cc_scenario, year, ev_rate, bd_rate)

    ############## Supply ##############
    # Add utility solar generators
    gen_prop = add_upv_generators(gen_prop, solar_bus_ids)

    # Add wind generators to the model
    gen_prop = add_wind_generators(gen_prop, wind_bus_ids)

    # Note in the original model there is cost info: skipping for now
    # since it't not used in the 2040 analysis

    ##################################################
    ##################################################
    # Transmission interface limits
    if_scenario = 0 # DIFFERENT FROM MAIN??
    iflimup = Matrix(CSV.read("Data/Iflim/iflimup_$(year)_$(if_scenario).csv", DataFrame, header=false))
    iflimdn = Matrix(CSV.read("Data/Iflim/iflimdn_$(year)_$(if_scenario).csv", DataFrame, header=false))
    iflimup[9, :] .= iflimup[9, :] ./ 8750 .* 8450

    # Robert-Moses Niagra hydro production, quarter monthly
    Naghydro = CSV.read("Data/hydrodata/nypaNiagaraEnergy.climate.change.csv", DataFrame)
    #  Moses-SaundersPower Dam production, quarter monthly
    Mshydro = CSV.read("Data/hydrodata/nypaMosesSaundersEnergy.climate.change.csv", DataFrame)

    # Select appropriate year and baseline scenario
    if cc_scenario != 0
        colname1 = Symbol("nypaNiagaraEnergy.$(cc_scenario)")
        colname2 = Symbol("nypaMosesSaundersEnergy.$(cc_scenario)")
    else
        colname1 = :nypaNiagaraEnergy
        colname2 = :nypaMosesSaundersEnergy
    end

    nyhy = Naghydro[Naghydro.Year.==year, colname1]
    mshy = Mshydro[Mshydro.Year.==year, colname2]

    daytime = 0
    nt = 365 * 24
    starttime = 1 + daytime * nt

    # Load power system model
    # Not sure what the details are here
    mpc_data = matread("mpc2050.mat")
    mpcreduced = mpc_data["mpcreduced"]
    nogen = size(mpcreduced["gen"], 1) # number of generators

    # Load renewable generation data: only for certain buses so record those bus ids
    SolarUPV = CSV.read("data_vivienne_hopper/Solar/Scenario$(cc_scenario)/solarUPV$(year).csv", DataFrame, header=false)
    SolarDPV = CSV.read("data_vivienne_hopper/Solar/Scenario$(cc_scenario)/solarDPV$(year).csv", DataFrame, header=false)
    Wind = CSV.read("data_vivienne_hopper/Wind/Wind$(year).csv", DataFrame, header=false)

    SolarUPV = round.(SolarUPV, digits=2)
    SolarDPV = round.(SolarDPV, digits=2)
    Wind = round.(Wind, digits=2)

    SolarUPVbus = Int.(SolarUPV[:, 1])
    SolarDPVbus = Int.(SolarDPV[:, 1])

    SolarUPV = Matrix(SolarUPV[:, (starttime+1):(starttime+nt)]) * solar_cap
    SolarDPV = Matrix(SolarDPV[:, (starttime+1):(starttime+nt)]) * solar_cap

    Windbus = Int.(Wind[:, 1])
    Wind = Matrix(Wind[:, (starttime+1):(starttime+nt)]) * wind_cap

    # Gave up trying to replicate ext2int: read instead
    mpc = copy(mpcreduced)
    mpc["gen"] = Matrix(CSV.read("./mpc_gen.csv", DataFrame, header=false))
    mpc["gen"][mpc["gen"].==-987654321] .= Inf

    mpc["branch"] = Matrix(CSV.read("./mpc_branch.csv", DataFrame, header=false))

    mpc["bus"] = Matrix(CSV.read("./mpc_bus.csv", DataFrame, header=false))

    # Additional generators
    cleanpath1 = [36, 0, 0, 100, -100, 1, 100, 1, 1300, -1300, zeros(11)...]
    cleanpath2 = [48, 0, 0, 100, -100, 1, 100, 1, 1300, -1300, zeros(11)...]
    CHPexpress1 = [15, 0, 0, 100, -100, 1, 100, 1, 1250, -1250, zeros(11)...]
    CHPexpress2 = [48, 0, 0, 100, -100, 1, 100, 1, 1250, -1250, zeros(11)...]
    HQgen = [15, 0, 0, 100, -100, 1, 100, 1, 1250, -1250, zeros(11)...]
    mpc["gen"] = vcat(mpc["gen"], reshape(HQgen, 1, :))
    mpc["gen"] = vcat(mpc["gen"], reshape(cleanpath1, 1, :))
    mpc["gen"] = vcat(mpc["gen"], reshape(cleanpath2, 1, :))
    mpc["gen"] = vcat(mpc["gen"], reshape(CHPexpress1, 1, :))
    mpc["gen"] = vcat(mpc["gen"], reshape(CHPexpress2, 1, :))

    # Prepare load data
    load_data = Matrix(newload[:, starttime:starttime+nt-1])
    ngen = size(mpc["gen"], 1)
    nbus = size(mpc["bus"], 1)
    nbranch = size(mpc["branch"], 1)

    # Get the load
    load = get_load(cc_scenario, year, ev_rate, bd_rate)

    # Adjust loads with behind-the-meter solar (SolarDPV)
    for i in eachindex(SolarDPVbus)
        bus_idx = findfirst(==(SolarDPVbus[i]), mpcreduced["bus"][:, 1])
        load_data[bus_idx, :] .-= SolarDPV[i, :]
    end

    # Small hydro negative load
    smallhydro = CSV.read("Data/hydrodata/SmallHydroCapacity.csv", DataFrame)
    smallhydrogen = CSV.read("Data/hydrodata/smallhydrogen.csv", DataFrame, header=false)
    smallhydrogen = Matrix(smallhydrogen[:, starttime:starttime+nt-1])
    smallhydrobusid = smallhydro[!, "bus index"]

    for i in eachindex(smallhydrobusid)
        bus_idx = findfirst(==(smallhydrobusid[i]), mpcreduced["bus"][:, 1])
        # println(bus_idx);
        load_data[bus_idx, :] .-= smallhydrogen[i, :]
    end

    # Add storage data
    Storage = Matrix(CSV.read("Data/StorageData/StorageAssignment.csv", DataFrame, header=false))
    Storagebus = Storage[:, 1]
    Storagebus_indices = [findfirst(==(id), mpcreduced["bus"][:, 1]) for id in Storagebus]

    # Initialize Gmax and Gmin for the generators
    Gmax = repeat(mpc["gen"][1:nogen, 9], 1, nt)
    Gmin = repeat(mpc["gen"][1:nogen, 10], 1, nt)

    # Extend Gmax and Gmin for wind generators
    Gmax = vcat(Gmax, Wind)
    Gmin = vcat(Gmin, zeros(size(Wind)))

    # Extend Gmax and Gmin for solar generators
    Gmax = vcat(Gmax, SolarUPV)
    Gmin = vcat(Gmin, zeros(size(SolarUPV)))

    # Add Gmax and Gmin for the last 12 generators
    for i in (size(mpc["gen"], 1)-12):size(mpc["gen"], 1)
        Gmax = vcat(Gmax, repeat(mpc["gen"][i:i, 9], 1, nt))
        Gmin = vcat(Gmin, repeat(mpc["gen"][i:i, 10], 1, nt))
    end

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