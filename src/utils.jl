using CSV
using DataFrames

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
# Load
##############################
function get_load(cc_scenario, year, ev_rate, bd_rate, bus_ids, nt=8760)
    """
    Reads and sums the four kinds of load (base, commerical, residential, EV)

    NOTE:
    - In the original code, the building and EV loads can be scaled by zone-specific rates; this is simplified here

    """
    # Base load
    base_load = Matrix(CSV.read("$(tmp_data_dir)/load/BaseLoad/Scenario$(cc_scenario)/simload_$(year).csv", DataFrame, header=false))
    # base_load = hcat(bus_ids, base_load) # prepend bus_ids (UPDATE THIS!)
    @assert size(base_load, 2) == nt "Base load is incorrect size"

    # EV load, only for certain buses
    ev_load = Matrix(CSV.read("$(tmp_data_dir)/load/EVload/EVload_Bus.csv", DataFrame, header=false))
    @assert size(ev_load, 2) == nt + 1 "EV load is incorrect size"
    ev_load_busid = ev_load[:, 1]

    # Residential load, for certain buses
    res_load = Matrix(CSV.read("$(tmp_data_dir)/load/ResLoad/Scenario$(cc_scenario)/ResLoad_Bus_$(year).csv", DataFrame, header=false))
    @assert size(res_load, 2) == nt + 1 "Residential load is incorrect size"
    res_load_busid = res_load[:, 1]

    # Commerical load, for certain buses
    com_load = Matrix(CSV.read("$(tmp_data_dir)/load/ComLoad/Scenario$(cc_scenario)/ComLoad_Bus_$(year).csv", DataFrame, header=false))
    @assert size(com_load, 2) == nt + 1 "Commercial load is incorrect size"
    com_load_busid = com_load[:, 1]

    # Total load
    total_load = copy(base_load)

    # Add EV load
    for i in eachindex(ev_load_busid)
        bus_idx = findfirst(==(ev_load_busid[i]), bus_ids)
        total_load[bus_idx, :] .+= (ev_load[i, 2:end] .* ev_rate)
    end

    # Add residential load
    for i in eachindex(res_load_busid)
        bus_idx = findfirst(==(res_load_busid[i]), bus_ids)
        total_load[bus_idx, :] .+= (res_load[i, 2:end] .* bd_rate)
    end

    # Add commercial load
    for i in eachindex(com_load_busid)
        bus_idx = findfirst(==(com_load_busid[i]), bus_ids)
        total_load[bus_idx, :] .+= (com_load[i, 2:end] .* bd_rate)
    end

    return total_load
end

function subtract_solar_dpv(load, bus_ids, cc_scenario, year, solar_cap, nt=8760)
    """
    Adjusts the load data with behind-the-meter solar (SolarDPV)
    """
    # Load renewable generation data: only for certain buses so record those bus ids
    solar_dpv = Matrix(CSV.read("$(tmp_data_dir)/gen/Solar/Scenario$(cc_scenario)/solarDPV$(year).csv", DataFrame, header=false))
    @assert size(solar_dpv, 2) == nt + 1 "Solar DPV is incorrect size"
    solar_dpv_busid = Int.(solar_dpv[:, 1])

    # Adjust loads with behind-the-meter solar
    for i in eachindex(solar_dpv_busid)
        bus_idx = findfirst(==(solar_dpv_busid[i]), bus_ids)
        load[bus_idx, :] .-= (solar_dpv[i, 2:end] .* solar_cap)
    end

    return load
end

function subtract_small_hydro(load, bus_ids, nt=8760)
    """
    Adjusts the load data with small hydro generation
    """
    # Read small hydro
    small_hydro_gen = Matrix(CSV.read("$(tmp_data_dir)/hydrodata/smallhydrogen.csv", DataFrame, header=false))
    @assert size(small_hydro_gen, 2) == nt "Small hydro generation is incorrect size"
    # Read small hydro bus ids (UPDATE THIS!!)
    small_hydro_busid = CSV.read("$(tmp_data_dir)/hydrodata/SmallHydroCapacity.csv", DataFrame)[!, "bus index"]

    # Subtract from existing load
    for i in eachindex(small_hydro_busid)
        bus_idx = findfirst(==(small_hydro_busid[i]), bus_ids)
        load[bus_idx, :] .-= small_hydro_gen[i, :]
    end

    return load
end

############################################################
# Generation
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
    solargen[:, 22] .= "SolarUPV" # generation type

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
    windgen[:, 22] .= "Wind" # generation type

    # Append to gen_prop
    return vcat(gen_prop, windgen)
end

function get_hydro(cc_scenario, year)
    # Robert-Moses Niagra hydro production, quarter monthly
    niagra_hydro = CSV.read("$(tmp_data_dir)/hydrodata/nypaNiagaraEnergy.climate.change.csv", DataFrame)
    #  Moses-SaundersPower Dam production, quarter monthly
    moses_saund_hydro = CSV.read("$(tmp_data_dir)/hydrodata/nypaMosesSaundersEnergy.climate.change.csv", DataFrame)

    # Select appropriate year and baseline scenario
    if cc_scenario != 0
        colname1 = Symbol("nypaNiagaraEnergy.$(cc_scenario)")
        colname2 = Symbol("nypaMosesSaundersEnergy.$(cc_scenario)")
    else
        colname1 = :nypaNiagaraEnergy
        colname2 = :nypaMosesSaundersEnergy
    end

    niagra_hydro = niagra_hydro[niagra_hydro.Year.==year, colname1]
    moses_saund_hydro = moses_saund_hydro[moses_saund_hydro.Year.==year, colname2]

    return niagra_hydro, moses_saund_hydro
end

function get_solar_upv(cc_scenario, year, solar_cap, nt=8760)
    # SolarUPV generation data
    solar_upv = CSV.read("$(tmp_data_dir)/gen/Solar/Scenario$(cc_scenario)/solarUPV$(year).csv", DataFrame, header=false)
    @assert size(solar_upv, 2) == nt + 1 "Solar UPV is incorrect size"
    solar_upv = Matrix(solar_upv[:, 1:end]) * solar_cap
    solar_upv_busid = Int.(solar_upv[:, 1])
    return solar_upv, solar_upv_busid
end

function get_wind(cc_scenario, year, wind_cap, nt=8760)
    # Wind generation data
    wind = CSV.read("$(tmp_data_dir)/gen/Wind/Scenario$(cc_scenario)/wind$(year).csv", DataFrame, header=false)
    @assert size(wind, 2) == nt + 1 "Wind is incorrect size"
    wind = Matrix(wind[:, 1:end]) * wind_cap
    wind_busid = Int.(wind[:, 1])
    return wind, wind_busid
end

function get_gen_limits(gen_prop, solar_upv, wind, nt=8760)
    # Make sure no solar or wind in gen_prop
    gen_prop_temp = gen_prop[gen_prop[:, 22].!="SolarUPV", :]
    gen_prop_temp = gen_prop_temp[gen_prop_temp[:, 22].!="Wind", :]

    # Initialize Gmax and Gmin for the generators
    Gmax = repeat(gen_prop_temp[:, "PMAX"], 1, nt) # Maximum real power output (MW)
    Gmin = repeat(gen_prop_temp[:, "PMIN"], 1, nt) # Minimum real power output (MW)

    # Extend Gmax and Gmin for solar generators
    Gmax = vcat(Gmax, solar_upv)
    Gmin = vcat(Gmin, zeros(size(solar_upv)))

    # Extend Gmax and Gmin for wind generators
    Gmax = vcat(Gmax, wind)
    Gmin = vcat(Gmin, zeros(size(wind)))

    # # Add Gmax and Gmin for the last 12 generators
    # for i in (size(mpc["gen"], 1)-12):size(mpc["gen"], 1)
    #     Gmax = vcat(Gmax, repeat(mpc["gen"][i:i, 9], 1, nt))
    #     Gmin = vcat(Gmin, repeat(mpc["gen"][i:i, 10], 1, nt))
    # end

    return Gmax, Gmin
end