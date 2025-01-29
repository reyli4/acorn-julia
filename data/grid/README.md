This directory holds the NYS grid network information. These files are the result of the network reduction in [Vivienne's 2021 paper](https://ieeexplore.ieee.org/document/9866561).

### Bus information
- `bus_prop_boyuan.csv`: Bus properties from [Bo's python repo](https://github.com/boyuan276/NYgrid-python)
- `bus_prop_vivienne_2050.csv`: Bus properties from [Vivenne's 2050 repo](https://github.com/AndersonEnergyLab-Cornell/ny-clcpa2050)
- Note that these files are identical other than column 3 (real power demand). They also agree with Elnaz's 2030 repo.
- *Additional bus information* is given in `npcc_new.csv`, taken from [Bo's python repo](https://github.com/boyuan276/NYgrid-python). This seems to have been amended from the original NPCC 140-bus system information in [PSAT](http://faraday1.ucd.ie/psat.html). I could not verify this other than finding the original csv file [here](https://github.com/CURENT/andes/tree/master/andes/cases).
- As far as I can tell, only the bus network connections and lat/lon locations are used. The other information does not enter into the OPF analysis.

Bus Data Format:
    1   bus number (positive integer)
    2   bus type
            PQ bus          = 1
            PV bus          = 2
            reference bus   = 3
            isolated bus    = 4
    3   Pd, real power demand (MW)
    4   Qd, reactive power demand (MVAr)
    5   Gs, shunt conductance (MW demanded at V = 1.0 p.u.)
    6   Bs, shunt susceptance (MVAr injected at V = 1.0 p.u.)
    7   area number, (positive integer)
    8   Vm, voltage magnitude (p.u.)
    9   Va, voltage angle (degrees)
(-)     (bus name)
    10  baseKV, base voltage (kV)
    11  zone, loss zone (positive integer)
(+) 12  maxVm, maximum voltage magnitude (p.u.)
(+) 13  minVm, minimum voltage magnitude (p.u.)

### Line information
- `branch_prop_boyuan.csv`: Branch properties from [Bo's python repo](https://github.com/boyuan276/NYgrid-python)
- `branch_prop_vivienne_2050.csv`: Branch properties from [Vivenne's 2050 repo](https://github.com/AndersonEnergyLab-Cornell/ny-clcpa2050)
- These files are mostly identical. There are 3 additional branches in Bo's, representing 2 external and 1 internal connections. The flow limits are also different for some lines.
- Note that both contain a duplicate (connecting buses 39-73). Elnaz 2030 agrees with Vivienne.
- Original data taken from NPCC 140-bus system as verified [here](https://github.com/CURENT/andes/tree/master/andes/cases), plus some additional data related to flow limits. 

Branch Data Format:
    1   f, from bus number
    2   t, to bus number
(-)     (circuit identifier)
    3   r, resistance (p.u.)
    4   x, reactance (p.u.)
    5   b, total line charging susceptance (p.u.)
    6   rateA, MVA rating A (long term rating)
    7   rateB, MVA rating B (short term rating)
    8   rateC, MVA rating C (emergency rating)
    9   ratio, transformer off nominal turns ratio ( = 0 for lines )
        (taps at 'from' bus, impedance at 'to' bus,
         i.e. if r = x = 0, then ratio = Vf / Vt)
    10  angle, transformer phase shift angle (degrees), positive => delay
(-)     (Gf, shunt conductance at from bus p.u.)
(-)     (Bf, shunt susceptance at from bus p.u.)
(-)     (Gt, shunt conductance at to bus p.u.)
(-)     (Bt, shunt susceptance at to bus p.u.)
    11  initial branch status, 1 - in service, 0 - out of service
(2) 12  minimum angle difference, angle(Vf) - angle(Vt) (degrees)
(2) 13  maximum angle difference, angle(Vf) - angle(Vt) (degrees)
        (The voltage angle difference is taken to be unbounded below
         if ANGMIN < -360 and unbounded above if ANGMAX > 360.
         If both parameters are zero, it is unconstrained.)

### Generator information

- `gen_prop_boyuan.csv`: Generator matrix from [Bo's python repo](https://github.com/boyuan276/NYgrid-python)
- `gen_prop_vivienne_2050.csv`: Generator matrix from [Vivenne's 2050 repo](https://github.com/AndersonEnergyLab-Cornell/ny-clcpa2050)
- This data seems to come from a variety of sources, as described in the 2019 paper. A basic list of generator names and lats/lons is given on the [NYISO website](http://mis.nyiso.com/public/) but this is appended with additional information.
- These two csv files don't really agree -- some generators can be matched across the files (mainly hydro and nuclear) but even then there are differences in generation parameters. The import "generators" do not match. 

Generator Data Format:
    1   bus number
(-)     (machine identifier, 0-9, A-Z)
    2   Pg, real power output (MW)
    3   Qg, reactive power output (MVAr)
    4   Qmax, maximum reactive power output (MVAr)
    5   Qmin, minimum reactive power output (MVAr)
    6   Vg, voltage magnitude setpoint (p.u.)
(-)     (remote controlled bus index)
    7   mBase, total MVA base of this machine, defaults to baseMVA
(-)     (machine impedance, p.u. on mBase)
(-)     (step up transformer impedance, p.u. on mBase)
(-)     (step up transformer off nominal turns ratio)
    8   status,  >  0 - machine in service
                 <= 0 - machine out of service
(-)     (% of total VAr's to come from this gen in order to hold V at
            remote bus controlled by several generators)
    9   Pmax, maximum real power output (MW)
    10  Pmin, minimum real power output (MW)
(2) 11  Pc1, lower real power output of PQ capability curve (MW)
(2) 12  Pc2, upper real power output of PQ capability curve (MW)
(2) 13  Qc1min, minimum reactive power output at Pc1 (MVAr)
(2) 14  Qc1max, maximum reactive power output at Pc1 (MVAr)
(2) 15  Qc2min, minimum reactive power output at Pc2 (MVAr)
(2) 16  Qc2max, maximum reactive power output at Pc2 (MVAr)
(2) 17  ramp rate for load following/AGC (MW/min)
(2) 18  ramp rate for 10 minute reserves (MW)
(2) 19  ramp rate for 30 minute reserves (MW)
(2) 20  ramp rate for reactive power (2 sec timescale) (MVAr/min)
(2) 21  APF, area participation factor