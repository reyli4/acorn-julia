# Excel Formulas for Wind Farm Merchant Analysis

## Key Parameters
- Wind Farm Capacity: 100 MW = 100,000 kW
- Capital Cost: $1,400/kW
- Fixed Opex: $40/kW-yr
- Loan Term: 25 years
- Interest Rate: 5.0%
- Target DSCR: 2.25x

## Random Variables (Normal Distribution)
1. **Capacity Factor**: Mean = 42%, Std Dev = 4%
2. **Natural Gas Price**: Mean = $3.50/MMBtu, Std Dev = $0.35/MMBtu
3. **Heat Rate**: Mean = 10 MMBtu/MWh, Std Dev = 0.5 MMBtu/MWh
4. **Nodal Scalar**: Mean = 1.0, Std Dev = 0.03

## Excel Formulas for Monte Carlo Simulation

### Column A: Sample Number (1 to 1000)
```
A1: 1
A2: =A1+1
(Drag down to A1000)
```

### Column B: Capacity Factor
```
B1: =NORM.INV(RAND(),0.42,0.04)
(Drag down to B1000)
```

### Column C: Natural Gas Price
```
C1: =NORM.INV(RAND(),3.50,0.35)
(Drag down to C1000)
```

### Column D: Heat Rate
```
D1: =NORM.INV(RAND(),10,0.5)
(Drag down to D1000)
```

### Column E: Nodal Scalar
```
E1: =NORM.INV(RAND(),1.0,0.03)
(Drag down to E1000)
```

### Column F: Spot Price ($/MWh)
```
F1: =C1*D1*E1
(Drag down to F1000)
```

### Column G: Annual Generation (MWh)
```
G1: =B1*100*8760
(Drag down to G1000)
```

### Column H: Annual Revenue ($)
```
H1: =G1*F1
(Drag down to H1000)
```

### Column I: Fixed Opex ($)
```
I1: =40*100000
(Same for all rows)
```

### Column J: CFADS without Debt ($)
```
J1: =H1-I1
(Drag down to J1000)
```

## Key Results Calculations

### Average Revenue
```
=AVERAGE(H1:H1000)
```

### P-50 Revenue
```
=PERCENTILE(H1:H1000,0.5)
```

### P-99 Revenue
```
=PERCENTILE(H1:H1000,0.99)
```

### P-50 CFADS
```
=PERCENTILE(J1:J1000,0.5)
```

### P-99 CFADS
```
=PERCENTILE(J1:J1000,0.99)
```

### Maximum Annual Debt Service (for 2.25x DSCR)
```
=PERCENTILE(J1:J1000,0.5)/2.25
```

### Debt Capacity (Present Value of Annuity)
```
=PMT(0.05,25,-1)*PERCENTILE(J1:J1000,0.5)/2.25
```

### Total Project Cost
```
=1400*100000
```

### Equity Required
```
=140000000-[Debt Capacity]
```

### Equity Percentage
```
=[Equity Required]/140000000
```

## Results Summary
Based on the Python analysis with 1,000 samples:

- **Average Electricity Revenue**: $12,982,488
- **P-50 Revenue**: $12,877,200 (same as contracted case)
- **P-50 CFADS**: $4,898,763
- **P-99 CFADS**: $9,767,470
- **Debt Capacity**: $55,234,318
- **Equity Required**: $84,765,682
- **Equity Percentage**: 60.5%

## Comparison to Contracted Case
- Contracted case: ~100% debt financing
- Merchant case: 60.5% equity, 39.5% debt
- The increased uncertainty in merchant operations significantly reduces debt capacity and increases equity requirements.