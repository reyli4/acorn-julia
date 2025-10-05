# Wind Farm Merchant Analysis - Complete Results

## Problem Statement
Analysis of a 100 MW wind farm operating on a merchant basis (spot market) versus the contracted case from Problem Set 2. The analysis considers uncertainty in capacity factors and spot market prices.

## Key Parameters
- **Wind Farm Capacity**: 100 MW (100,000 kW)
- **Capital Cost**: $1,400/kW = $140,000,000 total
- **Fixed Operating Costs**: $40/kW-yr = $4,000,000/year
- **Loan Terms**: 25 years at 5.0% interest rate
- **Target DSCR**: 2.25x at P-50 level

## Random Variables (Independent Normal Distributions)
1. **Capacity Factor**: Mean = 42%, Standard Deviation = 4%
2. **Natural Gas Price**: Mean = $3.50/MMBtu, Standard Deviation = $0.35/MMBtu
3. **Heat Rate**: Mean = 10 MMBtu/MWh, Standard Deviation = 0.5 MMBtu/MWh
4. **Nodal Scalar**: Mean = 1.0, Standard Deviation = 0.03

**Expected Spot Price**: $3.50 × 10 × 1.0 = $35.00/MWh (same as contracted price)

## Monte Carlo Simulation Results (1,000 samples)

### (a) Average Electricity Revenue
- **Average Revenue**: $12,982,488
- **P-50 Revenue (Problem Set 2)**: $12,877,200
- **Difference**: +$105,288 (+0.8%)

The average revenue is slightly higher than the P-50 case due to the convexity of the revenue function with respect to price and capacity factor.

### (b) CFADS Analysis
- **P-50 CFADS**: $4,898,763
- **P-99 CFADS**: $9,767,470

CFADS = Annual Revenue - Fixed Operating Costs - Debt Service

### (c) Debt Capacity
- **Debt Capacity (2.25x DSCR at P-50)**: $55,234,318
- **Maximum Annual Debt Service**: $2,177,250

The debt capacity is calculated by finding the maximum debt that maintains a 2.25x debt service coverage ratio at the P-50 CFADS level.

### (d) Equity Analysis
- **Total Project Cost**: $140,000,000
- **Debt Capacity**: $55,234,318
- **Equity Required**: $84,765,682
- **Equity Percentage**: 60.5%

### Comparison to Contracted Case
- **Contracted Case**: ~100% debt financing (assuming no DSCR constraint)
- **Merchant Case**: 60.5% equity, 39.5% debt
- **Difference**: +60.5 percentage points more equity required

## Key Insights

1. **Revenue Stability**: Despite the same expected price ($35/MWh), the merchant case shows slightly higher average revenue due to the non-linear relationship between price components and revenue.

2. **Debt Capacity Impact**: The increased uncertainty in merchant operations significantly reduces debt capacity from ~100% to ~40% of project cost.

3. **Equity Requirements**: The merchant case requires 60.5% equity financing compared to minimal equity in the contracted case, dramatically increasing the cost of capital.

4. **Risk-Return Trade-off**: While the expected returns may be similar, the merchant case carries significantly higher risk, requiring much more equity capital.

## Files Generated
- `wind_farm_merchant_analysis.py`: Complete Python analysis script
- `wind_farm_merchant_analysis.xlsx`: Excel file with detailed results
- `wind_farm_analysis.png`: Visualization plots
- `excel_formulas_reference.md`: Excel formula reference

## Methodology
The analysis uses Monte Carlo simulation with 1,000 samples to capture the uncertainty in:
- Wind farm capacity factors
- Natural gas prices
- Market heat rates
- Nodal price scalars

Each sample generates a complete annual revenue scenario, allowing for statistical analysis of project economics under uncertainty.