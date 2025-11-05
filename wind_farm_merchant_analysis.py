#!/usr/bin/env python3
"""
Wind Farm Merchant Analysis
Problem Set 2 Modification - Merchant Wind Farm Financial Performance

This script analyzes a 100 MW wind farm operating on a merchant basis
with uncertain capacity factors and spot market prices.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# Set random seed for reproducibility
np.random.seed(42)

class WindFarmMerchantAnalysis:
    def __init__(self):
        # Wind farm parameters
        self.capacity_mw = 100  # MW
        self.capacity_kw = self.capacity_mw * 1000  # kW
        
        # Capacity factor parameters (normal distribution)
        self.cf_mean = 0.42  # 42%
        self.cf_std = 0.04   # 4%
        
        # Cost parameters
        self.capital_cost_per_kw = 1400  # $/kW
        self.fixed_opex_per_kw_yr = 40   # $/kW-yr
        
        # Loan parameters
        self.loan_term_years = 25
        self.interest_rate = 0.05  # 5%
        
        # Spot market price components (all normal distributions)
        self.ng_price_mean = 3.50  # $/MMBtu
        self.ng_price_std = 0.35  # $/MMBtu
        
        self.heat_rate_mean = 10.0  # MMBtu/MWh
        self.heat_rate_std = 0.5   # MMBtu/MWh
        
        self.nodal_scalar_mean = 1.0
        self.nodal_scalar_std = 0.03
        
        # Expected price calculation
        self.expected_price = self.ng_price_mean * self.heat_rate_mean * self.nodal_scalar_mean
        
        # Number of Monte Carlo samples
        self.n_samples = 1000
        
        print(f"Wind Farm Capacity: {self.capacity_mw} MW")
        print(f"Expected Spot Price: ${self.expected_price:.2f}/MWh")
        print(f"Expected Capacity Factor: {self.cf_mean:.1%}")
        print(f"Monte Carlo Samples: {self.n_samples:,}")
        print("-" * 50)
    
    def run_monte_carlo_simulation(self):
        """Run Monte Carlo simulation for all random variables"""
        print("Running Monte Carlo simulation...")
        
        # Generate random samples for all variables
        cf_samples = np.random.normal(self.cf_mean, self.cf_std, self.n_samples)
        ng_price_samples = np.random.normal(self.ng_price_mean, self.ng_price_std, self.n_samples)
        heat_rate_samples = np.random.normal(self.heat_rate_mean, self.heat_rate_std, self.n_samples)
        nodal_scalar_samples = np.random.normal(self.nodal_scalar_mean, self.nodal_scalar_std, self.n_samples)
        
        # Calculate spot prices
        spot_prices = ng_price_samples * heat_rate_samples * nodal_scalar_samples
        
        # Calculate annual generation (MWh)
        annual_generation = cf_samples * self.capacity_mw * 8760  # 8760 hours per year
        
        # Calculate annual electricity revenue
        annual_revenue = annual_generation * spot_prices
        
        # Store results
        self.results = pd.DataFrame({
            'capacity_factor': cf_samples,
            'ng_price': ng_price_samples,
            'heat_rate': heat_rate_samples,
            'nodal_scalar': nodal_scalar_samples,
            'spot_price': spot_prices,
            'annual_generation_mwh': annual_generation,
            'annual_revenue': annual_revenue
        })
        
        return self.results
    
    def calculate_cfads(self, debt_size=0):
        """Calculate Cash Flow Available for Debt Service (CFADS)"""
        if debt_size == 0:
            # Calculate maximum debt capacity first
            debt_size = self.calculate_debt_capacity()
        
        # Annual debt service
        annual_debt_service = self.calculate_annual_debt_service(debt_size)
        
        # Calculate CFADS for each scenario
        cfads = self.results['annual_revenue'] - self.fixed_opex_per_kw_yr * self.capacity_kw - annual_debt_service
        
        return cfads, annual_debt_service
    
    def calculate_annual_debt_service(self, debt_size):
        """Calculate annual debt service payment"""
        if debt_size == 0:
            return 0
        
        # Annual payment using PMT formula
        annual_payment = debt_size * (self.interest_rate * (1 + self.interest_rate)**self.loan_term_years) / \
                        ((1 + self.interest_rate)**self.loan_term_years - 1)
        
        return annual_payment
    
    def calculate_debt_capacity(self, target_dscr=2.25):
        """Calculate maximum debt capacity given target DSCR at P-50 level"""
        # First, calculate CFADS without debt
        cfads_no_debt = self.results['annual_revenue'] - self.fixed_opex_per_kw_yr * self.capacity_kw
        
        # P-50 CFADS (median)
        p50_cfads = np.percentile(cfads_no_debt, 50)
        
        # Maximum annual debt service that maintains target DSCR
        max_annual_debt_service = p50_cfads / target_dscr
        
        # Calculate debt capacity from annual payment
        if max_annual_debt_service <= 0:
            return 0
        
        # Solve for debt size using present value of annuity formula
        # PV = PMT * [(1 - (1 + r)^-n) / r]
        pv_factor = (1 - (1 + self.interest_rate)**(-self.loan_term_years)) / self.interest_rate
        debt_capacity = max_annual_debt_service * pv_factor
        
        return debt_capacity
    
    def analyze_results(self):
        """Analyze and display results"""
        print("\n" + "="*60)
        print("MONTE CARLO SIMULATION RESULTS")
        print("="*60)
        
        # Part (a): Average electricity revenue
        avg_revenue = self.results['annual_revenue'].mean()
        print(f"\n(a) Average Electricity Revenue: ${avg_revenue:,.0f}")
        
        # P-50 revenue from problem set 2 (assuming same as expected)
        p50_expected_revenue = self.cf_mean * self.capacity_mw * 8760 * self.expected_price
        print(f"    P-50 Revenue (Problem Set 2): ${p50_expected_revenue:,.0f}")
        print(f"    Difference: ${avg_revenue - p50_expected_revenue:,.0f} ({((avg_revenue/p50_expected_revenue - 1)*100):+.1f}%)")
        
        # Part (b): P-50 and P-99 CFADS
        cfads, annual_debt_service = self.calculate_cfads()
        p50_cfads = np.percentile(cfads, 50)
        p99_cfads = np.percentile(cfads, 99)
        
        print(f"\n(b) CFADS Analysis:")
        print(f"    P-50 CFADS: ${p50_cfads:,.0f}")
        print(f"    P-99 CFADS: ${p99_cfads:,.0f}")
        
        # Part (c): Debt capacity with 2.25x DSCR
        debt_capacity = self.calculate_debt_capacity(target_dscr=2.25)
        print(f"\n(c) Debt Capacity (2.25x DSCR at P-50): ${debt_capacity:,.0f}")
        
        # Part (d): Equity percentage
        total_project_cost = self.capital_cost_per_kw * self.capacity_kw
        equity_required = total_project_cost - debt_capacity
        equity_percentage = (equity_required / total_project_cost) * 100
        
        print(f"\n(d) Equity Analysis:")
        print(f"    Total Project Cost: ${total_project_cost:,.0f}")
        print(f"    Debt Capacity: ${debt_capacity:,.0f}")
        print(f"    Equity Required: ${equity_required:,.0f}")
        print(f"    Equity Percentage: {equity_percentage:.1f}%")
        
        # Comparison to contracted case (assuming 100% debt in contracted case)
        contracted_debt_percentage = 100.0
        print(f"    Contracted Case Debt %: {contracted_debt_percentage:.1f}%")
        print(f"    Difference: {equity_percentage - (100 - contracted_debt_percentage):+.1f} percentage points")
        
        return {
            'avg_revenue': avg_revenue,
            'p50_cfads': p50_cfads,
            'p99_cfads': p99_cfads,
            'debt_capacity': debt_capacity,
            'equity_percentage': equity_percentage
        }
    
    def create_visualizations(self):
        """Create visualizations of the results"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Revenue distribution
        axes[0, 0].hist(self.results['annual_revenue'], bins=50, alpha=0.7, color='blue')
        axes[0, 0].axvline(self.results['annual_revenue'].mean(), color='red', linestyle='--', 
                          label=f'Mean: ${self.results["annual_revenue"].mean():,.0f}')
        axes[0, 0].set_xlabel('Annual Revenue ($)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Distribution of Annual Electricity Revenue')
        axes[0, 0].legend()
        
        # Spot price distribution
        axes[0, 1].hist(self.results['spot_price'], bins=50, alpha=0.7, color='green')
        axes[0, 1].axvline(self.results['spot_price'].mean(), color='red', linestyle='--',
                          label=f'Mean: ${self.results["spot_price"].mean():.2f}/MWh')
        axes[0, 1].set_xlabel('Spot Price ($/MWh)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Distribution of Spot Market Prices')
        axes[0, 1].legend()
        
        # Capacity factor distribution
        axes[1, 0].hist(self.results['capacity_factor'], bins=50, alpha=0.7, color='orange')
        axes[1, 0].axvline(self.results['capacity_factor'].mean(), color='red', linestyle='--',
                          label=f'Mean: {self.results["capacity_factor"].mean():.1%}')
        axes[1, 0].set_xlabel('Capacity Factor')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Distribution of Capacity Factors')
        axes[1, 0].legend()
        
        # CFADS distribution
        cfads, _ = self.calculate_cfads()
        axes[1, 1].hist(cfads, bins=50, alpha=0.7, color='purple')
        axes[1, 1].axvline(np.percentile(cfads, 50), color='red', linestyle='--',
                          label=f'P-50: ${np.percentile(cfads, 50):,.0f}')
        axes[1, 1].axvline(np.percentile(cfads, 99), color='orange', linestyle='--',
                          label=f'P-99: ${np.percentile(cfads, 99):,.0f}')
        axes[1, 1].set_xlabel('CFADS ($)')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Distribution of CFADS')
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig('wind_farm_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def export_to_excel(self, filename='wind_farm_merchant_analysis.xlsx'):
        """Export results to Excel for further analysis"""
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            # Export raw simulation data
            self.results.to_excel(writer, sheet_name='Simulation_Data', index=False)
            
            # Export summary statistics
            summary_data = {
                'Metric': [
                    'Average Revenue',
                    'P-50 Revenue',
                    'P-99 Revenue',
                    'Average Spot Price',
                    'P-50 Spot Price',
                    'P-99 Spot Price',
                    'Average Capacity Factor',
                    'P-50 Capacity Factor',
                    'P-99 Capacity Factor',
                    'P-50 CFADS',
                    'P-99 CFADS',
                    'Debt Capacity',
                    'Equity Percentage'
                ],
                'Value': [
                    f"${self.results['annual_revenue'].mean():,.0f}",
                    f"${np.percentile(self.results['annual_revenue'], 50):,.0f}",
                    f"${np.percentile(self.results['annual_revenue'], 99):,.0f}",
                    f"${self.results['spot_price'].mean():.2f}/MWh",
                    f"${np.percentile(self.results['spot_price'], 50):.2f}/MWh",
                    f"${np.percentile(self.results['spot_price'], 99):.2f}/MWh",
                    f"{self.results['capacity_factor'].mean():.1%}",
                    f"{np.percentile(self.results['capacity_factor'], 50):.1%}",
                    f"{np.percentile(self.results['capacity_factor'], 99):.1%}",
                    f"${np.percentile(self.calculate_cfads()[0], 50):,.0f}",
                    f"${np.percentile(self.calculate_cfads()[0], 99):,.0f}",
                    f"${self.calculate_debt_capacity():,.0f}",
                    f"{((self.capital_cost_per_kw * self.capacity_kw - self.calculate_debt_capacity()) / (self.capital_cost_per_kw * self.capacity_kw) * 100):.1f}%"
                ]
            }
            
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
        
        print(f"\nResults exported to {filename}")

def main():
    """Main analysis function"""
    print("Wind Farm Merchant Analysis")
    print("=" * 50)
    
    # Initialize analysis
    analysis = WindFarmMerchantAnalysis()
    
    # Run Monte Carlo simulation
    results = analysis.run_monte_carlo_simulation()
    
    # Analyze results
    summary = analysis.analyze_results()
    
    # Create visualizations
    analysis.create_visualizations()
    
    # Export to Excel
    analysis.export_to_excel()
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main()