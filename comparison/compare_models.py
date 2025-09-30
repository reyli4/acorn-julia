#!/usr/bin/env python3
"""
Model Comparison Script

This script helps compare outputs between the original model and the new seasonal storage model.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

def load_output_files(original_dir, new_dir):
    """Load output files from both model runs"""
    original_files = {}
    new_files = {}
    
    # Load CSV files
    for file_path in Path(original_dir).glob("*.csv"):
        try:
            original_files[file_path.name] = pd.read_csv(file_path)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    for file_path in Path(new_dir).glob("*.csv"):
        try:
            new_files[file_path.name] = pd.read_csv(file_path)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    return original_files, new_files

def compare_capacity_results(original_files, new_files):
    """Compare capacity results between models"""
    print("=== CAPACITY COMPARISON ===")
    
    # Look for capacity-related files
    capacity_files = [f for f in original_files.keys() if 'capacity' in f.lower()]
    
    for file_name in capacity_files:
        if file_name in new_files:
            print(f"\nComparing {file_name}:")
            orig_df = original_files[file_name]
            new_df = new_files[file_name]
            
            # Basic comparison
            print(f"  Original shape: {orig_df.shape}")
            print(f"  New shape: {new_df.shape}")
            
            # If shapes match, compare values
            if orig_df.shape == new_df.shape:
                diff = orig_df - new_df
                print(f"  Max difference: {diff.max().max():.2f}")
                print(f"  Mean difference: {diff.mean().mean():.2f}")
            else:
                print("  Shapes don't match - detailed comparison needed")

def create_comparison_plots(original_files, new_files, output_dir):
    """Create comparison plots"""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Create capacity comparison plots
    capacity_files = [f for f in original_files.keys() if 'capacity' in f.lower()]
    
    for file_name in capacity_files:
        if file_name in new_files:
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
            
            orig_df = original_files[file_name]
            new_df = new_files[file_name]
            
            # Plot original
            if hasattr(orig_df, 'plot'):
                orig_df.plot(ax=ax1, title=f'Original Model - {file_name}')
            
            # Plot new
            if hasattr(new_df, 'plot'):
                new_df.plot(ax=ax2, title=f'New Model - {file_name}')
            
            # Plot difference
            if orig_df.shape == new_df.shape:
                diff = orig_df - new_df
                if hasattr(diff, 'plot'):
                    diff.plot(ax=ax3, title=f'Difference - {file_name}')
            
            plt.tight_layout()
            plt.savefig(output_path / f'comparison_{file_name.replace(".csv", ".png")}')
            plt.close()

def main():
    """Main comparison function"""
    original_dir = "original_model_outputs"
    new_dir = "seasonal_storage_outputs"
    output_dir = "comparison_results"
    
    print("Loading output files...")
    original_files, new_files = load_output_files(original_dir, new_dir)
    
    print(f"Found {len(original_files)} original files")
    print(f"Found {len(new_files)} new files")
    
    # Compare results
    compare_capacity_results(original_files, new_files)
    
    # Create plots
    print("\nCreating comparison plots...")
    create_comparison_plots(original_files, new_files, output_dir)
    
    print(f"\nComparison complete! Check {output_dir}/ for plots and analysis.")

if __name__ == "__main__":
    main()