# Model Comparison Directory

This directory contains outputs from both the original model and the new seasonal storage model for comparison purposes.

## Directory Structure

- `original_model_outputs/` - Output files from the original repo model (copied from cluster)
- `seasonal_storage_outputs/` - Output files from the new seasonal storage model
- `comparison_scripts/` - Scripts for analyzing differences between models

## How to Use

1. **Copy original model outputs**: Copy your cluster output files to `original_model_outputs/`
2. **Run seasonal storage model**: Run the new model and save outputs to `seasonal_storage_outputs/`
3. **Run comparison**: Use the comparison scripts to analyze differences

## Expected Output Files

For the "low RE high electric" scenario, you should have:
- Capacity results (CSV/Excel files)
- Generation time series
- Cost results
- Any other relevant output files from your original model run

## Notes

- Keep the same file naming conventions for easier comparison
- Document any differences in model configurations
- Save comparison plots and analysis results in this directory