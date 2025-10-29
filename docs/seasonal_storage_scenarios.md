# Seasonal storage scenario design and comparison plan

## 1. Objectives
- Establish a documented workflow for introducing a long-duration/seasonal storage technology into the existing NYISO ACORN study.
- Enumerate the repository artifacts that must be cloned or edited when branching scenarios off the current baseline.
- Define the core result files and derived metrics to compare when evaluating the effect of seasonal storage relative to short-duration batteries under multiple stress tests.

## 2. Baseline snapshot to preserve before editing
| Component | Purpose | Where to copy from |
| --- | --- | --- |
| Load-model notebooks | Source for the demand traces that feed all scenarios. | `01_baseline.ipynb`, `02_resstock.ipynb`, `03_comstock.ipynb` at repo root and the mirrors under `scripts/02_load_modeling/` |
| Baseline scenario inputs | Re-create GenX→ACORN coupling without storage edits. | Entire `runs/mod_RE_mod_elec_iter0/` directory (or the scenario you plan to fork) |
| GenX capacity outputs | Reference for storage/generator capacities before modification. | `data/genX/Combined_Capacity_S1_Mod_RE_Mod_Elec.xlsx` and the downscaled `Selected_CPA_Capacity_Mod_RE_Mod_Elec.csv` |
| Generator & storage properties | Check technology codes, default cost/duration parameters. | `data/genX/genX_generators.csv` |
| Grid metadata | Ensure storage siting logic stays aligned with bus assignments. | `data/grid/*.csv` and `runs/.../construct_inputs.ipynb` stored outputs |
| Historical load aggregates | Maintain the same processed load that baseline uses. | `data/nyiso/historical_load/combined/` |

## 3. Implementing a dedicated seasonal storage scenario
1. **Clone the run directory.**
   - Copy `runs/mod_RE_mod_elec_iter0/` to a new folder such as `runs/mod_RE_mod_elec_iter0_seasonal/`.
   - Update `config.yml` inside the copy: adjust `run_name`, `genX_file_name`, and `genX_downscaled_file_name` to point to the seasonal-storage GenX exports you will create.
   - Rename the SLURM launcher (`run_acorn.sh`) job name and log folder to avoid overwriting baseline logs.

2. **Create GenX outputs that include the seasonal device.**
   - Duplicate `Combined_Capacity_S1_Mod_RE_Mod_Elec.xlsx` and `Selected_CPA_Capacity_Mod_RE_Mod_Elec.csv`; append an identifying suffix (e.g., `_seasonal_storage`).
   - Within the workbook/CSV:
     - Add a new storage entry whose `technology` matches a seasonal device (for example, a hydrogen cavern or thermal storage) with a high energy-to-power ratio (≥100 h) by setting `Existing_Cap_MWh`, `Existing_Cap_MW`, and `Existing_Charge_Cap_MW` accordingly.
     - Verify `STOR` is flagged as `1`, `New_Build` is `0` if you want the capacity fixed, and populate `Eff_Up`/`Eff_Down`, `Self_Disch`, and `Var_OM_Cost_per_MWh` with technology-appropriate values.
     - If the seasonal resource is zonal rather than site-based, assign the `Zone` column to the NYISO zone that should host the asset; the coupling notebook will distribute buses within that zone.
   - Update `data/genX/genX_generators.csv` if the new technology code is not already defined (e.g., add a new `Resource` row with consistent cost/duration parameters).

3. **Adjust the input-construction notebook.**
   - Open the copied `construct_inputs.ipynb` in the new scenario folder.
   - In the storage-mapping cells, increase the "sites per zone" parameter only for the seasonal technology if you want a single large caverns at specific buses; otherwise keep one site per zone to preserve duration.
   - Re-run the notebook to regenerate:
     - `storage_resources.csv`, `storage_properties.csv`, and siting figures under `figs/` that confirm the seasonal asset locations.
     - `NG_matching.txt` to ensure thermal balancing is still acceptable (seasonal storage should not alter thermal matching, but rerunning validates there were no side effects).

4. **Document model assumptions.**
   - Inside the new scenario directory, add a short `notes.md` (or append to an existing README cell in the notebook) summarizing the chosen round-trip efficiency, power rating, charging limits, and any limitations (e.g., restricted to charge in Oct–Apr) that you encoded downstream.

5. **Launch ACORN runs.**
   - Execute the updated `run_acorn.sh` so that both the `nyiso_only` and `external_zones` variants are produced under `runs/<scenario>/outputs/<climate_scenario_years>/<save_name>/`.

## 4. Scenario matrix for comparative analysis
| Scenario | Description | Key file edits | Stress focus |
| --- | --- | --- | --- |
| **Baseline** | Original `mod_RE_mod_elec_iter0` without new storage. | None (just rerun for reference if needed). | Serves as the anchor for all metrics. |
| **Enhanced batteries** | Scale up the existing 4–8 h battery fleet before introducing seasonal storage. | Modify storage rows in `Combined_Capacity_*.xlsx` to increase `Existing_Cap_MW/MWh`; rerun `construct_inputs.ipynb`. | Tests diminishing returns of short-duration resources on VOLL. |
| **Seasonal storage** | Introduce the high-duration device created in §3. | Files listed in §3 plus updated `config.yml`. | Measures winter→summer shifting and reliability gains. |
| **Hybrid stack** | Combine enhanced batteries with the seasonal resource. | Apply both sets of edits to the GenX exports before coupling. | Evaluates complementarity between daily and seasonal storage. |
| **Summer peak stress** | Boost July–Aug load by +X % to mimic electrification surges. | In scenario notebook, adjust the load-scaling cell prior to saving `load_future.csv`. | Observes whether seasonal storage still contributes when stress is summer-peaked. |
| **Winter supply stress** | Derate gas or wind units to mimic extreme cold events. | Edit `NG_matching.txt` target totals or retire select gas units in the notebook; optionally scale wind timeseries downward. | Quantifies avoided VOLL when scarcity shifts to winter. |
| **Renewable drought** | Apply a multi-week solar/wind derate based on historical low-output periods. | In the notebook, multiply the renewable profiles for chosen weeks by a factor (e.g., 0.5) before exporting. | Tests how seasonal storage covers prolonged renewable deficits. |

Plan to run the scenarios sequentially: start with Baseline → Enhanced batteries → Seasonal storage → Hybrid, then apply each stress test to both the Baseline and Seasonal cases to isolate benefits under adverse conditions.

## 5. Results to extract and compare after each run
1. **Reliability and VOLL proxies**
   - `load_shedding_<year>.csv` (NYISO-only and external-zones variants) — sum by month/season to compute winter vs. summer unmet load.
   - Multiply seasonal load shedding (MWh) by ACORN's implied VOLL (10,000 $/MWh, per the objective coefficient in `src/julia/acorn.jl`) to estimate avoided outage cost when comparing scenarios.

2. **Storage behavior**
   - `charge_<year>.csv`, `discharge_<year>.csv`, and `batt_state_<year>.csv` — inspect seasonal SOC swings, peak charging weeks, and duration of sustained discharge.
   - For the hybrid case, separate short-duration battery buses from the seasonal asset buses to observe complementary dispatch.

3. **Generation and transmission interactions**
   - `gen_<year>.csv` — identify which thermal units backfill reduced load shedding once seasonal storage is added.
   - `flow_<year>.csv` — confirm whether interzonal transfers shift with the new storage siting.

4. **Diagnostic figures**
   - Regenerated plots in `runs/<scenario>/figs/` (capacity comparisons, storage siting maps, load duration curves) — use side-by-side with baseline figures to check that only intended inputs changed.

For each metric, build a comparison notebook (e.g., `analysis/seasonal_storage_postproc.ipynb`) that ingests the CSVs for Baseline, Enhanced battery, and Seasonal cases and outputs:
- Seasonal totals of load shedding (MWh) and avoided VOLL (USD).
- Frequency, duration, and timing of load-shedding events.
- Contribution of seasonal storage to winter reliability vs. summer.
- Sensitivity plots across the defined stress tests.

## 6. Scenario planning checklist
- [ ] Snapshot baseline artifacts listed in §2 before editing.
- [ ] Record every new GenX file name in the scenario `config.yml` so reruns stay reproducible.
- [ ] After each notebook rerun, commit the regenerated CSVs/figures together with `NG_matching.txt` so thermal mismatches remain auditable.
- [ ] Store post-processing notebooks/results under `analysis/` with descriptive filenames and include README cells that document VOLL assumptions and stress-test parameters.
- [ ] Maintain a changelog (either in the repo root or per-scenario `notes.md`) summarizing parameter tweaks, especially the seasonal storage efficiency, min/max state-of-charge constraints, and any manual profile edits used for stress testing.

Following this plan keeps the baseline intact, creates a traceable seasonal storage implementation, and ensures that each scenario run produces the comparison-ready metrics needed to quantify avoided outage costs from summer through winter.
