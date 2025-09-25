import os, json, glob, pandas as pd

def summarize_run(run_root):
    # Expected ACORN outputs dir: runs/<run_name>/outputs/<climate>/<save_name>/
    for outdir in glob.glob(os.path.join(run_root, "outputs", "*", "*")):
        if not os.path.isdir(outdir): 
            continue
        manifest = {
            "run_dir": run_root,
            "outputs_dir": outdir,
            "files": sorted([os.path.basename(p) for p in glob.glob(os.path.join(outdir, "*"))]),
        }
        # Save a small manifest
        man_path = os.path.join(outdir, "run_manifest.json")
        with open(man_path, "w") as f:
            json.dump(manifest, f, indent=2)

        # Try to read a couple of standard CSVs if they exist
        summary = {}
        ls_paths = sorted(glob.glob(os.path.join(outdir, "load_shedding_*.csv")))
        cur_paths = sorted(glob.glob(os.path.join(outdir, "curtailment_*.csv")))

        def summarize_series(csvs, value_col="value_MW", name=""):
            rows = []
            for p in csvs:
                try:
                    df = pd.read_csv(p)
                    # guess columns
                    cols = {c.lower(): c for c in df.columns}
                    val = cols.get(value_col.lower())
                    time = cols.get("time") or cols.get("datetime")
                    if val is None or time is None:
                        continue
                    df[time] = pd.to_datetime(df[time])
                    rows.append({
                        "file": os.path.basename(p),
                        "sum_MWh": df[val].sum(),   # assuming hourly
                        "max_MW": df[val].max(),
                        "hours_gt0": (df[val] > 0).sum(),
                        "first_ts": df[time].min().isoformat(),
                        "last_ts": df[time].max().isoformat(),
                    })
                except Exception:
                    pass
            if rows:
                out = pd.DataFrame(rows)
                out.to_csv(os.path.join(outdir, f"summary_{name}.csv"), index=False)

        summarize_series(ls_paths, value_col="load_shed_MW", name="load_shedding")
        summarize_series(cur_paths, value_col="curtailment_MW", name="curtailment")

        # You can add more tiny summaries here (prices, flows, emissions) following the same pattern.

if __name__ == "__main__":
    repo = os.path.expanduser("~/acorn-julia")
    for run_path in glob.glob(os.path.join(repo, "runs", "*")):
        if os.path.isdir(run_path):
            summarize_run(run_path)
