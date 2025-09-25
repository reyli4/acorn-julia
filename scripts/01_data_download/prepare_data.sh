#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PROJECT="$(cd "$SCRIPT_DIR/../.." && pwd)"
echo "Project root: $PROJECT"

# Where downloads will be stored
mkdir -p \
  "$PROJECT/data/nyiso" \
  "$PROJECT/data/nrel/sind" \
  "$PROJECT/data/nrel/wtk" \
  "$PROJECT/data/nrel/resstock" \
  "$PROJECT/data/nrel/comstock" \
  "$PROJECT/data/nys_gis"

# Make these visible to sub-scripts (use them if your scripts support it)
export PROJECT
export OUTDIR="$PROJECT/data"
# If you have an NREL API key, export it before running this script:
# export NREL_API_KEY=...

run() {
  local path="$1"; shift || true
  if [[ -x "$path" ]]; then
    echo "▶ $path $*"
    "$path" "$@"
  elif [[ -f "$path" ]]; then
    echo "▶ bash $path $*"
    bash "$path" "$@"
  else
    echo "⤵︎ skip (not found): $path"
  fi
}

# 01) NYISO load (Python) — pass an output dir if your script supports --out
if command -v python >/dev/null 2>&1; then
  echo "▶ python $SCRIPT_DIR/01_nyiso_load.py --out $PROJECT/data/nyiso"
  python "$SCRIPT_DIR/01_nyiso_load.py" --out "$PROJECT/data/nyiso" || echo "NYISO loader returned non-zero."
else
  echo "python not found; skipping 01_nyiso_load.py"
fi

# 02–07) Shell scripts
run "$SCRIPT_DIR/02_nrel_sind.sh"
run "$SCRIPT_DIR/03_nrel_wtk.sh"
# 04 is a markdown file; skip.
run "$SCRIPT_DIR/05_nrel_resstock.sh"
run "$SCRIPT_DIR/06_nrel_comstock.sh"
run "$SCRIPT_DIR/07_nys_gis.sh"

echo
echo "✔ Data prep finished. Current data directories:"
du -sh "$PROJECT/data"/* 2>/dev/null || true
