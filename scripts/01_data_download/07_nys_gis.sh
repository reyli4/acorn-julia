#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PROJECT="$(cd "$SCRIPT_DIR/../.." && pwd)"
OUT="$PROJECT/data/nys_gis"
mkdir -p "$OUT"

URL="https://www2.census.gov/geo/tiger/GENZ2018/shp/cb_2018_us_county_5m.zip"

TMP="$(mktemp)"
echo "Downloading $URL..."
curl -fsSL "$URL" -o "$TMP"

echo "Unzipping into $OUT..."
unzip -o "$TMP" -d "$OUT" >/dev/null

rm -f "$TMP"
echo "✔ Done. Files in $OUT:"
ls -lah "$OUT"
