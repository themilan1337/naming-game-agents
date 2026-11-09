#!/usr/bin/env bash
# Download the authors' released name-game data (GPL-3.0 repository scaplan/name-game, commit
# pinned below) into data/cache/. The files are third-party data: they are never committed.
set -euo pipefail
COMMIT=d8acbd7e2555623cf29033fbd59ba96389461413
BASE=https://raw.githubusercontent.com/scaplan/name-game/$COMMIT
DEST="$(cd "$(dirname "$0")/.." && pwd)/data/cache/ref"
mkdir -p "$DEST/MR/input"
fetch() { [ -s "$DEST/$1" ] || curl -fsSL "$BASE/data/$1" -o "$DEST/$1"; }
fetch NameGame-2015-2018-all.tsv
fetch Empirical-CBBB2018-Flipping.tsv
for f in 16_5_8 18_6_9 20_6_10 24_7_12; do
  fetch "MR/input/$f.csv"
  fetch "MR/v4_FINAL_TP_$(echo $f | sed -E 's/^([0-9]+)_([0-9]+)_([0-9]+)$/\1round_\2_\3/')_clean.csv"
done
sha256() { if command -v sha256sum >/dev/null 2>&1; then sha256sum "$@"; else shasum -a 256 "$@"; fi; }
sha256 "$DEST"/*.tsv "$DEST"/MR/*.csv > "$DEST/SHA256SUMS"
echo "Downloaded to $DEST"
