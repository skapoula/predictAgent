#!/usr/bin/env bash
# Download the VIAVI dataset from the O-RAN SC Nexus repository.
# Run from the repo root: bash scripts/download_data.sh

set -euo pipefail

DEST="viavi-dataset/raw"
BASE_URL="https://nexus3.o-ran-sc.org/repository/datasets"

mkdir -p "$DEST"

for FILE in CellReports.csv UEReports-flow.csv; do
    TARGET="$DEST/$FILE"
    if [ -f "$TARGET" ]; then
        echo "Already exists: $TARGET"
    else
        echo "Downloading $FILE ..."
        curl -fL --progress-bar "$BASE_URL/$FILE" -o "$TARGET"
        echo "Saved to $TARGET"
    fi
done

echo "Done."
