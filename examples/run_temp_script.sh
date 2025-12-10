#!/bin/bash
set -e

# Create output directory
mkdir -p output

echo "Splitting dataset..."
uv run stimulus split \
  --data ../tests/test_data/vcc_subset/vcc_training_subset.h5ad \
  --yaml split.yaml \
  --output output/vcc_split.h5ad

echo "Transforming (2000 HVG)..."
uv run stimulus transform \
  --data output/vcc_split.h5ad \
  --yaml transform_2000.yaml \
  --output output/vcc_2000.h5ad

echo "Transforming (500 HVG)..."
uv run stimulus transform \
  --data output/vcc_split.h5ad \
  --yaml transform_500.yaml \
  --output output/vcc_500.h5ad

echo "Done!"
