#!/bin/bash

# Test script for stimulus CLI applications

set -e  # Exit immediately if a command exits with a non-zero status

# Set up paths
TEST_DIR=$(pwd)/tests
DATA_LOC=$TEST_DIR/test_data/titanic/titanic_stimulus.csv
JSON_LOC=$TEST_DIR/test_output/titanic_stimulus-split-RandomSplitter_0.7_0.15_0.15.json
OUTPUT_LOC=$TEST_DIR/test_output/titanic_stimulus_split.csv

# Run the command
stimulus-split-csv -c $DATA_LOC -j $JSON_LOC -o $OUTPUT_LOC

echo "stimulus-split-csv test passed"