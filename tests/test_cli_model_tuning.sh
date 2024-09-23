#!/bin/bash

# Test script for stimulus CLI applications

set -e  # Exit immediately if a command exits with a non-zero status

# Set up paths
TEST_DIR=$(pwd)/tests
DATA_LOC=$TEST_DIR/test_output/titanic_stimulus_split.csv
JSON_LOC=$TEST_DIR/test_output/titanic_stimulus-split-RandomSplitter_0.7_0.15_0.15.json
MODEL_LOC=$TEST_DIR/test_model/titanic_model.py
CONFIG_LOC=$TEST_DIR/test_model/titanic_model_cpu.yaml
OUTPUT_LOC=$TEST_DIR/

# Run the command
stimulus-tuning -c $CONFIG_LOC -m $MODEL_LOC -d $DATA_LOC -e $JSON_LOC -o $OUTPUT_LOC --gpus 0

echo "stimulus-tuning test passed"

