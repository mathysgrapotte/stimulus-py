#!/bin/bash

# Test script for stimulus CLI applications

set -e  # Exit immediately if a command exits with a non-zero status

# Set up paths
TEST_DIR=$(pwd)/tests
DATA_LOC=$TEST_DIR/test_data/titanic/titanic_stimulus.csv
JSON_LOC=$TEST_DIR/test_data/titanic/titanic_stimulus.json
MODEL_LOC=$TEST_DIR/test_model/titanic_model.py
CONFIG_LOC=$TEST_DIR/test_model/titanic_model_cpu.yaml

# Run the command
stimulus-check-model -d $DATA_LOC -m $MODEL_LOC -e $JSON_LOC -c $CONFIG_LOC --gpus 0

echo "stimulus-check-model test passed"

