#!/usr/bin/env bash

# Exit immediately if a command exits with a non-zero status
set -e

# Set the PYTHONPATH to the current working directory
export PYTHONPATH=$(pwd)

# Run the Python script
python3 app/run.py
