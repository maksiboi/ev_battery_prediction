#!/usr/bin/env bash

# Exit immediately if a command exits with a non-zero status
set -e

# Set the PYTHONPATH to the current working directory
export PYTHONPATH=$(pwd)

# Check if a file argument was provided
if [ -z "$1" ]; then
  echo "Usage: $0 <python_file_to_run>"
  exit 1
fi

# Run the Python script passed as the argument
python3 "$1"

#Usage: start.sh <python_file_to_run>