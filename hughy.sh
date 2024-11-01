#!/bin/bash

# Check if a word argument was provided
if [ -z "$1" ]; then
    echo "Please provide a word to pass as an argument to information_theory.py"
    exit 1
fi

# Absolute path to information_theory.py
SCRIPT_PATH="/c/Users/dalli/source/acme_senior/vl3labs/InformationTheory/information_theory.py"

# Run the Python script with the word argument
python "$SCRIPT_PATH" "$1"
