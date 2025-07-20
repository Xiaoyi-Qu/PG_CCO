#!/bin/bash
OUTPUT_DIR="./outputs"
CSV_FILE="./preprocessing/multipliers.csv"

# Skip the header and read the first column of each line (each line is separated by column)
tail -n +2 "$CSV_FILE" | while IFS=',' read -r problem _; do

    problem_clean=$(echo "$problem" | tr -d '\r\n')

    /home/xiq322/miniconda3/bin/python cutest_modified.py --name $problem_clean

    echo "----------------------------------------"
done