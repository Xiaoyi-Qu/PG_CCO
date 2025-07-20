#!/bin/bash
OUTPUT_DIR="./outputs"
CSV_FILE="cutest_filter.csv"

# Skip the header and read each line
tail -n +2 "$CSV_FILE" | while IFS=',' read -r problem; do

    problem_clean=$(echo "$problem" | tr -d '\r\n')

    /home/xiq322/miniconda3/bin/python multiplier_preprocessing.py --name $problem_clean

    echo "----------------------------------------"
done