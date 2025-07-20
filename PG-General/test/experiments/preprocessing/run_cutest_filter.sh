#!/bin/bash
# OUTPUT_DIR="./outputs"

for start_index in 0 100 200 300 400 500
# for start_index in 0 10 20 30
    do
        echo "Filter CUTEst test problems starting from index $start_index."

        /home/xiq322/miniconda3/bin/python preprocessing.py --index $start_index
        #  python aml_experiments/quantize_and_evaluate_bert_squad.py \
        #         --model_path "$MODEL_PATH" \
        #         --output_dir "$OUTPUT_DIR" \
        #         --sparsity "$sparsity" > "$LOG_FILE" 2>&1

        # if [ $? -eq 0 ]; then
        #     echo "Successfully processed sparsity ${sparsity}"
        # else
        #     echo "Error processing sparsity ${sparsity}"
        # fi

        echo "----------------------------------------"
    done