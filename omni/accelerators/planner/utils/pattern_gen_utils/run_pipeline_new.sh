#!/bin/bash

# run_pipeline_new.sh
# Shell script to run pipeline_new.py with specified or default parameters

# Default parameters
INPUT_FOLDER="./activation_datas/24bs+6kinput+4host/decode"
OUTPUT_CSV="topk_ids_count_longbench_6k_decode_with_ceiling.csv"
NUM_LAYERS=58
NUM_RANKS=64
RANK_ID_RANGE=""
NUM_DEVICES=128
NUM_REDUNDANT_LAYERS="0 10 20 30 58"
EXPERT_REDUNDANT_LIMIT=11
NUM_MOE_LAYERS=58
NUM_EPS=256
DATASET_NAME="longbench_6k_decode_redundant_0513"
OUTPUT_FILE_PREFIX="DSV3_0513_longbench_6k_decode_redundant"

# Function to display usage
usage() {
    echo "Usage: $0 [--input_folder <input_folder>] [--output_csv <output_csv>] [--num_layers <num_layers>]"
    echo "          [--num_ranks <num_ranks>] [--rank_id_range <min_rank_id> <max_rank_id>]"
    echo "          [--num_devices <num_devices>] [--num_redundant_layers <num_redundant_layers>]"
    echo "          [--expert_redundant_limit <limit>] [--num_moe_layers <num_moe_layers>]"
    echo "          [--num_eps <num_eps>] [--dataset_name <dataset_name>] [--output_file_prefix <prefix>]"
    echo "Note: input_folder must be a subdirectory under ./activation_datas"
    echo "Example: $0 --input_folder ./activation_datas/my_data --num_ranks 32 --output_csv count.csv --num_redundant_layers \"5 15 25\""
    exit 1
}

# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --input_folder) INPUT_FOLDER="$2"; shift ;;
        --output_csv) OUTPUT_CSV="$2"; shift ;;
        --num_layers) NUM_LAYERS="$2"; shift ;;
        --num_ranks) NUM_RANKS="$2"; shift ;;
        --rank_id_range) RANK_ID_RANGE="$2 $3"; shift 2 ;;
        --num_devices) NUM_DEVICES="$2"; shift ;;
        --num_redundant_layers) NUM_REDUNDANT_LAYERS="$2"; shift ;;
        --expert_redundant_limit) EXPERT_REDUNDANT_LIMIT="$2"; shift ;;
        --num_moe_layers) NUM_MOE_LAYERS="$2"; shift ;;
        --num_eps) NUM_EPS="$2"; shift ;;
        --dataset_name) DATASET_NAME="$2"; shift ;;
        --output_file_prefix) OUTPUT_FILE_PREFIX="$2"; shift ;;
        -h|--help) usage ;;
        *) echo "Unknown parameter: $1"; usage ;;
    esac
    shift
done

# Ensure Python is available
if ! command -v python &> /dev/null; then
    echo "Error: python is not installed or not found in PATH."
    exit 1
fi

# Ensure the input folder exists and is under ./activation_datas
if [[ ! -d "$INPUT_FOLDER" ]]; then
    echo "Error: Input folder $INPUT_FOLDER does not exist."
    exit 1
fi
if [[ ! "$INPUT_FOLDER" =~ ^./activation_datas/ ]]; then
    echo "Error: Input folder $INPUT_FOLDER must be a subdirectory under ./activation_datas."
    exit 1
fi

# Ensure num_ranks is provided
if [[ -z "$NUM_RANKS" ]]; then
    echo "Error: --num_ranks is required."
    usage
fi

# Ensure required Python scripts exist
for script in pipeline_new.py generate_csv.py step_2_placement_pattern_generation.py step_3_placement_pattern_checking_and_plot.py step_4_load_analysis_and_plot.py; do
    if [[ ! -f "$script" ]]; then
        echo "Error: Required script $script is missing."
        exit 1
    fi
done

# Run the pipeline
echo "Running pipeline_new.py with the following parameters:"
echo "Input Folder: $INPUT_FOLDER"
echo "Output CSV: $OUTPUT_CSV"
echo "Number of Layers: $NUM_LAYERS"
echo "Number of Ranks: $NUM_RANKS"
echo "Rank ID Range: ${RANK_ID_RANGE:-"0 $((NUM_RANKS-1))"}"
echo "Number of Devices: $NUM_DEVICES"
echo "Number of Redundant Layers: $NUM_REDUNDANT_LAYERS"
echo "Expert Redundant Limit: $EXPERT_REDUNDANT_LIMIT"
echo "Number of MoE Layers: $NUM_MOE_LAYERS"
echo "Number of Experts: $NUM_EPS"
echo "Dataset Name: $DATASET_NAME"
echo "Output File Prefix: $OUTPUT_FILE_PREFIX"

python pipeline_new.py \
    --input_folder "$INPUT_FOLDER" \
    --output_csv "$OUTPUT_CSV" \
    --num_layers "$NUM_LAYERS" \
    --num_ranks "$NUM_RANKS" \
    ${RANK_ID_RANGE:+--rank_id_range $RANK_ID_RANGE} \
    --num_devices "$NUM_DEVICES" \
    --num_redundant_layers $NUM_REDUNDANT_LAYERS \
    --expert_redundant_limit "$EXPERT_REDUNDANT_LIMIT" \
    --num_moe_layers "$NUM_MOE_LAYERS" \
    --num_eps "$NUM_EPS" \
    --dataset_name "$DATASET_NAME" \
    --output_file_prefix "$OUTPUT_FILE_PREFIX"

if [[ $? -eq 0 ]]; then
    echo "Pipeline completed successfully."
else
    echo "Pipeline failed. Check the output for errors."
    exit 1
fi