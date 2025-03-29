#!/bin/bash

CONFIG_FILE="config.yaml"

# Ensure yq is installed
if ! command -v yq &> /dev/null; then
    echo "Error: 'yq' is not installed. Install it with 'pip install yq'."
    exit 1
fi

echo "🔹 Loading configuration from $CONFIG_FILE"

# Load Model Config
MODEL_NAME=$(yq e '.model.name' $CONFIG_FILE)
MODEL_PATH=$(yq e '.model.path' $CONFIG_FILE)
MAX_LAYERS=$(yq e '.model.max_layers' $CONFIG_FILE)

# Load Pruning Config
PRUNING_METHOD=$(yq e '.pruning.method' $CONFIG_FILE)
SELECTED_LAYERS=$(yq e '.pruning.selected_layers | join(",")' $CONFIG_FILE)

# Load Optimization Config
OPT_ALGO=$(yq e '.optimization.algorithm' $CONFIG_FILE)
NUM_GENERATIONS=$(yq e '.optimization.num_generations' $CONFIG_FILE)
POP_SIZE=$(yq e '.optimization.population_size' $CONFIG_FILE)

# Load Evaluation Config
EVAL_MODEL=$(yq e '.evaluation.model_name' $CONFIG_FILE)
EVAL_TASKS=$(yq e '.evaluation.tasks' $CONFIG_FILE)
EVAL_DEVICE=$(yq e '.evaluation.device' $CONFIG_FILE)
EVAL_BATCH_SIZE=$(yq e '.evaluation.batch_size' $CONFIG_FILE)

echo "🔹 Pruning model using $PRUNING_METHOD method..."
python scripts/prune_model.py --model "$MODEL_NAME" --path "$MODEL_PATH" --method "$PRUNING_METHOD" --layers "$SELECTED_LAYERS"

echo "🔹 Running optimization using $OPT_ALGO..."
python scripts/optimize.py --algorithm "$OPT_ALGO" --generations "$NUM_GENERATIONS" --population "$POP_SIZE"

echo "🔹 Evaluating the pruned model..."
python scripts/evaluate.py --model_path "$MODEL_PATH" \
                           --model_name "$EVAL_MODEL" \
                           --tasks "$EVAL_TASKS" \
                           --device "$EVAL_DEVICE" \
                           --batch_size "$EVAL_BATCH_SIZE"

echo "✅ Process complete!"
