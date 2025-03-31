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

# Load Pruning Config
PRUNING_METHOD=$(yq e '.pruning.method' $CONFIG_FILE)
SELECTED_LAYERS=$(yq e '.pruning.selected_layers | join(",")' $CONFIG_FILE)

# Load Optimization Config
OPT_ALGO=$(yq e '.optimization.algorithm' $CONFIG_FILE)
NUM_GENERATIONS=$(yq e '.optimization.num_generations' $CONFIG_FILE)
POP_SIZE=$(yq e '.optimization.population_size' $CONFIG_FILE)

# Load Training Config
NUM_EPOCHS=$(yq e '.training.num_train_epochs' $CONFIG_FILE)
LR=$(yq e '.training.learning_rate' $CONFIG_FILE)
BATCH_SIZE=$(yq e '.training.per_device_train_batch_size' $CONFIG_FILE)
GRAD_ACC_STEPS=$(yq e '.training.gradient_accumulation_steps' $CONFIG_FILE)
SEQ_LENGTH=$(yq e '.training.max_seq_length' $CONFIG_FILE)
OUTPUT_DIR=$(yq e '.training.output_dir' $CONFIG_FILE)
SAVE_STEPS=$(yq e '.training.save_steps' $CONFIG_FILE)

# Load Evaluation Config
EVAL_TASKS=$(yq e '.evaluation.tasks' $CONFIG_FILE)
EVAL_DEVICE=$(yq e '.evaluation.device' $CONFIG_FILE)
EVAL_BATCH_SIZE=$(yq e '.evaluation.batch_size' $CONFIG_FILE)

echo "🔹 Running optimization using $OPT_ALGO..."
python scripts/optimize.py --algorithm "$OPT_ALGO" --generations "$NUM_GENERATIONS" --population "$POP_SIZE"

echo "🔹 Pruning model using $PRUNING_METHOD method..."
python scripts/prune_model.py --model "$MODEL_NAME" --path "$MODEL_PATH" --method "$PRUNING_METHOD" --layers "$SELECTED_LAYERS"

echo "🔹 Retraining the pruned model..."
python retrain.py --model_name "$MODEL_NAME" \
                  --num_train_epochs "$NUM_EPOCHS" \
                  --learning_rate "$LR" \
                  --per_device_train_batch_size "$BATCH_SIZE" \
                  --gradient_accumulation_steps "$GRAD_ACC_STEPS" \
                  --max_seq_length "$SEQ_LENGTH" \
                  --output_dir "$OUTPUT_DIR" \
                  --save_steps "$SAVE_STEPS" \
                  --prune_layers "$SELECTED_LAYERS" \
                  --device "$EVAL_DEVICE" \
                  --batch_size "$EVAL_BATCH_SIZE" \
                  --tasks "$EVAL_TASKS"
                  "

echo "🔹 Evaluating the retrained model..."
python scripts/evaluate.py --model_path "$OUTPUT_DIR/final_merged_checkpoint" \
                           --model_name "$MODEL_NAME" \
                           --tasks "$EVAL_TASKS" \
                           --device "$EVAL_DEVICE" \
                           --batch_size "$EVAL_BATCH_SIZE"

echo "Process complete!"
