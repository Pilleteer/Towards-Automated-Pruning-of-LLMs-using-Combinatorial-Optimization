import os
import json
import argparse
import subprocess
from utils import generate_config, delete_tensor_files, get_available_layers

def prune_model(model_name, prune_layers, output_path, max_layer):
    """Generates config and prunes the model."""
    config = generate_config(model_name, prune_layers, max_layer)
    
    with open("./configs/prune_layer_config.yaml", "w") as f:
        json.dump(config, f)

    delete_tensor_files(output_path)

    cmd = ["mergekit-yaml", "./configs/prune_layer_config.yaml", output_path, "--cuda"]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    print(result.stdout.decode(), result.stderr.decode())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="LLAMA3")
    parser.add_argument("--prune_layers", type=str, default="0,1,2,7,3,4,6,8,9,10")
    parser.add_argument("--output_path", type=str, default="./LLAMA3/prune_llm")

    args = parser.parse_args()
    prune_layers = [int(i) for i in args.prune_layers.split(",")]

    prune_model(args.model_name, prune_layers, args.output_path)
