import torch
import json
import subprocess
import re
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
# from accelerate import Accelerator
import numpy as np

print("PyTorch version:", torch.__version__)
print("CUDA version:", torch.version.cuda)

# accelerator = Accelerator()
# device = accelerator.device

model_name = "LLAMA3"
original_model_size = 15316.60 #Mb
original_model_acc = 0.7918

pruned_path = "./LLAMA3/prune_llm"

def count_nonzero_params(model):
    return sum(p.nonzero().size(0) for p in model.parameters())

def get_acc(result):
    accuracy_pattern = r'\|hellaswag\s*\|\s*\d+\|[^\|]*\|[^\|]*\|acc\s*\|[^|]*\|\s*([\d.]+)\|'
    accuracy_norm_pattern = r'\|\s*\|\s*\\|[^\|]*\|[^\|]*\|acc_norm\s*\|[^|]*\|\s*([\d.]+)\|'

    accuracy_match = re.search(accuracy_pattern, result.stdout)
    accuracy_norm_match = re.search(accuracy_norm_pattern, result.stdout)

    if accuracy_match:
        accuracy = float(accuracy_match.group(1).strip())
        
        print(f"Accuracy: {accuracy}")
    else:
        print("Accuracy not found in the output.")

    if accuracy_norm_match:
        accuracy_norm = float(accuracy_norm_match.group(1).strip())
        print(f"Normalized Accuracy: {accuracy_norm}")
        return accuracy_norm
    else:
        if result.stderr:
            print(result.stderr)
        print("Normalized Accuracy not found in the output.")
    


def get_model_size(model):
    return sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024 / 2

def get_path_size(model_path):
    # Initialize the total model size
    model_size = 0

    # List all files in the specified directory
    files = os.listdir(model_path)
    
    # Filter out the shard files based on naming pattern
    shard_files = [file for file in files if file.startswith('model-') and file.endswith('.safetensors')]
    
    # Sum the size of each shard file
    for shard_file in shard_files:
        model_size += os.path.getsize(os.path.join(model_path, shard_file))
    print(f"Model size: {model_size / (1024 * 1024):.2f} MB")
    return model_size / (1024 * 1024)

def src_config(begin, dest, model_name):
    return {
        "sources": [
            {
                "layer_range": [begin, dest],
                "model": {
                    "model": {
                        "path": model_name
                    }
                }
            }
        ]
    }

def generate_config(model_name, x):
    # x as array of layer that want to be sliced from 1 to 32
    config = {
        "dtype": "bfloat16",
        "merge_method": "passthrough",
        "slices": [
        ]
    }
    x = sorted(x)
    begin = 0
    dest = 32
    if len(x) == 0:
        config["slices"].append(src_config(begin, dest, model_name))
    else:
        for i in x:
            if i == begin:
                begin = i+1
                continue
            dest = i-1
            config["slices"].append(src_config(begin, dest, model_name))
            begin = i+1
        dest = 32
        config["slices"].append(src_config(begin, dest, model_name))
    return config

def delete_tensor_files(model_path):
    files = os.listdir(model_path)
    
    # Filter out the shard files based on naming pattern
    shard_files = [file for file in files if file.startswith('model-') and file.endswith('.safetensors')]
    
    # Sum the size of each shard file and delete the file
    for shard_file in shard_files:
        file_path = os.path.join(model_path, shard_file)
        
        try:
            os.remove(file_path)  # Attempt to delete the file
            print(f"Successfully deleted {shard_file}")
        except Exception as e:
            print(f"Failed to delete {shard_file}: {e}")

def objective_func(x):
    print("x: ", x)
    x = [int(i) for i in x]
    print("============================================")
    config = generate_config(model_name, x)
    with open("prune_layer_config.yaml", "w") as f:
        json.dump(config, f)
    # Command: mergekit-yaml ./yaml_config/prune_layer_config.yaml ./LLAMA3/prune_llm --cuda
    result = subprocess.run(["mergekit-yaml", "./prune_layer_config.yaml", f"./{model_name}/prune_llm", "--cuda"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
#     tokenizer = AutoTokenizer.from_pretrained(f"./{model_name}")
#     tokenizer.pad_token = tokenizer.eos_token
#     pruned_model = AutoModelForCausalLM.from_pretrained(f"./{model_name}/prune_llm")#.to(device)
    pruned_model_size = get_path_size(pruned_path)
#     print(f"Model size: {pruned_model_size} Mb")
    command = [
        'accelerate', 'launch', '-m',
        'lm_eval', 
        '--model', 'hf',
        '--model_args', 'pretrained=LLAMA3/prune_llm',
        '--tasks', 'hellaswag',
        '--device', 'cuda:0',
        '--batch_size', '8'
    ]

    result = subprocess.run(command, capture_output=True, text=True)
#     print(result.stdout)
    pruned_model_acc = get_acc(result)
    
    size_diff = (original_model_size - pruned_model_size) / original_model_size
    print(size_diff)
    delete_tensor_files(pruned_path)
    return [size_diff, pruned_model_acc]
