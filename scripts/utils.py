import os
import json

def get_path_size(model_path):
    total_size = sum(os.path.getsize(os.path.join(model_path, f)) for f in os.listdir(model_path) if f.startswith("model-"))
    return total_size / (1024 * 1024)

def generate_config(model_name, prune_layers):
    config = {"dtype": "bfloat16", "merge_method": "passthrough", "slices": []}
    prune_layers = sorted(prune_layers)
    
    begin = 0
    for i in prune_layers:
        config["slices"].append({"sources": [{"layer_range": [begin, i-1], "model": {"path": model_name}}]})
        begin = i + 1
    config["slices"].append({"sources": [{"layer_range": [begin, 32], "model": {"path": model_name}}]})
    
    return config

def delete_tensor_files(model_path):
    for f in os.listdir(model_path):
        if f.startswith("model-") and f.endswith(".safetensors"):
            os.remove(os.path.join(model_path, f))
