import torch
import json
import subprocess
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import Accelerator
import numpy as np
import time

def get_acc(result):
    accuracy_pattern = r'\|hellaswag\|[^\|]*\|[^\|]*\|[^\|]*\|acc\s*\|[^\|]*\|([0-9.]+)\|'
    accuracy_norm_pattern = r'\|\s*\|[^\|]*\|[^\|]*\|[^\|]*\|acc_norm\s*\|[^\|]*\|([0-9.]+)\|'

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
    
    return accuracy_norm

command = [
        'accelerate', 'launch', 
        '-m', 'lm_eval', 
        '--model', 'hf',
        '--model_args', 'pretrained=LLAMA3',
        '--tasks', 'hellaswag',
        '--device', 'cuda:0',
        '--batch_size', '2'
    ]

# Track time
start = time.time()
result = subprocess.run(command, capture_output=True, text=True)
end = time.time()
print(f"Execution time: {end - start}")
pruned_model_acc = get_acc(result)


