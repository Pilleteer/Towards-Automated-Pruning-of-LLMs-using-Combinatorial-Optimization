import os
import re
import time
import subprocess
import argparse

def get_acc(result):
    """Extracts accuracy from evaluation output."""
    accuracy_pattern = r'\|hellaswag\s*\|\s*\d+\|[^\|]*\|[^\|]*\|acc\s*\|[^|]*\|\s*([\d.]+)\|'

    accuracy_match = re.search(accuracy_pattern, result.stdout)
    return float(accuracy_match.group(1).strip()) if accuracy_match else None

def evaluate_model(model_path, model_name, tasks, device, batch_size):
    """Evaluates the pruned model accuracy with configurable parameters."""
    command = [
        'accelerate', 'launch', '-m',
        'lm_eval', '--model', model_name,
        '--model_args', f'pretrained={model_path}',
        '--tasks', tasks,
        '--device', device,
        '--batch_size', str(batch_size)
    ]

    start_time = time.time()
    result = subprocess.run(command, capture_output=True, text=True)
    taken_time = time.time() - start_time

    acc = get_acc(result)
    print(f"Evaluation Time: {taken_time:.2f}s, Accuracy: {acc}")
    return acc

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a pruned model with lm_eval.")
    parser.add_argument("--model_path", type=str, default="LLAMA3", help="Path to the pruned model")
    parser.add_argument("--model_name", type=str, default="hf", help="Model type (e.g., 'hf', etc.)")
    parser.add_argument("--tasks", type=str, default="hellaswag", help="Evaluation tasks")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to run evaluation (e.g., 'cuda:0')")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for evaluation")

    args = parser.parse_args()
    
    evaluate_model(args.model_path, args.model_name, args.tasks, args.device, args.batch_size)
