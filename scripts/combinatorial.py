import argparse
import importlib
from mealpy import FloatVar, Problem
from mealpy.utils import io
from prune import prune_model
from evaluate import evaluate_model
from utils import get_available_layers

global model_name, tasks, device, batch_size, max_layer, prune_path

model_name = None
tasks = None
device = None
batch_size = None
max_layer = None
prune_path = "./prune_model"

def objective_func(x):
    """Objective function for optimization: Prune, evaluate, and return accuracy."""
    x = [int(i) for i in x]
    prune_model(model_name, x, prune_path, max_layer)
    acc = evaluate_model(prune_path, model_name, tasks, device, batch_size)
    return [acc]

def get_optimizer_class(optimizer_name):
    """Dynamically import any Mealpy optimizer."""
    for module in ["mealpy"]:
        try:
            mod = importlib.import_module(module)
            for attr in dir(mod):
                if optimizer_name.lower() in attr.lower():  # Match optimizer name dynamically
                    optimizer_class = getattr(mod, attr)
                    return optimizer_class
        except ImportError:
            continue
    return None

def optimize_pruning(layer_to_prune, optimizer_name, max_layer):
    """Runs the selected optimizer to find the best layer combination."""
    
    problem = {
        "obj_func": objective_func,
        "bounds": FloatVar(lb=[i for i in range(layer_to_prune)], ub=[i for i in range(max_layer - layer_to_prune, max_layer)]),
        "minmax": "max",
        "obj_weights": [1]  # Maximize accuracy
    }

    # Automatically find optimizer class
    optimizer_class = get_optimizer_class(optimizer_name)
    
    if optimizer_class is None:
        print(f"Error: {optimizer_name} is not a valid Mealpy optimizer.")
        return

    optimizer = optimizer_class(epoch=10, pop_size=50)  # Generic constructor

    best_solution, best_fitness = optimizer.solve(problem)

    print("\n=== Optimization Results ===")
    print(f"Best Layers: {best_solution}, Best Accuracy: {best_fitness}")

    io.save_model(optimizer, f"results/{optimizer_name}_model.pkl")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer_to_prune", type=int, default=5, help="Number of layers to prune")
    parser.add_argument("--optimizer", type=str, default="PSO", help="Mealpy optimization algorithm to use")
    parser.add_argument("--model_name", type=str, default="LLAMA3", help="Path to the model")
    parser = argparse.ArgumentParser(description="Evaluate a pruned model with lm_eval.")
    parser.add_argument("--tasks", type=str, default="hellaswag", help="Evaluation tasks")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to run evaluation (e.g., 'cuda:0')")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for evaluation")
    args = parser.parse_args()

    # Ensure all arguments have values
    if not args.model_name or not args.tasks or not args.device or not args.batch_size:
        raise ValueError("One or more required arguments are missing. Please provide values for all arguments.")

    model_name = args.model_name
    tasks = args.tasks
    device = args.device
    batch_size = args.batch_size

    max_layer = get_available_layers(model_name)
    
    print(f"Max layer available for pruning: {max_layer}")
    # Run the optimization process
    print(f"Running optimization with {args.optimizer}...")
    optimize_pruning(args.layer_to_prune, args.optimizer, max_layer)