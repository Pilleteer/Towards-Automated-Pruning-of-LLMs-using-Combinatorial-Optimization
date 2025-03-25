import argparse
import importlib
from mealpy import FloatVar, Problem
from mealpy.utils import io
from prune import prune_model
from evaluate import evaluate_model

def objective_func(x):
    """Objective function for optimization: Prune, evaluate, and return accuracy."""
    x = [int(i) for i in x]
    prune_model("LLAMA3", x, "./LLAMA3/prune_llm")
    acc = evaluate_model("./LLAMA3/prune_llm")
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

def optimize_pruning(layer_to_prune, optimizer_name):
    """Runs the selected optimizer to find the best layer combination."""
    
    problem = {
        "obj_func": objective_func,
        "bounds": FloatVar(lb=[i for i in range(layer_to_prune)], ub=[i for i in range(32 - layer_to_prune, 32)]),
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

    args = parser.parse_args()
    optimize_pruning(args.layer_to_prune, args.optimizer)
