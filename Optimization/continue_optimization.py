import torch
import json
import subprocess
import re
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
# from accelerate import Accelerator
import numpy as np
import time
from Optimization.yaml_gen import generate_config, count_nonzero_params, get_acc, get_model_size, get_path_size, src_config, objective_func, delete_tensor_files
print("PyTorch version:", torch.__version__)
print("CUDA version:", torch.version.cuda)

# accelerator = Accelerator()
# device = accelerator.device

model_name = "LLAMA3"
original_model_size = 15316.60 #Mb
original_model_acc = 0.7918
original_model_time = 566.7846715450287

pruned_path = "./LLAMA3/prune_llm"

from mealpy import PSO, FloatVar, Problem
from mealpy.utils import io
# Load mealpy model
saved_model = io.load_model("mealpy_results/model.pkl")
# Continue solving the problem
optimizer.epoch = 4  # Set additional epochs to continue solving
optimizer.pop_size = 10  # Modify population size (optional)
g_best_new = optimizer.solve(problem)
print(f"Continued Best Solution: {g_best_new.solution}, Best Fitness: {g_best_new.target.fitness}")