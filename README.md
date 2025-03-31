# Automated Pruning Framework for Large Language Models using Combinatorial Optimization
# Model Pruning and Optimization Pipeline

This repository provides a streamlined pipeline for pruning, optimizing, and evaluating large language models (LLMs). The process is fully configurable via a YAML file.

## 📌 Features
- **Model Pruning**: Supports different pruning strategies to reduce model size.
- **Optimization**: Uses a combinatorial optimization algorithm to fine-tune the pruning process.
- **Evaluation**: Benchmarks the pruned model on specified tasks.
- **Automated Execution**: The entire pipeline is managed by a single script (`run.sh`).

## 🚀 Installation
Ensure you have the required dependencies installed:
```bash
pip install yq
pip install -r requirements.txt
```
Additionally, install `yq` for parsing YAML in bash scripts:
```bash
sudo apt-get install -y yq  # For Ubuntu
brew install yq  # For macOS
```

## 🔧 Configuration
The pipeline is configured using `config.yaml`. Below is an example:
```yaml
model:
  name: "llama-7b"
  path: "/path/to/original/model"
  max_layers: 32

pruning:
  method: "magnitude"
  selected_layers: [2, 4, 6, 8, 10]

optimization:
  algorithm: "genetic"
  num_generations: 50
  population_size: 20

evaluation:
  model_name: "llama-7b-pruned"
  tasks: ["hellaswag", "piqa", "arc"]
  device: "cuda"
  batch_size: 16
```

## ▶️ Running the Pipeline
Execute the script with:
```bash
bash run.sh
```
This will:
1. **Prune the model** based on the specified method.
2. **Optimize the pruning strategy** using a search algorithm.
3. **Evaluate the pruned model** on defined benchmark tasks.

## ⚡ Notes
- Ensure the model path exists before running the script.
- Adjust `config.yaml` based on the model and hardware availability.
- The optimization process might take significant time depending on parameters.