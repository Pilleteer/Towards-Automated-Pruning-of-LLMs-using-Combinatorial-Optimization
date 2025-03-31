import os
from dataclasses import dataclass, field
from typing import Optional
from datasets.arrow_dataset import Dataset
import torch
import json
import subprocess
from datasets import load_dataset
from peft import LoraConfig
from peft import AutoPeftModelForCausalLM
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    AutoTokenizer,
    TrainingArguments,
)

from utils import generate_config, delete_tensor_files, get_available_layers
from evaluate import evaluate_model
from transformers import HfArgumentParser, TrainingArguments
from trl import SFTTrainer
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script Arguments")
    parser.add_argument("--model_name", type=str, required=True, help="The model name to use.")
    parser.add_argument("--dataset_name", type=str, default="vicgalle/alpaca-gpt4", required=True, help="The dataset name to use.")
    parser.add_argument("--num_train_epochs", type=int, default=1, help="Number of training epochs.")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="Learning rate.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1, help="Batch size per device.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="Gradient accumulation steps.")
    parser.add_argument("--max_seq_length", type=int, default=256, help="Maximum sequence length.")
    parser.add_argument("--output_dir", type=str, default="./results_packing", help="Output directory.")
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every X updates steps.")
    parser.add_argument("--prune_layers", type=str, help="Comma-separated list of layers to prune. eg: 0,1,2,3")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to run evaluation (e.g., 'cuda:0')")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for evaluation")
    parser.add_argument("--tasks", type=str, default="hellaswag", help="Evaluation tasks")
    parser.add_argument("--max_layers", type=int, default=0, help="Maximum number of layers available for pruning.")

    args = parser.parse_args()
    
pruned_path = "./pruned_model"
model_name = args.model_name
if not os.path.exists(pruned_path):
    os.makedirs(pruned_path)
x = [int(i) for i in args.prune_layers.split(",")]
max_layers = get_available_layers(model_name)
config = generate_config(model_name, x, max_layers)
with open("./prune_layer_config.yaml", "w") as f:
    json.dump(config, f)
delete_tensor_files(pruned_path)
# Command: mergekit-yaml ./yaml_config/prune_layer_config.yaml MODEL_PATH --cuda
result = subprocess.run(["mergekit-yaml", "./prune_layer_config.yaml", pruned_path, "--cuda"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)


torch.manual_seed(42)


@dataclass
class ScriptArguments:
    """
    These arguments vary depending on how many GPUs you have, what their capacity and features are, and what size model you want to train.
    """


    local_rank: Optional[int] = field(default=-1, metadata={"help": "Used for multi-gpu"})


    per_device_train_batch_size: Optional[int] = field(default=args.per_device_train_batch_size)
    per_device_eval_batch_size: Optional[int] = field(default=2)
    gradient_accumulation_steps: Optional[int] = field(default=args.gradient_accumulation_steps)
    learning_rate: Optional[float] = field(default=args.learning_rate)
    max_grad_norm: Optional[float] = field(default=0.3)
    weight_decay: Optional[int] = field(default=0.01)
    lora_alpha: Optional[int] = field(default=16)
    lora_dropout: Optional[float] = field(default=0.1)
    lora_r: Optional[int] = field(default=8)
    max_seq_length: Optional[int] = field(default=args.max_seq_length)
    model_name: Optional[str] = field(
        default=pruned_path,
        metadata={
            "help": "The model that you want to train from the Hugging Face hub. E.g. gpt2, gpt2-xl, bert, etc."
        }
    )
    dataset_name: Optional[str] = field(
        default=args.dataset_name,
        metadata={"help": "The preference dataset to use."},
    )

    use_4bit: Optional[bool] = field(
        default=True,
        metadata={"help": "Activate 4bit precision base model loading"},
    )
    use_nested_quant: Optional[bool] = field(
        default=False,
        metadata={"help": "Activate nested quantization for 4bit base models"},
    )
    bnb_4bit_compute_dtype: Optional[str] = field(
        default="float16",
        metadata={"help": "Compute dtype for 4bit base models"},
    )
    bnb_4bit_quant_type: Optional[str] = field(
        default="nf4",
        metadata={"help": "Quantization type fp4 or nf4"},
    )
    num_train_epochs: Optional[int] = field(
        default=args.num_train_epochs,
        metadata={"help": "The number of training epochs for the reward model."},
    )
    fp16: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables fp16 training."},
    )
    bf16: Optional[bool] = field(
        default=True,
        metadata={"help": "Enables bf16 training."},
    )
    packing: Optional[bool] = field(
        default=True,
        metadata={"help": "Use packing dataset creating."},
    )
    gradient_checkpointing: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables gradient checkpointing."},
    )
    optim: Optional[str] = field(
        default="adamw_torch",
        metadata={"help": "The optimizer to use."},
    )
    lr_scheduler_type: str = field(
        # default="cosine_with_warmup",
        # default="cosine",
        default="constant_with_warmup",

        metadata={"help": "Learning rate schedule. Constant a bit better than cosine, and has advantage for analysis"},
    )
    # max_steps: int = field(default=4375, metadata={"help": "How many optimizer update steps to take"})
    warmup_steps: int = field(default=200, metadata={"help": "# of steps to do a warmup for"})
    group_by_length: bool = field(
        default=True,
        metadata={
            "help": "Group sequences into batches with same length. Saves memory and speeds up training considerably."
        },
    )
    save_steps: int = field(default=args.save_steps, metadata={"help": "Save checkpoint every X updates steps."})
    # save_strategy: Optional[str] = field(
    #             default="epoch",
    #             metadata={"help": "The strategy to use for saving checkpoints. Options: 'steps', 'epoch'."},
    # )
    logging_steps: int = field(default=10, metadata={"help": "Log every X updates steps."})
    merge_and_push: Optional[bool] = field(
        default=True,
        metadata={"help": "Merge and push weights after training"},
    )
    output_dir: str = field(
        default=args.output_dir,
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."},
    )




parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]




def gen_batches_train():
    ds = load_dataset(script_args.dataset_name, streaming=True, split="train")




    for sample in iter(ds):


        # Extract instruction and input from the sample
        instruction = str(sample['instruction'])
        input_text = str(sample['input'])
        out_text = str(sample['output'])
        formatted_prompt = None 
            
        if input_text is None or input_text == "":
            formatted_prompt = (
                f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
                f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:\n"
                f"<|eot_id|><|start_header_id|>asssitant<|end_header_id|>\n\n",
                f"{str(out_text)}"
                f"<|eot_id|><|end_of_text|>"
            )
        else:
            formatted_prompt = (
                f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
                f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n"
                f"<|eot_id|><|start_header_id|>asssitant<|end_header_id|>\n\n"
                f"{str(out_text)}"
                f"<|eot_id|><|end_of_text|>"
            )
        
        formatted_prompt = "".join(formatted_prompt)
        yield {'text': formatted_prompt}






def create_and_prepare_model(args):
    compute_dtype = getattr(torch, args.bnb_4bit_compute_dtype)
    # commented qlora stuff 
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=args.use_4bit,
        bnb_4bit_quant_type=args.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=args.use_nested_quant,
    )


    if compute_dtype == torch.float16 and args.use_4bit:
        major, _ = torch.cuda.get_device_capability()
        if major >= 8:
            print("=" * 80)
            print("Your GPU supports bfloat16, you can accelerate training with the argument --bf16")
            print("=" * 80)


    # Load the entire model on the GPU 0
    # switch to `device_map = "auto"` for multi-GPU
    # device_map = {"": 0}
    device_map = "auto"


    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, 
        quantization_config=bnb_config, 
        device_map=device_map, 
        token=True,
    )
    
    
    peft_config = LoraConfig(
        lora_alpha=script_args.lora_alpha,
        lora_dropout=script_args.lora_dropout,
        # target_modules=["query_key_value"], 
        r=script_args.lora_r,
        bias="none",
        task_type="CAUSAL_LM", 
        target_modules=['q_proj', 'v_proj'],
    )


    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token


    return model, peft_config, tokenizer

training_arguments = TrainingArguments(
    num_train_epochs=script_args.num_train_epochs,
    output_dir=script_args.output_dir,
    per_device_train_batch_size=script_args.per_device_train_batch_size,
    gradient_accumulation_steps=script_args.gradient_accumulation_steps,
    optim=script_args.optim,
    save_steps=script_args.save_steps,
    # save_strategy=script_args.save_strategy,
    logging_steps=script_args.logging_steps,
    learning_rate=script_args.learning_rate,
    fp16=script_args.fp16,
    bf16=script_args.bf16,
    max_grad_norm=script_args.max_grad_norm,
    # max_steps=script_args.max_steps,
    warmup_steps=script_args.warmup_steps,
    group_by_length=script_args.group_by_length,
    lr_scheduler_type=script_args.lr_scheduler_type,
)


model, peft_config, tokenizer = create_and_prepare_model(script_args)


train_gen = Dataset.from_generator(gen_batches_train)


tokenizer.padding_side = "right"


trainer = SFTTrainer(
    model=model,
    train_dataset=train_gen,
    peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length=script_args.max_seq_length,
    tokenizer=tokenizer,
    args=training_arguments,
    packing=script_args.packing,
)


trainer.train()
# trainer.train(resume_from_checkpoint=True)

if script_args.merge_and_push:
    output_dir = os.path.join(script_args.output_dir, "final_checkpoints")
    trainer.model.save_pretrained(output_dir)


    # Free memory for merging weights
    del model
    torch.cuda.empty_cache()


    model = AutoPeftModelForCausalLM.from_pretrained(output_dir, device_map="auto", torch_dtype=torch.bfloat16)
    model = model.merge_and_unload()


    output_merged_dir = os.path.join(script_args.output_dir, "final_merged_checkpoint")
    print(f"Saving merged model to {output_merged_dir}")
    model.save_pretrained(output_merged_dir, safe_serialization=True)
    evaluate_model(output_merged_dir, model_name, args.tasks, args.device, args.batch_size)



