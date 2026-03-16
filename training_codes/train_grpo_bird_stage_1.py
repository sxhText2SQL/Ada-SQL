import torch
from datasets import load_dataset, Dataset
from peft import LoraConfig, get_peft_model
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import GRPOConfig, GRPOTrainer
import json
import deepspeed
import os
from reward_functions_stage1 import calculate_reward
import argparse
from transformers import GenerationConfig
import argparse
import logging
from logging.handlers import RotatingFileHandler


logger = logging.getLogger('pr_grpo')
logger.setLevel(logging.INFO)
log_file = ''

handler = RotatingFileHandler(
    filename=log_file,
    maxBytes=20 * 1024 * 1024,
    backupCount=3,
    encoding='utf-8'
)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

logger.addHandler(handler)


model_path = ""
data_path = ""
output_dir = ""

dataset = load_dataset("json", data_files=data_path)

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
train_data = []
for item in dataset["train"]:
    prompt = tokenizer.apply_chat_template( [{"role": "user", "content": item["input_prompt"]}], add_generation_prompt=True, tokenize=False )
    train_data.append({
        "prompt": prompt,
        "completion": item["output_answer"]
    })

train_data = Dataset.from_list(train_data)

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    dtype=torch.bfloat16,
    # device_map="auto",
    attn_implementation="flash_attention_2"
)

peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    task_type="CAUSAL_LM"
)


def reward_process_bird(prompts, completions, **kwargs):
    global trainer
    step = trainer.state.global_step
    total_steps = trainer.state.max_steps

    prompt = prompts[0]
    reward_list = calculate_reward(prompt, completions,step,total_steps=total_steps)
    print(reward_list)
    return reward_list


training_args = GRPOConfig(
    output_dir=output_dir,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=2,
    learning_rate=2e-6,
    num_train_epochs=1.0,
    logging_steps=10,
    save_steps=500,
    save_total_limit=100,



    lr_scheduler_type='cosine',

    num_generations=16,
    generation_batch_size=16,
    gradient_checkpointing=True,

    use_vllm=True,
    vllm_mode="colocate",
    vllm_gpu_memory_utilization=0.5,


    max_prompt_length=6144,
    max_completion_length=2048,
    top_p=1.0,
    temperature=0.8,
    top_k=-1,
    repetition_penalty = 1.0,


    beta = 0.001,
    epsilon=0.20
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

trainer = GRPOTrainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    reward_funcs=[reward_process_bird]
)

trainer.train()
trainer.save_model(f"{output_dir}/final_checkpoint")
