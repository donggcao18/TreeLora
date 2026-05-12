#!/bin/bash
# Run Tree_LoRA on the executable benchmark.

export HF_HOME=./.cache
export HF_DATASETS_CACHE=./.cache

set -euo pipefail

# Allow override via environment variables.
gpu_nodes="${GPU_NODES:-0}"
export CUDA_VISIBLE_DEVICES="$gpu_nodes"

# Dataset root for executable benchmark.
data_path="${DATA_PATH:-/path/to/LLM-CL-Benchmark_5000}"

# Model selection
model_name="Qwen2.5-Coder-1.5B"

num_train=-1
num_eval=3
num_test=-1

now=$(date +"%m%d_%H%M%S")
port=$(shuf -i25000-30000 -n1)

# Train:
echo "Start training..."
deepspeed --include=localhost:$gpu_nodes --master_port "$port" training/main.py  \
    --data_path "$data_path" \
    --dataset_name all \
    --benchmark executable \
    --model_name_or_path ./PTM/$model_name \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 8 \
    --max_prompt_len 1024 \
    --max_ans_len 2048 \
    --learning_rate 1e-4 \
    --weight_decay 0. \
    --num_train_epochs 3,3,3,3,3,3,3,3,3 \
    --gradient_accumulation_steps 8 \
    --lr_scheduler_type cosine \
    --num_warmup_steps 0 \
    --seed 1234 \
    --zero_stage 2 \
    --deepspeed \
    --print_loss \
    --CL_method Tree_LoRA \
    --output_dir ./outputs_LLM-CL/cl/$model_name/Tree_LoRA_executable_$now \
    --num_train $num_train \
    --num_eval $num_eval \
    --num_test $num_test

# Inference:
echo "Start inference..."
python inference/infer_multi_command.py  \
    --gpus=$gpu_nodes \
    --data_path "$data_path" \
    --benchmark executable \
    --inference_tasks python,cpp,swift,rust,csharp,java,php,typescript,shell \
    --model_name_or_path ./PTM/$model_name \
    --inference_model_path ./outputs_LLM-CL/cl/$model_name/Tree_LoRA_executable_$now \
    --inference_batch 8 \
    --max_prompt_len 1024 \
    --max_ans_len 2048 \
    --temperature 0.2 \
    --top_p 0.95 \
    --repetition_penalty 1.0 \
    --do_sample \
    --num_return_sequences 1 \
    --seed 1234 \
    --CL_method Tree_LoRA \
    --inference_output_path ./outputs_LLM-CL/cl/$model_name/Tree_LoRA_executable_$now/predictions \
    --num_eval $num_eval \
    --num_test $num_test \
    --num_train $num_train
