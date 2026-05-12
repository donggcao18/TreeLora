#!/bin/bash
# Run Tree_LoRA on the executable benchmark.
# This script uses 2 GPUs, 24GB VRAM each.
# Allow override via environment variables.
gpu_nodes="${GPU_NODES:-0,1}"
export CUDA_VISIBLE_DEVICES="$gpu_nodes"
HF_MODEL_REPO_ID="ankhanhtran02/TreeLoRA_Qwen2.5-Coder-1.5B_executable"

# Model selection
model_name_or_path="Qwen/Qwen2.5-Coder-1.5B"
model_name="Qwen2.5-Coder-1.5B"

epochs=1,1,1,1,1,1,1,1,1

reg=0.5
num_train=-1
num_eval=3
num_test=-1

now=$(date +"%m%d_%H%M%S")

# Train:
echo "Start training..."
deepspeed --include=localhost:$gpu_nodes training/main.py  \
    --data_path EXECUTABLE_HF \
    --dataset_name all \
    --benchmark executable \
    --model_name_or_path $model_name_or_path \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 16 \
    --max_prompt_len 1024 \
    --max_ans_len 2048 \
    --learning_rate 1e-4 \
    --weight_decay 0. \
    --num_train_epochs $epochs \
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
    --num_test $num_test \
    --reg $reg \
    --eval_after_task \
    --do_sample 


python upload_output_to_hf.py \
  --output-dir "./outputs_LLM-CL/cl/$model_name/Tree_LoRA_executable_$now" \
  --repo-id "$HF_MODEL_REPO_ID" \
  --commit-message "Upload TreeLoRA executable outputs"