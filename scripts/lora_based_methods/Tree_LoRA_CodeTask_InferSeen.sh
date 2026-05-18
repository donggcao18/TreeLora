#!/usr/bin/env bash

# Evaluate Tree_LoRA checkpoints 0..7 on all seen CodeTask test sets.
# Set checkpoint_path to the folder that contains subfolders 0, 1, ..., 7.

gpu_nodes="1"
model_name_or_path="Qwen/Qwen2.5-Coder-1.5B"
checkpoint_path="./output_models/TreeLora_Qwen2.5-Coder-1.5B/Tree_LoRA_0513_093148"

codetask_tasks="CONCODE,CodeTrans,CodeSearchNet,BFP,KodCode,RunBugRun,TheVault_Csharp,CoST"
max_prompt_len="320,320,256,130,512,256,256,256"
max_ans_len="150,256,128,120,300,128,128,128"

CUDA_VISIBLE_DEVICES=$gpu_nodes python inference/infer_treelora_seen_tasks.py \
    --data_path CODETASK_HF \
    --model_name_or_path $model_name_or_path \
    --checkpoint_path $checkpoint_path \
    --start_task_id 0 \
    --end_task_id 7 \
    --inference_tasks $codetask_tasks \
    --max_prompt_len $max_prompt_len \
    --max_ans_len $max_ans_len \
    --per_device_eval_batch_size 128 \
    --num_test -1 \
    --seed 1234 \
    --output_dir "$checkpoint_path/seen_task_eval"
