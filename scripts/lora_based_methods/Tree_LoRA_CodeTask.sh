# This is the script to run the tran and evaluate Tree_LoRA method on the TRACE continual learning benchmark.

#get current time:
now=$(date +"%m%d_%H%M%S")
#get GPUs:
gpu_nodes="0"

#huggingface model name or path
model_name_or_path="Qwen/Qwen2.5-Coder-1.5B"
# model_name="Qwen2.5-Coder-1.5B"

codetask_tasks="CONCODE,CodeTrans,CodeSearchNet,BFP,KodCode,RunBugRun,TheVault_Csharp,CoST"
# codetask_tasks="CONCODE,CodeTrans,CodeSearchNet"

epochs=3,3,3,3,3,3,3,3

reg=0.5
num_train=10
num_eval=10
num_test=10

# Train:
echo "Start training..."
deepspeed --include=localhost:$gpu_nodes training/main.py  \
    --data_path CODETASK_HF \
    --dataset_name $codetask_tasks \
    --model_name_or_path $model_name_or_path \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --num_train $num_train \
    --num_eval $num_eval \
    --num_test $num_test \
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
    --eval_after_task \
    --output_dir ./outputs_LLM-CL/cl/$model_name/Tree_LoRA_$now \
    --reg $reg
