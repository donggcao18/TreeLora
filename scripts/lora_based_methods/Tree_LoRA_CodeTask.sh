# This is the script to run the tran and evaluate Tree_LoRA method on the TRACE continual learning benchmark.

#get current time:
now=$(date +"%m%d_%H%M%S")
#get GPUs:
gpu_nodes="0"

# model_name_or_path="meta-llama/Llama-3.2-1B-Instruct"
#model_name_or_path="meta-llama/Llama-2-7b-chat"
#model_name_or_path="meta-llama/Llama-3.1-8B-Instruct"
model_name_or_path="Qwen/Qwen2.5-Coder-1.5B"
model_name="Qwen2.5-Coder-1.5B"
#model_name_or_path="mistralai/Mistral-7B-Instruct-v0.3"
#model_name_or_path="google/gemma-2b-it"

#epochs=1,1,5,5,1,5,5,5
epochs=1,1,1,1
#epochs=5,3,7,5,3,5,5,7

reg=0.5
num_train=100
num_eval=100
num_test=100

# Train:
echo "Start training..."
deepspeed --include=localhost:$gpu_nodes training/main.py  \
    --data_path CODETASK_HF \
    --dataset_name CONCODE,CodeTrans,CodeSearchNet,BFP \
    --model_name_or_path $model_name_or_path \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 8 \
    --max_prompt_len 1024 \
    --max_ans_len 512 \
    --num_train $num_train \
    --num_eval $num_eval \
    --num_test $num_test \
    --learning_rate 1e-4 \
    --weight_decay 0. \
    --num_train_epochs $epochs \
    --gradient_accumulation_steps 1 \
    --lr_scheduler_type cosine \
    --num_warmup_steps 0 \
    --seed 1234 \
    --zero_stage 2 \
    --deepspeed \
    --print_loss \
    --CL_method Tree_LoRA \
    --output_dir ./outputs_LLM-CL/cl/$model_name/Tree_LoRA_$now \
    --reg $reg


# Inference:
echo "Start inference..."
python inference/infer_multi_command.py  \
    --gpus=$gpu_nodes \
    --data_path CODETASK_HF \
    --inference_tasks CONCODE,CodeTrans,CodeSearchNet,BFP \
    --model_name_or_path $model_name_or_path \
    --inference_model_path ./outputs_LLM-CL/cl/$model_name/Tree_LoRA_$now \
    --inference_batch 32 \
    --max_prompt_len 1024 \
    --max_ans_len 512 \
    --num_train $num_train \
    --num_eval $num_eval \
    --num_test $num_test \
    --seed 1234 \
    --CL_method Tree_LoRA \
    --inference_output_path ./outputs_LLM-CL/cl/$model_name/Tree_LoRA_$now/predictions

# Collect results:
echo "Start collecting results..."
python inference/collect_results.py --inference_tasks CONCODE,CodeTrans,CodeSearchNet,BFP --data_path ./outputs_LLM-CL/cl/$model_name/Tree_LoRA_$now/predictions
