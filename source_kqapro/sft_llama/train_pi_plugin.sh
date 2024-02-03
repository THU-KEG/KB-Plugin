nvidia-smi

total_bs=256
bs=16
num_gpus=1
lr=1e-5
model_name_or_path="../../models_hf/llama-2-7b"
num_train_epochs=3
save_steps=1000

total_per_device=$((${total_bs}/${num_gpus}))
accu=$(( ${total_per_device} / ${bs} ))

run_name="train_pi_plugin_bs${bs}_accu${accu}_kb-epoch3"

data_path="../../data/kqapro/PI/multi_kb"
schema_plugin_path="${data_path}/schema_plugin_paths_3epoch.json"
echo "Run: ${run_name}"

cache_dir=../../cache
out_dir=../../checkpoints/kqapro/pi_plugin/$run_name
mkdir -p $out_dir

deepspeed --include localhost:1 --master_port 61016 \
    train_pi_plugin.py \
    --model_name_or_path $model_name_or_path \
    --cache_dir $cache_dir \
    --processed_data_dir $data_path \
    --schema_plugin_path $schema_plugin_path \
    --remove_unused_columns false \
    --run_name $run_name \
    --logging_steps 1 \
    --logging_dir $out_dir \
    --log_level info \
    --output_dir $out_dir \
    --prompt_column input \
    --response_column answer \
    --num_train_epochs $num_train_epochs \
    --per_device_train_batch_size $bs \
    --per_device_eval_batch_size $bs \
    --gradient_accumulation_steps $accu \
    --lora_r 16 \
    --lora_alpha 32 \
    --target_modules q_proj,k_proj,v_proj,o_proj,down_proj,gate_proj,up_proj \
    --save_strategy "steps" \
    --save_steps $save_steps \
    --disable_tqdm false \
    --report_to "none" \
    --learning_rate $lr \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --gradient_checkpointing 1 \
    --deepspeed ds_config/stage2.json \
    --bf16 \
    2>&1 | tee $out_dir/output.log
