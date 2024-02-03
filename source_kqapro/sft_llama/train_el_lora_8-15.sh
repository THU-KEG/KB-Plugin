nvidia-smi

total_bs=128
bs=128
num_gpus=1
lr=1e-5
model_name_or_path="../../models_hf/llama-2-7b"
num_train_epochs=3
save_steps=1000

total_per_device=$((${total_bs}/${num_gpus}))
accu=$(( ${total_per_device} / ${bs} ))

kb_idxs=(9 10 11 12 13 14 15)
for kb_idx in ${kb_idxs[@]}; do
    run_name="train_el_lora_${kb_idx}_3epoch"
    data_path="../../data/kqapro/diff_kb/${kb_idx}/hrt"

    echo "Run: ${run_name}"

    cache_dir=../../cache
    out_dir=../../checkpoints/kqapro/schema_plugin/$run_name
    mkdir -p $out_dir

    deepspeed --include localhost:`expr ${kb_idx} % 8` --master_port `expr 61000 + ${kb_idx}` \
        train_el_lora.py \
        --model_name_or_path $model_name_or_path \
        --cache_dir $cache_dir \
        --processed_data_dir $data_path \
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
        --save_strategy "epoch" \
        --disable_tqdm false \
        --report_to "none" \
        --save_total_limit 3 \
        --learning_rate $lr \
        --weight_decay 0. \
        --warmup_ratio 0.03 \
        --lr_scheduler_type "cosine" \
        --gradient_checkpointing 1 \
        --deepspeed ds_config/stage2.json \
        --bf16 \
        > $out_dir/output.log 2>&1 &
done

kb_idx=8
run_name="train_el_lora_${kb_idx}_3epoch"
data_path="../../data/kqapro/diff_kb/${kb_idx}/hrt"

echo "Run: ${run_name}"

cache_dir=../../cache
out_dir=../../checkpoints/kqapro/schema_plugin/$run_name
mkdir -p $out_dir

deepspeed --include localhost:`expr ${kb_idx} % 8` --master_port `expr 61000 + ${kb_idx}` \
    train_el_lora.py \
    --model_name_or_path $model_name_or_path \
    --cache_dir $cache_dir \
    --processed_data_dir $data_path \
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
    --save_strategy "epoch" \
    --disable_tqdm false \
    --report_to "none" \
    --save_total_limit 3 \
    --learning_rate $lr \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --gradient_checkpointing 1 \
    --deepspeed ds_config/stage2.json \
    --bf16 \
    2>&1 | tee $out_dir/output.log