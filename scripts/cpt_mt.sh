#! /bin/bash
set -eux
ROOT_DIR=$(dirname $(dirname `readlink -f $0`))

export HF_HOME="$ROOT_DIR/cache/"
export HF_DATASETS_CACHE="$ROOT_DIR/cache/huggingface_cache/datasets"
export HF_EVALUATE_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# model
model_name=Qwen2.5-7B
model_dir=/mnt/luoyingfeng/model_card/$model_name
config_file=$ROOT_DIR/configs/ds_z2_config_bf16.json
# resume_from_checkpoint=

# data
dataset_dir=$ROOT_DIR/data
train_data=train1,train2,train3,train4
eval_dataset=valid_cpt
template=default
per_device_train_batch_size=4
per_device_eval_batch_size=8
gradient_accumulation_steps=16
max_lengths=2048
max_steps=10000

# save
task=cpt
tag=test

output_dir=$ROOT_DIR/exps/$model_name/$task/$tag
mkdir -p $output_dir
cp $0 $output_dir

llamafactory-cli train \
    --deepspeed  $config_file \
    --stage pt \
    --streaming True \
    --finetuning_type full \
    --model_name_or_path $model_dir \
    --dataset_dir $dataset_dir \
    --dataset $train_data \
    --eval_dataset $eval_dataset \
    --template $template \
    --cutoff_len  $max_lengths \
    --do_train True \
    --do_eval True \
    --use_fast_tokenizer True \
    --learning_rate 2e-5 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.01 \
    --max_steps $max_steps \
    --per_device_train_batch_size $per_device_train_batch_size \
    --per_device_eval_batch_size $per_device_eval_batch_size \
    --gradient_accumulation_steps $gradient_accumulation_steps \
    --evaluation_strategy steps \
    --save_strategy steps \
    --logging_strategy steps \
    --eval_steps 0.1 \
    --save_steps 0.1 \
    --logging_steps 0.01 \
    --num_beams 5 \
    --max_new_tokens 256 \
    --do_sample False \
    --dataloader_num_workers 8 \
    --output_dir $output_dir \
    --overwrite_output_dir True \
    --bf16 True \
    --seed 42  \
    --cache_dir $ROOT_DIR/cache \
    --report_to "tensorboard" \
    --plot_loss True \
    --ddp_timeout 180000000  | tee $output_dir/train.log

