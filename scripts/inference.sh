#! /bin/bash
set -eux
ROOT_DIR=$(dirname $(dirname `readlink -f $0`))

export HF_HOME="$ROOT_DIR/cache/"
export HF_DATASETS_CACHE="$ROOT_DIR/cache/huggingface_cache/datasets"
export HF_EVALUATE_OFFLINE=1
export HF_DATASETS_OFFLINE=1

config_file=$ROOT_DIR/configs/accelerate_config.yaml

# model
predict_model_dir=/mnt/luoyingfeng/lora4mt/exps/Qwen2.5-7B/data_quantity/15k_new/checkpoint-2145

# data
dataset_dir=$ROOT_DIR/data
template=default

# eval
comet_model=/mnt/luoyingfeng/model_card/wmt22-comet-da/checkpoints/model.ckpt 
xcome_model=/mnt/luoyingfeng/model_card/XCOMET-XXL/checkpoints/model.ckpt

lang_pair_strs=""
src_file_strs=""
ref_file_strs=""
hypo_file_strs=""

for lang in en ja ko ru de fr it pt es;do
    for src in $lang zh ;do 

        if [ $src = "zh" ]; then # en2zh
            src_lang=zh
            tgt_lang=$lang 
        else  # zh2en
            src_lang=$lang
            tgt_lang=zh 
        fi

        lp=$src_lang2$tgt_lang
        src_file=$ROOT_DIR/data/flores200/zh-${lang}/test.zh-$lang.$src_lang
        src_file=$ROOT_DIR/data/flores200/zh-${lang}/test.zh-$lang.$tgt_lang

        test_dataset=test_$lp
        output_dir=$predict_model_dir/decode_result/$lp
        mkdir -p $output_dir
        cp $0 $output_dir

        llamafactory-cli train \
            --model_name_or_path $predict_model_dir \
            --dataset_dir $dataset_dir \
            --eval_dataset $test_dataset \
            --template $template \
            --cutoff_len 1024 \
            --max_length 1024 \
            --max_new_tokens 256 \
            --do_train False \
            --do_eval False \
            --do_predict \
            --use_fast_tokenizer True \
            --per_device_eval_batch_size 8 \
            --predict_with_generate \
            --dataloader_num_workers 8 \
            --preprocessing_num_workers 16 \
            --output_dir $output_dir \
            --overwrite_output_dir True \
            --num_beams 5 \
            --do_sample False \
            --fp16 True \
            --seed 42  \
            --cache_dir $ROOT_DIR/cache/ \
            | tee $output_dir/train.log

        jq -r '.predict' $output_dir/generated_predictions.jsonl > $output_dir/hypo.$lp.txt
        
        lang_pair_strs=${lang_pair_strs:+$lang_pair_strs,}$lp
        src_file_strs=${src_file_strs:+$src_file_strs,}$src_file
        ref_file_strs=${ref_file_strs:+$ref_file_strs,}$ref_file
        hypo_file_strs=${hypo_file_strs:+$hypo_file_strs,}$hypo_file

    done
done

# metric="bleu,comet_22,xcomet_xxl" 
metric="bleu,comet_22,xcomet_xxl" 
python $ROOT_DIR/src/mt_scoring.py \
    --metric $metric  \
    --comet_22_path $comet_model \
    --xcomet_xxl_path $xcome_model \
    --lang_pair $lang_pair_strs \
    --src_file $src_file_strs \
    --ref_file $ref_file_strs \
    --hypo_file $hypo_file_strs \
    --record_file "result.xlsx"
