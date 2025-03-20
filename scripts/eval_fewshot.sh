#! /bin/bash

set -eux
ROOT_DIR=$(dirname $(dirname `readlink -f $0`))
export HF_HOME="./cache/"
export HF_DATASETS_CACHE="./cache/huggingface_cache/datasets"
export HF_EVALUATE_OFFLINE=1
export HF_DATASETS_OFFLINE=1
# export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32

# Meta-Llama-3.1-8B   Qwen2.5-7B LLaMAX3-8B Ministral-8B-Instruct-2410
for model_name in  DeepSeek-R1-Distill-Llama-8B; do
for l in en ja ko ru de fr it pt es ar; do
for src in zh $l; do
	if [ $src = "zh" ]; then
    	tgt=$l
	else
		tgt=zh
	fi

	fix_lp=zh2$l

	eval_mode=fewshot
	shot=3
	lang_pair=${src}-$tgt
	lp=${src}2${tgt}
	test_file=$ROOT_DIR/data/zh-MT/zh-$l/test.$fix_lp.json
	few_shot_file=$ROOT_DIR/data/zh-MT/zh-$l/valid.json
	src_file=$ROOT_DIR/data/zh-MT/zh-$l/test.$fix_lp.$src
	ref_file=$ROOT_DIR/data/zh-MT/zh-$l/test.$fix_lp.$tgt
	save_dir=$ROOT_DIR/exps/$model_name/zh-MT-fewoshot
	hypo_file=$save_dir/hypo.$lp.$tgt

	mkdir -p $save_dir
	cp $0 $save_dir
	model_path=/mnt/luoyingfeng/model_card/$model_name
	
	accelerate launch --config_file $ROOT_DIR/configs/accelerate_config.yaml --main_process_port 29501 $ROOT_DIR/src/eval_fewshot_zh.py \
		--model_name_or_path $model_path \
		--test_file $test_file \
		--few_shot_file $few_shot_file \
		--res_file $hypo_file \
		--lang_pair $lang_pair \
		--eval_mode $eval_mode \
		--shot $shot \
		--num_batch 3

done
done
bash eval_multi_new.sh $save_dir

done