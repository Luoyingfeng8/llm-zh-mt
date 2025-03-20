#! /bin/bash
set -eux
ROOT_DIR=$(dirname $(dirname `readlink -f $0`))

export HF_HOME="./cache/"
export HF_DATASETS_CACHE="./cache/huggingface_cache/datasets"
export HF_EVALUATE_OFFLINE=1
export HF_DATASETS_OFFLINE=1

eval_file=./eval_result.txt
comet_model=/mnt/luoyingfeng/model_card/wmt22-comet-da/checkpoints/model.ckpt
config_file=./configs/accelerate_config.yaml


## decode param
max_new_tokens=1024
num_batch=4

# nllb transformers=4.41.2, llmft
# ALMA-7B TowerInstruct-7B-v0.2 ALMA-13B TowerInstruct-13B-v0.2  Meta-Llama-3-8B-Instruct
for model_name in DeepSeek-R1-Distill-Llama-8B ; do
for l in en ja ko ru de fr it pt es ar; do
# for l in de; do
for src in zh $l; do
	if [ $src = "zh" ]; then
    	tgt=$l
	else
		tgt=zh
	fi

    ## model
    model_path=/mnt/luoyingfeng/model_card/$model_name

    ## data
    fix_lp=zh2$l
    lang_pair=${src}-$tgt
	lp=${src}2${tgt}
	src_file=$ROOT_DIR/data/zh-MT/zh-$l/test.$fix_lp.$src
	ref_file=$ROOT_DIR/data/zh-MT/zh-$l/test.$fix_lp.$tgt

    save_dir=$ROOT_DIR/exps/$model_name/zh-MT-fewoshot
    hypo_file=$save_dir/hypo.$lp.$tgt

	mkdir -p $save_dir
	cp $0 $save_dir


	mkdir -p $save_dir
	cp $0 $save_dir

    accelerate launch --config_file $ROOT_DIR/configs/accelerate_config.yaml  $ROOT_DIR/src/llm_multi_gen.py \
        --model_name_or_path $model_path \
        --test_file $src_file \
        --hypo_file $hypo_file \
        --lang_pair $lang_pair\
        --max_new_tokens $max_new_tokens \
        --num_batch $num_batch \
        --do_sample False \
        --num_beams 1
        # --topp 0.8 \
        # --topk 50 \
        # --temperature 0.1 \
        # --num_beams 5 \

    

done
done
bash eval_multi_new.sh $save_dir

done