# !/bin/bash
set -eux
ROOT_DIR=$(dirname $(dirname `readlink -f $0`))

export HF_HOME="./cache/"
export HF_DATASETS_CACHE="./cache/huggingface_cache/datasets"
export HF_EVALUATE_OFFLINE=1
export HF_DATASETS_OFFLINE=1

decode_dir=${1:-"/mnt/luoyingfeng/effllm/exps/sampling_50/TowerInstruct-7B-v0.2/wmt23"}
suffix=

comet_model=/mnt/luoyingfeng/model_card/wmt22-comet-da/checkpoints/model.ckpt 
xcome_model=/mnt/luoyingfeng/model_card/XCOMET-XXL/checkpoints/model.ckpt

src_file_strs=""
ref_file_strs=""
hypo_file_strs=""
lang_pair_strs=""

for l in en ja ko ru de fr it pt es; do
for src in zh $l; do
    if [ $src = "zh" ]; then
        tgt=$l
    else 
        tgt=zh
    fi
    
    fix_lp=zh-$l
    lang_pair=${src}-$tgt
    lp=${src}2${tgt}
    # hypo_file=$decode_dir/${lang_pair}.txt
    # hypo_file=$decode_dir/hypo.${lp}.txt
    # hypo_file=$decode_dir/$lp/hypo.${lp}.txt
    # hypo_file=$decode_dir/niu.${lp}.txt
    hypo_file=$decode_dir/devtest.${lp}
    # hypo_file=$decode_dir/hypo_${lang_pair}.txt
    src_file=$ROOT_DIR/data/flores200/zh-$l/test.$fix_lp.$src
	ref_file=$ROOT_DIR/data/flores200/zh-$l/test.$fix_lp.$tgt
    
    src_file_strs=${src_file_strs:+$src_file_strs,}$src_file
    ref_file_strs=${ref_file_strs:+$ref_file_strs,}$ref_file
    hypo_file_strs=${hypo_file_strs:+$hypo_file_strs,}$hypo_file
    lang_pair_strs=${lang_pair_strs:+$lang_pair_strs,}$lp
        
done
done


# metric="bleu,comet_22,xcomet_xxl" 
metric="bleu,comet_22" 
python $ROOT_DIR/src/mt_scoring.py \
    --metric $metric  \
    --comet_22_path $comet_model \
    --xcomet_xxl_path $xcome_model \
    --lang_pair $lang_pair_strs \
    --src_file $src_file_strs \
    --ref_file $ref_file_strs \
    --hypo_file $hypo_file_strs \
    --record_file "result.xlsx"
