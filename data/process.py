# coding=utf8
import json
import tqdm
import random
random.seed(42)
import os
 

def merge_valid():
    for lang in ["en", "ja", "ko", "ru", "de", "fr", "it", "pt", "es"]:
        d1 = json.load(open(f"/mnt/luoyingfeng/llm-zh-mt/data/flores200/zh-{lang}/valid.json")) 
        # d2 = json.load(open(f"/mnt/luoyingfeng/llm-zh-mt/data/NTREX/zh-{lang}/valid.json")) 
        # data = d1 + d2
        f_out = open(f"/mnt/luoyingfeng/llm-zh-mt/data/valid/valid.zh-{lang}.jsonl", mode='w')
        for ele in d1:
            f_out.write(json.dumps(ele, ensure_ascii=False) + "\n")


def sample_mono_data(need):
    # tokens/G
    mono_ratio = {"zh":0.2403, "en":0.2427, "ja":0.2719, "ko":0.3156, "ru":0.2153, "de":0.2997, "fr":0.2951, "it":0.3123, "pt":0.3031, "es":0.2886}

    data = []
    # make monolingual data
    for lang in mono_ratio.keys():
        size = (need[lang]/mono_ratio[lang]) * 1024 * 1024 * 1024
        print(f"monollingual, {lang}, sample size: {size/( 1024 * 1024)} M")
        acc_size = 0
        acc_n = 0
        for line in tqdm.tqdm(open(f"./mono_data/{lang}.jsonl")):
            text = json.loads(line)["text"]
            acc_size += len(text.encode('utf-8'))
            if acc_size < size:
                data.append(line)
                acc_n += 1
            else:
                break
        print(f"monollingual, {lang}, sample line: {acc_n}\n")
    print(f"all monollingual: {len(data)}\n")
    return data

prompt = "{src} {direc} {tgt}"

def sample_bili_data(need):
    ## 得到预算数量，正向和反向数据各占一半
    # tokens/G
    bili_ratio = { "zh-en":0.3340, "zh-ja":0.3727, "zh-ko":0.3869, "zh-ru":0.2656, "zh-de":0.3711, "zh-fr":0.3540, "zh-it":0.3730, "zh-pt":0.3596, "zh-es":0.3389}

    data = []
    # make bilingual data
    for pair in bili_ratio.keys():
        size = (need[pair]/bili_ratio[pair]) * 1024 * 1024 * 1024
        print(f"bilingual, {pair}, sample size: {size/( 1024 * 1024)} M")
        src_lang, tgt_lang = pair.split("-")
        acc_size = 0
        acc_n = 0
        for line in tqdm.tqdm(open(f"./bili_data/{pair}.jsonl")):
            ele = json.loads(line)
            acc_size += len((ele["translation"][src_lang] + ele["translation"][tgt_lang]).encode('utf-8'))
            if acc_size < size:
                if random.choice([0,1]) == 0:
                    direction = f"<{src_lang}2{tgt_lang}>"
                    text = prompt.format(src=ele["translation"][src_lang], direc=direction, tgt=ele["translation"][tgt_lang])
                else:
                    direction = f"<{tgt_lang}2{src_lang}>"
                    text = prompt.format(src=ele["translation"][tgt_lang], direc=direction, tgt=ele["translation"][src_lang])
                sample= {"text":text}
                data.append(json.dumps(sample, ensure_ascii=False) + '\n')
                acc_n += 1
            else:
                break
        print(f"{pair}, sample line: {acc_n}\n")
    print(f"all bilingual: {len(data)}\n")
    return data


def make_train_data():
    # 需要单语B tokens
    need_mono = {"zh":1, "en":1, "ja":0.5, "ko":0.5, "ru":0.5, "de":0.5, "fr":0.5, "it":0.5, "pt":0.5, "es":0.5}
    # 需要双语B tokens
    need_bili = {"zh-en":0.5, "zh-ja":0.5, "zh-ko":0.5, "zh-ru":0.5, "zh-de":0.5, "zh-fr":0.5, "zh-it":0.5, "zh-pt":0.5, "zh-es":0.5}
    
    out_url = "./cpt_10b"
    os.makedirs(out_url, exist_ok=True)

    all_data = sample_mono_data(need_mono) + sample_bili_data(need_bili)
    random.shuffle(all_data)

    chunk_size = 10000000
    # chunk_size = 1000000
    num_files = (len(all_data) + chunk_size - 1) // chunk_size
    
    for i in range(num_files):
        filename = f"train{i+1}.jsonl"
        start_index = i * chunk_size
        end_index = start_index + chunk_size
        chunk_data = all_data[start_index:end_index]

        # 将数据写入JSONL文件
        with open(f"{out_url}/{filename}", 'w') as file:
            for item in chunk_data:
                file.write(item)

        print(f"Saved {filename} with {len(chunk_data)} records")


def make_valid_data():
    ## 双语+单语混合的验证集，其中双语每条数据同时制作正向反向
    out_url = "./cpt_10b"
   
    data = []
    mono_langs = ["zh", "en", "ja", "ko", "ru", "de", "fr", "it", "pt", "es"]
    bili_langs = ["zh-en", "zh-ja", "zh-ko", "zh-ru", "zh-de", "zh-fr", "zh-it", "zh-pt", "zh-es"]

    data += [line  for lang in mono_langs for line in open(f"mono_data/valid.{lang}.jsonl")]
    for pair in bili_langs:
        for line in open(f"./bili_data/valid.{pair}.jsonl"):
            ele = json.loads(line)
            src_lang, tgt_lang = pair.split("-")
            text1 = prompt.format(src=ele["translation"][src_lang], direc=f"<{src_lang}2{tgt_lang}>", tgt=ele["translation"][tgt_lang])
            text2 = prompt.format(src=ele["translation"][tgt_lang], direc=f"<{tgt_lang}2{src_lang}>", tgt=ele["translation"][src_lang])
            sample1 = {"text":text1}
            sample2 = {"text":text2}
            data.append(json.dumps(sample1, ensure_ascii=False) + '\n')
            data.append(json.dumps(sample2, ensure_ascii=False) + '\n')
    print(f"all valid: {len(data)}")
    f_out = open(f"{out_url}/valid.jsonl", mode='w')
    f_out.write("".join(data))


trans_prompt = "将以下内容从{src_lang}翻译到{tgt_lang}："
lan_dict = {"zh": "中文", "de": "德语", "en": "英语", "es": "西班牙语", "fr": "法语", "it": "意大利语", "ja": "日语", "ko": "韩语", "pt": "葡萄牙语", "ru": "俄语", }


def make_test_data():
    src_lang = "zh"
    tgt_langs = ["en", "ja", "ko", "ru", "de", "fr", "it", "pt", "es"]
    
    in_url = "/mnt/luoyingfeng/llm-zh-mt/data/flores200"
    out_url = "/mnt/luoyingfeng/llm-zh-mt/data/sft_data"

    # 遍历每种语言对
    for tgt_lang in tgt_langs:
        print(f"{src_lang}-{tgt_lang}")
        # 文件路径
        file_path = f"{in_url}/zh-{tgt_lang}/test.zh-{tgt_lang}.json"

        zh2x_data = []
        x2zh_data = []

        for line in json.load(open(file_path)):
            translation = line["translation"]
            src_text = translation[src_lang]
            tgt_text = translation[tgt_lang]
            sample_zh2x = {"instruction": trans_prompt.format(src_lang=lan_dict["zh"], tgt_lang=lan_dict[tgt_lang]) + src_text, "input": "", "output": tgt_text}
            sample_x2zh = {"instruction": trans_prompt.format(src_lang=lan_dict[tgt_lang], tgt_lang=lan_dict["zh"]) + tgt_text, "input": "", "output": src_text}

            zh2x_data.append(sample_zh2x)
            x2zh_data.append(sample_x2zh)

        with open(f"{out_url}/test.{src_lang}2{tgt_lang}.jsonl", "w") as f1, open(f"{out_url}/test.{tgt_lang}2{src_lang}.jsonl", "w") as f2:
            for ele1, ele2 in zip(zh2x_data, x2zh_data):
                f1.write(json.dumps(ele1, ensure_ascii=False) + "\n")
                f2.write(json.dumps(ele2, ensure_ascii=False) + "\n")
        
    print("Done!")


def make_sft_valid_data():
    in_url = "./flores200"
    out_url = "./sft_data"

    data = []
    bili_langs = ["zh-en", "zh-ja", "zh-ko", "zh-ru", "zh-de", "zh-fr", "zh-it", "zh-pt", "zh-es"]
    for pair in bili_langs:
        for line in json.load(open(f"{in_url}/{pair}/valid.json")):
            src_lang, tgt_lang = pair.split("-")
            translation = line["translation"]
            src_text = translation[src_lang]
            tgt_text = translation[tgt_lang]
            sample_zh2x = {"instruction": trans_prompt.format(src_lang=lan_dict[src_lang], tgt_lang=lan_dict[tgt_lang]) + src_text, "input": "", "output": tgt_text}
            sample_x2zh = {"instruction": trans_prompt.format(src_lang=lan_dict[tgt_lang], tgt_lang=lan_dict[src_lang]) + tgt_text, "input": "", "output": src_text}
            data.append(json.dumps(sample_zh2x, ensure_ascii=False) + '\n')
            data.append(json.dumps(sample_x2zh, ensure_ascii=False) + '\n')
    with open(f"{out_url}/valid.jsonl", mode='w') as f:
        f.write("".join(data))


def make_sft_train_data():
    in_url = "./sft_100k"
    out_url = "./sft_data"

    data = []
    bili_langs = ["zh2en", "zh2ja", "zh2ko", "zh2ru", "zh2de", "zh2fr", "zh2it", "zh2pt", "zh2es"]
    for pair in bili_langs:
        for line in open(f"{in_url}/{pair}"):
            src_lang, tgt_lang = pair.split("2")
            item = line.strip().split("\t")
            if len(item) == 2 and item[0].strip() and item[1].strip():
                src_text = item[0].strip()
                tgt_text = item[1].strip()
                sample_zh2x = {"instruction": trans_prompt.format(src_lang=lan_dict[src_lang], tgt_lang=lan_dict[tgt_lang]) + src_text, "input": "", "output": tgt_text}
                sample_x2zh = {"instruction": trans_prompt.format(src_lang=lan_dict[tgt_lang], tgt_lang=lan_dict[src_lang]) + tgt_text, "input": "", "output": src_text}
                data.append(json.dumps(sample_zh2x, ensure_ascii=False) + '\n')
                data.append(json.dumps(sample_x2zh, ensure_ascii=False) + '\n')
    random.shuffle(data)
    with open(f"{out_url}/train.jsonl", mode='w') as f:
        f.write("".join(data))


if __name__ == "__main__":
    # merge_valid()
    # make_train_data()
    # make_valid_data()
    # ## sft
    # make_test_data()
    # make_sft_valid_data()
    # make_sft_train_data()
    for i,line in enumerate(open("/mnt/luoyingfeng/llm_evaluation/data/alexandrainst/m_hellaswag/data/zh/val.jsonl")):
        try:
            ele = json.loads(line)
            for item in ele["endings"]:
                if type(item) is not str:
                    print(i, ele)
        except:
            print(i, line)
