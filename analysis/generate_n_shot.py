import datasets
import json
import random
from collections import defaultdict
from collections import defaultdict
import datasets
from typing import Dict, List, Optional, Iterator, Callable, Union, Tuple
import tqdm
from collections import defaultdict
import json
import pandas as pd
def save_to_json(data: List[Dict], filename: str):
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)

n_shot = 5

shot_data = []
count=0

shp_cat = [
    'askacademia', 'askanthropology', 'askbaking', 'askcarguys', 'askculinary',
    'askdocs', 'askengineers', 'askhistorians', 'askhr', 'askphilosophy',
    'askphysics', 'askscience', 'asksciencefiction', 'asksocialscience',
    'askvet', 'changemyview', 'explainlikeimfive', 'legaladvice'
]
for method in ["preference","sft"]:
    if method=="sft":
        for split in ["val","test"]:
            if split=="val":
                shot_data = []
                for cate in shp_cat:
                    path = f"../data/eval/shp_all/dev/shp_{cate}_sft_val.json"
                    with open(path, 'r') as file:
                        data = json.load(file)
                    
                    
                    if not isinstance(data, dict):
                        random_keys = range(min(5,len(data)))
                        for key in random_keys:
                            instance = data[key]
                            instance['id'] = f'shp_{count}'
                            shot_data.append(instance)
                            count += 1
                    else:
                        keys = list(data.keys())
                        random_keys = random.sample(keys, min(n_shot, len(keys)))
                        shot_data.extend({key: data[key]} for key in random_keys)
                print(f"Total combined data length: {len(shot_data)}")
                # 对于sft数据，直接保存即可
                save_to_json(shot_data,filename=f"../data/eval/shp_all/dev/shp_all_{n_shot}_shot_sft_val.json")
            else:
                shot_data = []
                for cate in shp_cat:
                    path = f"../data/eval/shp_all/test/shp_{cate}_sft_test.json"
                    with open(path, 'r') as file:
                        data = json.load(file)
                    
                    
                    if not isinstance(data, dict):
                        random_keys = range(min(5,len(data)))
                        for key in random_keys:
                            instance = data[key]
                            instance['id'] = f'shp_{count}'
                            shot_data.append(instance)
                            count += 1
                    else:
                        keys = list(data.keys())
                        random_keys = random.sample(keys, min(n_shot, len(keys)))
                        shot_data.extend({key: data[key]} for key in random_keys)
                print(f"Total combined data length: {len(shot_data)}")
                # 对于sft数据，直接保存即可
                save_to_json(shot_data,filename=f"../data/eval/shp_all/test/shp_all_{n_shot}_shot_sft_test.json")
    if method=="preference":
        for split in ["val","test"]:
            if split=="val":
                shot_data = []
                for cate in shp_cat:
                    path = f"../data/eval/shp_all/dev/shp_{cate}_preference_val.json"
                    with open(path, 'r') as file:
                        data = json.load(file)
                    
                    
                    if not isinstance(data, dict):
                        random_keys = range(min(5,len(data)))
                        for key in random_keys:
                            instance = data[key]
                            instance['id'] = f'shp_{count}'
                            shot_data.append(instance)
                            count += 1
                    else:
                        keys = list(data.keys())
                        random_keys = random.sample(keys, min(n_shot, len(keys)))
                        shot_data.extend({key: data[key]} for key in random_keys)
                print(f"Total combined data length: {len(shot_data)}")
                combined_data = defaultdict(list)
   
                for item in shot_data:
                    for key, value in item.items():
                        combined_data[key].append(value)
                # 转换回普通字典
                combined_data = dict(combined_data)
                for key,value in combined_data.items():
                    combined_data[key] = value[0]

                save_to_json(combined_data,filename=f"../data/eval/shp_all/dev/shp_all_{n_shot}_shot_preference_val.json")
            else:
                shot_data = []
                for cate in shp_cat:
                    path = f"../data/eval/shp_all/test/shp_{cate}_preference_test.json"
                    with open(path, 'r') as file:
                        data = json.load(file)
                    
                    
                    if not isinstance(data, dict):
                        random_keys = range(min(5,len(data)))
                        for key in random_keys:
                            instance = data[key]
                            instance['id'] = f'shp_{count}'
                            shot_data.append(instance)
                            count += 1
                    else:
                        keys = list(data.keys())
                        random_keys = random.sample(keys, min(n_shot, len(keys)))
                        shot_data.extend({key: data[key]} for key in random_keys)
                print(f"Total combined data length: {len(shot_data)}")

                combined_data = defaultdict(list)
                for item in shot_data:
                    for key, value in item.items():
                        combined_data[key].append(value)
                # 转换回普通字典
                combined_data = dict(combined_data)
                for key,value in combined_data.items():
                    combined_data[key] = value[0]
                save_to_json(combined_data,filename=f"../data/eval/shp_all/test/shp_all_{n_shot}_shot_preference_test.json")