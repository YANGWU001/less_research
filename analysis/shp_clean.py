from collections import defaultdict
import datasets
from typing import Dict, List, Optional, Iterator, Callable, Union, Tuple
import tqdm
import json
def get_shp(split: str, silent: bool = False, cache_dir: str = None) -> Dict[str, Dict[str, Union[List[Tuple[int, int]], List[str], str]]]:
    """Load the Stanford Human Preferences dataset from Huggingface and convert it to the necessary format. See hh for the format.

       We filter preference pairs to only keep pairs where the score ratio is at least 2.
       For this dataset, the sft_target is the response with the highest score.
    """
    print(f'Loading SHP dataset ({split} split) from Huggingface...')
    dataset = datasets.load_dataset('stanfordnlp/SHP', data_dir="askbaking",split=split, cache_dir=cache_dir)
    print('done')

    data = defaultdict(lambda: defaultdict(list))
    for row in tqdm.tqdm(dataset, desc='Processing SHP', disable=silent):
        prompt = '\n\nHuman: ' + row['history'] + '\n\nAssistant:'
        responses = [' ' + row['human_ref_A'], ' ' + row['human_ref_B']]
        scores = [row['score_A'], row['score_B']]
        if prompt in data:
            n_responses = len(data[prompt]['responses'])
        else:
            n_responses = 0
        score_ratio = max(scores[0] / scores[1], scores[1] / scores[0])
        if score_ratio < 3:
            continue

        # according to https://huggingface.co/datasets/stanfordnlp/SHP
        data[prompt]['pairs'].append((n_responses, n_responses + 1) if row['labels'] == 1 else (n_responses + 1, n_responses))
        data[prompt]['responses'].extend(responses)
        data[prompt]['scores'].extend(scores)

    for prompt in data:
        data[prompt]['sft_target'] = max(data[prompt]['responses'], key=lambda x: data[prompt]['scores'][data[prompt]['responses'].index(x)])
        data[prompt]['rejected_target'] = min(data[prompt]['responses'], key=lambda x: data[prompt]['scores'][data[prompt]['responses'].index(x)])
        # del data[prompt]['scores']

    return data

# preference data
# validation = get_shp("validation")


def transform_sft_data(data: Dict[str, Dict[str, Union[List[Tuple[int, int]], List[str], str]]], dataset_name: str) -> List[Dict]:
    transformed_data = []
    count = 0
    
    for prompt, details in data.items():
        entry = {
            'dataset': dataset_name,
            'id': f'{dataset_name}_{count}',
            'messages': [
                {
                    'role': 'user',
                    'content': prompt
                },
                {
                    'role': 'assistant',
                    'content': details['sft_target']
                }
            ]
        }
        transformed_data.append(entry)
        count += 1

    return transformed_data


# shp_sft_val = transform_sft_data(validation, "shp")
# sft data


def save_to_json(data: List[Dict], filename: str):
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)


# save_to_json(shp_sft_test,filename="shp_sft_test.json")


# processed_datasets = datasets.load_dataset(
#     "json",
#     data_files=["shp_val.json"],
# )["train"]

