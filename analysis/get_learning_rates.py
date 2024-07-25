import json
import pandas as pd
import sys

# file_path = '/maas-us/notebook/users/yang-2ewu0520/less_result/out/llama2-7b-less-dpo-shp-p0.05-lora-seed200-ckpt1108/trainer_state.json'

file_path = sys.argv[1]
# 打开并读取 JSON 文件
with open(file_path, 'r', encoding='utf-8') as file:
    data = json.load(file)

# 将数据转换为 DataFrame
log_history = data['log_history']
df = pd.DataFrame(log_history)

# 自定义分组
bins = [0, 1, 2, 3, 4]  # 定义分组区间
labels = ['0-1', '1-2', '2-3', '3-4']  # 定义分组标签

df['epoch_group'] = pd.cut(df['epoch'], bins=bins, labels=labels, right=True)

# 按分组计算平均值
grouped_df = df.groupby('epoch_group').mean()
learning_rate = grouped_df["learning_rate"].tolist()

# 打印结果
# print(learning_rate)

for lr in learning_rate:
    print(lr)

