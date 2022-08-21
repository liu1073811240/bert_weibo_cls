from datasets import load_dataset
from datasets import list_datasets, load_dataset, load_from_disk

# data = load_dataset(path="seamew/Weibo", split='train')  # 自行从网络下载数据
# print(data)

# 从本地加载数据
train_data = load_dataset(path="csv", data_files="data/Weibo/validation.csv", split='train')
print(train_data)  # 注意：原始数据集train/dev/test  标签没有写对分好。

# for data in train_data:
#     print(data)




