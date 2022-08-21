from torch.utils.data import Dataset
from datasets import load_dataset

class MyDataset(Dataset):
    def __init__(self, split):
        # 从磁盘中加载csv数据
        self.dataset = load_dataset(path="csv", data_files=f"data/Weibo/{split}.csv", split="train")  # 原始数据集 split划分的都是train

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        text = self.dataset[item]["text"]
        label = self.dataset[item]["label"]

        return text, label

if __name__ == '__main__':
    dataset = MyDataset("train")

    for i, (text, label) in enumerate(dataset):
        print(i)
        print(text)
        print(label)
        print()








