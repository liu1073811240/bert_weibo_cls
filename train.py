import torch
from MyData import MyDataset
from torch.utils.data import DataLoader
from net import Model
from transformers import AdamW, BertTokenizer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCH = 30000

token = BertTokenizer.from_pretrained("bert-base-chinese")
# token = BertTokenizer.from_pretrained("yechen/bert-large-chinese")


# 在代码中制定GPU
# import os
# os.environ['CUDA_VISIBLE_DEVICE'] = 1

# 在命令终端指定：CUDA_VISIBLE_DEVICES=1 python train.py


def collate_fn(data):
    sentes = [i[0] for i in data]
    label = [i[1] for i in data]

    # 编码
    data = token.batch_encode_plus(batch_text_or_text_pairs=sentes,
                                   truncation=True,
                                   padding="max_length",
                                   max_length=500,
                                   return_tensors="pt",
                                   return_length=True)

    input_ids = data["input_ids"]
    attention_mask = data["attention_mask"]
    token_type_ids = data["token_type_ids"]
    labels = torch.LongTensor(label)

    return input_ids, attention_mask, token_type_ids, labels


# 创建数据集
train_dataset = MyDataset("train")
val_dataset = MyDataset("validation")

# 创建dataloader
train_loader = DataLoader(dataset=train_dataset,
                          batch_size=128,
                          shuffle=True,
                          drop_last=True,
                          collate_fn=collate_fn)

val_loader = DataLoader(dataset=val_dataset,
                        batch_size=32,
                        shuffle=True,
                        drop_last=True,
                        collate_fn=collate_fn)

if __name__ == '__main__':
    # 开始训练
    print(DEVICE)
    model = Model().to(DEVICE)
    optimizer = AdamW(model.parameters())
    loss_func = torch.nn.CrossEntropyLoss()  # 交叉熵损失函数 自带 one-hot编码

    model.train()
    for epoch in range(EPOCH):
        sum_val_loss = 0
        sum_val_acc = 0

        # 训练
        for i, (input_ids, attention_mask, token_type_ids, labels) in enumerate(train_loader):
            input_ids, attention_mask, token_type_ids, labels = input_ids.to(DEVICE), \
                                                                attention_mask.to(DEVICE), \
                                                                token_type_ids.to(DEVICE), \
                                                                labels.to(DEVICE)
            out = model(input_ids, attention_mask, token_type_ids)

            loss = loss_func(out, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 5:
                out = out.argmax(dim=1)
                # acc = (out == labels).sum() / len(labels)
                acc = (out == labels).sum().item() / len(labels)
                print(epoch, i, loss.item(), acc)

        # torch.save(model.state_dict(), f"params/{epoch}-bert-weibo.pth")
        # 模型验证
        for i, (input_ids, attention_mask, token_type_ids, labels) in enumerate(val_loader):
            input_ids, attention_mask, token_type_ids, labels = input_ids.to(DEVICE), \
                                                                attention_mask.to(DEVICE), \
                                                                token_type_ids.to(DEVICE), \
                                                                labels.to(DEVICE)
            out = model(input_ids, attention_mask, token_type_ids)

            loss = loss_func(out, labels)

            out = out.argmax(dim=1)
            acc = (out == labels).sum() / len(labels)

            sum_val_loss += loss
            sum_val_acc += acc

        avg_val_loss = sum_val_loss / len(val_loader)
        avg_val_acc = sum_val_acc / len(val_loader)
        print(f"val==>epoch:{epoch}, avg_val_loss:{avg_val_loss}, avg_val_acc:{avg_val_acc}")

        torch.save(model.state_dict(), f"params/{epoch}-bert-weibo.pth")
        print(epoch, "参数保存成功！")



