from transformers import BertModel

import torch
import numpy as np

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 切换预训练模型，还要记得修改最后一层的输出特征数
pretrain_model = BertModel.from_pretrained("bert-base-chinese").to(DEVICE)  # 768
# pretrain_model = BertModel.from_pretrained("yechen/bert-large-chinese").to(DEVICE)  # 1024
# print(pretrain_model)



# 根据所输出的特征定义下游任务
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc = torch.nn.Linear(768, 8)

    # input_ids 就是编码后的词  # attention_mask pad的位置是0,其他位置是1
    # token_type_ids 第一个句子和特殊符号的位置是0,第二个句子和特殊符号的位置是1.两个句子段。
    def forward(self, input_ids, attention_mask, token_type_ids):
        with torch.no_grad():
            out = pretrain_model(input_ids=input_ids, attention_mask=attention_mask,
                                 token_type_ids=token_type_ids)

        # print(out.last_hidden_state)
        # print(np.shape(out.last_hidden_state))  # torch.Size([32, 500, 768])
        # print('----')
        # print(out.last_hidden_state[:, 0])  # torch.Size([32, 768])
        # print(np.shape(out.last_hidden_state[:, 0]))
        out = self.fc(out.last_hidden_state[:, 0])
        out = out.softmax(dim=1)

        return out





