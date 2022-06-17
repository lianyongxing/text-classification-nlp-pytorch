# -*- coding: utf-8 -*-
# @Time    : 6/17/22 11:20 AM
# @Author  : LIANYONGXING
# @FileName: model.py
# @Software: PyCharm
# @Repo    : https://github.com/lianyongxing/text-classification-nlp-pytorch

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertConfig, BertTokenizer
from utils.text_preprocessing import text_filter
import warnings
warnings.filterwarnings('ignore')


# 定义 bert
class Bert(nn.Module):

    def __init__(self, bert_path, classes=2):
        super(Bert, self).__init__()
        self.bert_path = bert_path
        self.config = BertConfig.from_pretrained(self.bert_path, local_files_only=True)  # 导入模型超参数
        self.bert = BertModel.from_pretrained(self.bert_path, local_files_only=True)  # 加载预训练模型权重
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path, tokenize_chinese_chars=True, local_files_only=True)

        self.fc = nn.Linear(self.config.hidden_size, classes)  # 直接分类
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_len = 128

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.bert(input_ids, attention_mask, token_type_ids)
        out_pool = outputs[1]  # 池化后的输出 [bs, config.hidden_size]
        logit = self.fc(out_pool)  # [bs, classes]
        return logit

    def predict(self, raw_text):
        text = text_filter(raw_text)
        if text == "":
            return 0, 0
        input_ids, input_masks, input_types = [], [], []  # input char ids, segment type ids, attention mask  # 标签
        encode_dict = self.tokenizer.encode_plus(text, max_length=self.max_len, padding='max_length', truncation=True)
        input_ids.append(encode_dict['input_ids'])
        input_types.append(encode_dict['token_type_ids'])
        input_masks.append(encode_dict['attention_mask'])
        logits = self.forward(torch.LongTensor(input_ids).to(self.DEVICE),
                              torch.LongTensor(input_masks).to(self.DEVICE),
                              torch.LongTensor(input_types).to(self.DEVICE))
        y_pred_res = torch.argmax(logits, dim=1).detach().cpu().numpy().tolist()[0]
        y_pred_prob = F.softmax(logits, dim=1).detach().cpu().numpy()[0][1]
        return y_pred_res, y_pred_prob


if __name__ == "__main__":
    import random
    random.seed(2022)


    bert_path = "/Users/user/Desktop/git_projects/text-classification-nlp-pytorch/resources/chinese_bert"  # 该文件夹下存放三个文件（'vocab.txt', 'pytorch_model.bin', 'config.json'）
    model = Bert(bert_path)
    model.eval()
    result = model.predict("今天真是个好天气")
    print(result)