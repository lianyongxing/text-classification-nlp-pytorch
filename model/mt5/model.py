# -*- coding: utf-8 -*-
# @Time    : 6/20/22 11:20 AM
# @Author  : LIANYONGXING
# @FileName: model.py
# @Software: PyCharm
# @Repo    : https://github.com/lianyongxing/text-classification-nlp-pytorch
import torch
import torch.nn as nn
from transformers import MT5EncoderModel, MT5Config, MT5Tokenizer
from utils.text_preprocessing import text_filter
import torch.nn.functional as F


class Pooler(nn.Module):

    def __init__(self, config):
        super(Pooler, self).__init__()
        self.dense = nn.Linear(config.d_model, config.d_model)
        self.activate = nn.Tanh()

    def forward(self, x):
        first_token = x[:, 0]
        pooled_output = self.dense(first_token)
        pooled_output = self.activate(pooled_output)
        return pooled_output


class MT5(nn.Module):

    def __init__(self, classes=2, mt5_path="google/mt5-small"):
        super(MT5, self).__init__()
        self.max_len = 128
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.config = MT5Config.from_pretrained(mt5_path)
        self.mt5 = MT5EncoderModel.from_pretrained(mt5_path)
        self.tokenizer = MT5Tokenizer.from_pretrained(mt5_path, local_files_only=True)
        self.pooler = Pooler(config=self.config)
        self.fc = nn.Linear(self.config.d_model, classes)

    def forward(self, x):
        encoder_out = self.mt5(x)
        pooled_out = self.pooler(encoder_out.last_hidden_state)
        out = self.fc(pooled_out)
        return out

    def predict(self, raw_text):
        text = text_filter(raw_text)
        if text == "":
            return 0, 0
        input_ids = []
        encode_dict = self.tokenizer.encode_plus(text, max_length=self.max_len, truncation=True, padding='max_length')
        input_ids.append(encode_dict['input_ids'])
        logits = self.forward(torch.LongTensor(input_ids).to(self.DEVICE))
        y_pred_res = torch.argmax(logits, dim=1).detach().cpu().numpy().tolist()[0]
        y_pred_prob = F.softmax(logits, dim=1).detach().cpu().numpy()[0][1]
        return y_pred_res, y_pred_prob


if __name__ == "__main__":
    model = MT5()
    text1 = '今天天气真的很不错'
