# -*- coding: utf-8 -*-
# @Time    : 6/20/22 2:14 PM
# @Author  : LIANYONGXING
# @FileName: bert_dataset.py
# @Software: PyCharm
# @Repo    : https://github.com/lianyongxing/text-classification-nlp-pytorch
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer


class BertDataset(Dataset):

    def __init__(self, encodings, labs):
        self.encodings = encodings
        self.labs = labs

    def __getitem__(self, idx):
        item = {key: torch.LongTensor(val[idx]) for key, val in self.encodings.items()}
        item['label'] = torch.LongTensor([float(i) for i in self.labs])[idx]
        return item

    def __len__(self):
        return len(self.labs)


if __name__ == '__main__':

    dats = [i.strip().split('\t') for i in open('../resources/data/THUCNews/data/train3.txt')]
    texts = [str(i[0]) for i in dats]
    train_labs = [i[1] for i in dats]

    batch_size = 128

    tokenizer = BertTokenizer.from_pretrained('../resources/chinese_bert')
    train_encodings = tokenizer(texts, max_length=30, padding='max_length', truncation=True)
    train_dataset = BertDataset(train_encodings, train_labs)
    train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)

    for idx, batch in enumerate(train_dataloader):
        inputs = {k: v.to('cpu') for k, v in batch.items() if k != 'label'}
        y = batch['label']
        print(len(y))
        break
