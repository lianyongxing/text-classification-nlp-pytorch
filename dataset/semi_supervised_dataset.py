# -*- coding: utf-8 -*-
# @Time    : 6/22/22 3:06 PM
# @Author  : LIANYONGXING
# @FileName: semi_supervised_dataset.py
# @Software: PyCharm
# @Repo    : https://github.com/lianyongxing/text-classification-nlp-pytorch

from torch.utils.data import Dataset, DataLoader
import torch
from transformers import BertTokenizer


class PiModelDataset(Dataset):

    def __init__(self, encodings, labs=None, supervised=True, dev=False):

        self.encodings = encodings
        self.labs = labs
        self.supervised = supervised
        self.dev = dev

    def __getitem__(self, idx):

        item1 = {key: torch.LongTensor(val[idx]) for key, val in self.encodings.items()}
        item2 = item1.copy()

        if self.dev:
            item1['label'] = torch.LongTensor([float(i) for i in self.labs])[idx]
            return item1

        if self.supervised:
            item_label = torch.LongTensor([float(i) for i in self.labs])[idx]
            return item1, item2, item_label
        else:
            return item1, item2


    def __len__(self):

        if 'input_ids' in self.encodings:
            return len(self.encodings['input_ids'])
        else:
            return len(self.encodings)


if __name__ == "__main__":

    # dataset && loader test code

    dats = [i.strip().split('\t') for i in open('../resources/data/THUCNews/data/train3.txt')][:500]
    texts = [str(i[0]) for i in dats][:100]
    train_labs = [i[1] for i in dats][:100]

    unlab_texts = [str(i[0]) for i in dats][100:500]

    batch_size = 4

    tokenizer = BertTokenizer.from_pretrained('../resources/chinese_bert')
    train_encodings = tokenizer(texts, max_length=30, padding='max_length', truncation=True)
    train_dataset = PiModelDataset(train_encodings, train_labs)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    unlabeled_train_encodings = tokenizer(unlab_texts, max_length=30, padding='max_length', truncation=True)
    unlabeled_train_dataset = PiModelDataset(unlabeled_train_encodings, supervised=False)
    unlabeled_train_dataloader = DataLoader(unlabeled_train_dataset, batch_size=batch_size*4, shuffle=True)

    unlabeled_train_iter = iter(unlabeled_train_dataloader)
    for idx, batch in enumerate(train_dataloader):
        x1_labeled, x2_labeled, lab = batch

        try:
            x1_unlabeled, x2_unlabeled = next(unlabeled_train_iter)
        except StopIteration:
            # 如果用完数据，从头重新生成
            unlabeled_train_iter = iter(unlabeled_train_dataloader)
            x1_unlabeled, x2_unlabeled = next(unlabeled_train_iter)


        x1_cat = {k: torch.cat([v, x1_unlabeled[k]], dim=0) for k,v in x1_labeled.items()}
        x2_cat = {k: torch.cat([v, x2_unlabeled[k]], dim=0) for k,v in x2_labeled.items()}
