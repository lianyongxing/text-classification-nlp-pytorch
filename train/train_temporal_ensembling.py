# -*- coding: utf-8 -*-
# @Time    : 7/12/22 3:58 PM
# @Author  : LIANYONGXING
# @FileName: train_temporal_ensembling.py
# @Software: PyCharm
# @Repo    : https://github.com/lianyongxing/text-classification-nlp-pytorch
import sys
sys.path.append(r'..')

import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, AdamW, get_cosine_schedule_with_warmup
from dataset.bert_dataset import BertDataset
from model.bert.model import Bert
import numpy as np
import torch.nn.functional as F


def train_temporal_ensemble(model,ls

                            train_loader,
                            dev_loader,
                            optimizer,
                            scheduler,
                            n_sample,
                            epoch,
                            n_class=10,
                            alpha=0.6,
                            MAX_EPOCHS=3):
    
    print("******************** start training temporal-ensemble ********************")

    n_datas = len(train_loader.dataset)
    temp_z = torch.zeros(n_sample, n_class).float()
    Z = torch.zeros(n_sample, n_class).float()
    outputs = torch.zeros(n_sample, n_class).float()

    for ep in range(epoch):
        wt = get_weight_schedule(ep, MAX_EPOCHS, n_datas, n_sample)
        wt = torch.autograd.Variable(torch.FloatTensor([wt]), requires_grad=False)

        start = 0
        for i, batch in enumerate(train_loader):

            model.train()
            inputs = {k: v for k, v in batch.items() if k != 'label'}
            y = batch['label']

            optimizer.zero_grad()
            out = model(**inputs)
            end = start + len(y)
            outputs[start:end] = out.data.clone()
            final_loss, sup_loss, unsup_loss, nbsup = temporal_loss(out, temp_z[start:end], wt, y)
            start = end

            final_loss.backward()
            optimizer.step()
            scheduler.step()

        Z = alpha * Z + (1. - alpha) * outputs
        temp_z = Z * (1. / (1. - alpha ** (ep + 1)))

        evaluate(model, dev_loader)
        torch.save(model.state_dict(), '../output/best_temporal_model_ep%s.pth' % (ep+1))


def evaluate(model, dev_loader):
    pass


def ramp_up(epoch, max_epochs, max_val=30, mult=-5):
    if epoch == 0:
        return 0
    elif epoch >= max_epochs:
        return max_val
    return max_val * np.exp(mult * (1. - float(epoch) / max_epochs) ** 2)


def get_weight_schedule(epoch, max_epochs, n_datas, n_sample, max_val=30, mult=-5):
    max_val = max_val * (float(n_datas) / n_sample)
    return ramp_up(epoch, max_epochs, max_val, mult)


def temporal_loss(out, out2, w, y):

    def mse_loss(o1, o2):
        quad_diff = torch.sum((F.softmax(o1, dim=1) - F.softmax(o2, dim=1)) ** 2)
        return quad_diff / o1.data.nelement()

    def masked_crossentropy(o1, o2):
        nbsup = len(torch.nonzero(o2 >= 0))
        loss = F.cross_entropy(o1, o2, size_average=False, ignore_index=-1)
        if nbsup != 0:
            loss = loss / nbsup
        return loss, nbsup

    sup_loss, sup_num = masked_crossentropy(out, y)
    unsup_loss = mse_loss(out, out2)

    return sup_loss + w * unsup_loss, sup_loss, unsup_loss, sup_num


if __name__ == "__main__":

    # load model（bert）
    bert_path = "/Users/user/Desktop/git_projects/text-classification-nlp-pytorch/resources/chinese_bert"
    m = Bert(bert_path, classes=10)

    # define data
    dats = [i.strip().split('\t') for i in open('../resources/data/THUCNews/data/train3.txt')][:500]
    texts = [str(i[0]) for i in dats][:100]
    train_labs = [i[1] for i in dats][:100]

    unlab_texts = [str(i[0]) for i in dats][100:500]
    train_unlabs = ['-1'] * len(unlab_texts)  # 将未标注的数据label指定为-1

    all_train_texts = texts + unlab_texts
    all_train_labs = train_labs + train_unlabs

    batch_size = 4

    tokenizer = BertTokenizer.from_pretrained('../resources/chinese_bert', local_files_only=True)
    train_text_encodings = tokenizer(all_train_texts, max_length=30, padding='max_length', truncation=True)

    train_dataset = BertDataset(train_text_encodings, all_train_labs)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # mock dev data
    dev_dataset = BertDataset(train_text_encodings, all_train_labs)
    dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=True)

    # define training
    EPOCHS = 5
    MAX_EPOCHS = 3  # max weight schedule epochs
    alpha = 0.6
    classes = 10
    optimizer = AdamW(m.parameters(), lr=2e-5, weight_decay=1e-4)
    scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                num_warmup_steps=len(train_dataloader),
                                                num_training_steps=EPOCHS*len(train_dataloader))

    train_temporal_ensemble(m,
                            train_dataloader,
                            dev_dataloader,
                            optimizer,
                            scheduler,
                            n_sample=len(all_train_texts),
                            epoch=EPOCHS,
                            n_class=classes,
                            alpha=alpha,
                            MAX_EPOCHS=MAX_EPOCHS)
