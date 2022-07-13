# -*- coding: utf-8 -*-
# @Time    : 7/13/22 11:55 AM
# @Author  : LIANYONGXING
# @FileName: train_mixup.py
# @Software: PyCharm
# @Repo    : https://github.com/lianyongxing/text-classification-nlp-pytorch
# -*- coding: utf-8 -*-
# @Time    : 6/20/22 8:15 PM
# @Author  : LIANYONGXING
# @FileName: train_norm.py
# @Software: PyCharm
# @Repo    : https://github.com/lianyongxing/text-classification-nlp-pytorch
import sys
sys.path.append(r'..')

import numpy as np
import time
import torch
from torch.utils.data import DataLoader
from model.bert.model import Bert
from dataset.bert_dataset import BertDataset
import torch.nn as nn
from transformers import AdamW, get_cosine_schedule_with_warmup
from tqdm import tqdm
from accelerate import Accelerator
from evaluation import eval


def get_perm(y):
    # batch乱序
    batch_size = y.size()[0]
    index = torch.randperm(batch_size)
    return index


def mixup_loss(y_pred, y, y_randidx, lam, criterion):
    # 计算mixup loss
    return lam * criterion(y_pred, y) + (1 - lam) * criterion(y_pred, y_randidx)


def train_mixup(model, train_loader, dev_loader, criterion, optimizer, scheduler, epoch, method='sentence'):

    best_f1 = 0
    for ep in tqdm(range(epoch)):

        start = time.time()
        train_loss_sum = 0

        for idx, batch in enumerate(train_loader):

            model.train()

            inputs = {k: v for k, v in batch.items() if k != 'label'}
            y = batch['label']

            # 构造乱序样本
            indexs = get_perm(y)
            inputs_randidx = {k: v[indexs] for k, v in batch.items() if k != 'label'}
            y_randidx = y[indexs]
            lam = np.random.beta(1, 1)

            optimizer.zero_grad()
            
            if method == 'sentence':
                y_pred = model.forward_sentence_mixup(inputs, inputs_randidx, lam)
            elif method == 'encoder':
                y_pred = model.forward_encoder_mixup(inputs, inputs_randidx, lam)
            else:
                y_pred = 0
                print('请选择mix-up方式')
                exit(-1)

            loss = mixup_loss(y_pred, y, y_randidx, lam, criterion)
            accelerator.backward(loss)
            # loss.backward()
            optimizer.step()
            scheduler.step()

            train_loss_sum += loss.item()

            if (idx+1) % (len(train_loader)//4) == 0:   # 一个Epoch打印四次
                print('Epoch {:04d} - Step {:04d}/{:04d} - Loss {:.4f} - Time {:.4f}'.format(
                    ep+1, idx+1, len(train_loader), train_loss_sum/(idx+1), time.time() - start
                ))

        dev_acc, dev_f1 = eval(model, dev_loader)

        print('Epoch {:04d} - Dev F1 {:.4f} - Dev Acc {:.4f}'.format(
            ep+1, dev_f1, dev_acc
        ))

        if dev_f1 > best_f1:
            best_f1 = dev_f1
            print('Epoch-%s, Update Model with new F1=%s' % (ep+1, best_f1))
            torch.save(model.state_dict(), '../output/best_model.pth')


if __name__ == "__main__":

    # define model
    bert_path = "../resources/chinese_bert"
    m = Bert(bert_path, classes=10)

    # define data
    dats = [i.strip().split('\t') for i in open('../resources/data/THUCNews/data/train3.txt')][:500]
    texts = [str(i[0]) for i in dats]
    train_labs = [i[1] for i in dats]

    batch_size = 128
    # DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_encodings = m.tokenizer(texts, max_length=30, padding='max_length', truncation=True)
    train_dataset = BertDataset(train_encodings, train_labs)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # # define training
    EPOCHS = 3

    accelerator = Accelerator()

    optimizer = AdamW(m.parameters(), lr=2e-5, weight_decay=1e-4)
    scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                num_warmup_steps=len(train_dataloader),
                                                num_training_steps=EPOCHS*len(train_dataloader))
    #
    m, optimizer, train_dataloader = accelerator.prepare(m, optimizer, train_dataloader)
    criterion = nn.CrossEntropyLoss()
    #

    mixup_method = 'encoder'

    train_mixup(m, train_dataloader, train_dataloader, criterion, optimizer, scheduler, EPOCHS, mixup_method)
