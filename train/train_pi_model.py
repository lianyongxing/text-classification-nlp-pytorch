# -*- coding: utf-8 -*-
# @Time    : 6/23/22 10:53 AM
# @Author  : LIANYONGXING
# @FileName: train_pi_model.py
# @Software: PyCharm
# @Repo    : https://github.com/lianyongxing/text-classification-nlp-pytorch
from dataset.semi_supervised_dataset import PiModelDataset
from transformers import BertTokenizer, AdamW, get_cosine_schedule_with_warmup
from torch.utils.data import DataLoader
import torch
from model.bert.model import Bert
from tqdm import tqdm
import torch.nn.functional as F
import torch.nn as nn
import time
from evaluation import eval


def train_pi_model(model,
                   labeled_dataloader,
                   unlabeled_dataloader,
                   labeled_criterion,
                   unlabeled_criterion,
                   dev_dataloader,
                   optimizer,
                   scheduler,
                   epoch):

    print("******************** start training pi-model ********************")

    unlabeled_dataiter = iter(unlabeled_dataloader)

    best_f1 = 0
    for ep in tqdm(range(epoch)):

        start = time.time()
        train_loss_sum = 0
        for idx, batch in enumerate(labeled_dataloader):
            m.train()

            x1_labeled, x2_labeled, lab = batch

            try:
                x1_unlabeled, x2_unlabeled = next(unlabeled_dataiter)
            except StopIteration:
                # re-generate data
                unlabeled_train_iter = iter(unlabeled_dataloader)
                x1_unlabeled, x2_unlabeled = next(unlabeled_train_iter)

            x1_cat = {k: torch.cat([v, x1_unlabeled[k]], dim=0) for k, v in x1_labeled.items()}
            x2_cat = {k: torch.cat([v, x2_unlabeled[k]], dim=0) for k, v in x2_labeled.items()}

            # predict batch1
            logits1 = model(**x1_cat)

            # predict batch2 no grad
            with torch.no_grad():
                logits2 = model(**x2_cat)

            # 1. labeled(supervised) ce loss
            logits_labeled = logits1[:x1_labeled['input_ids'].size(0)]
            loss_labeled = labeled_criterion(logits_labeled, lab)

            # 2. unlabeled mse loss
            probs1 = F.softmax(logits1, dim=-1)
            probs2 = F.softmax(logits2, dim=-1)

            loss_unlabeled = unlabeled_criterion(probs1, probs2)

            loss = loss_labeled + alpha * loss_unlabeled

            loss.backward()
            optimizer.step()
            scheduler.step()

            train_loss_sum += loss.item()

            if (idx+1) % (len(labeled_dataloader)//4) == 0:   # 一个Epoch打印四次
                print('Epoch {:04d} - Step {:04d}/{:04d} - Loss {:.4f} - Time {:.4f}'.format(
                    ep+1, idx+1, len(labeled_dataloader), train_loss_sum/(idx+1), time.time() - start
                ))

        dev_acc, dev_f1 = eval(model, dev_dataloader)
        print('Epoch {:04d} - Dev F1 {:.4f} - Dev Acc {:.4f}'.format(
            ep+1, dev_f1, dev_acc
        ))

        if dev_f1 > best_f1:
            best_f1 = dev_f1
            print('Epoch-%s, Update Model with new F1=%s' % (ep+1, best_f1))
            torch.save(model.state_dict(), '../output/best_pi_model.pth')


if __name__ == "__main__":

    #  training test code

    bert_path = "/Users/user/Desktop/git_projects/text-classification-nlp-pytorch/resources/chinese_bert"
    m = Bert(bert_path, classes=10)

    dats = [i.strip().split('\t') for i in open('../resources/data/THUCNews/data/train3.txt')][:500]
    texts = [str(i[0]) for i in dats][:100]
    train_labs = [i[1] for i in dats][:100]

    unlab_texts = [str(i[0]) for i in dats][100:500]

    batch_size = 4

    # define data
    tokenizer = BertTokenizer.from_pretrained('../resources/chinese_bert', local_files_only=True)
    train_encodings = tokenizer(texts, max_length=30, padding='max_length', truncation=True)
    train_dataset = PiModelDataset(train_encodings, train_labs)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    unlabeled_train_encodings = tokenizer(unlab_texts, max_length=30, padding='max_length', truncation=True)
    unlabeled_train_dataset = PiModelDataset(unlabeled_train_encodings, supervised=False)
    unlabeled_train_dataloader = DataLoader(unlabeled_train_dataset, batch_size=batch_size*4, shuffle=True)
    unlabeled_train_iter = iter(unlabeled_train_dataloader)

    dev_dataset = PiModelDataset(train_encodings, train_labs, supervised=False, dev=True)
    dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=True)
    bb = next(iter(dev_dataloader))

    # define training
    EPOCHS = 1
    alpha = 0.1

    # add embedding perturbation
    m.config.hidden_dropout_prob = 0.3
    m.config.attention_probs_dropout_prob = 0.3

    sup_criterion = nn.CrossEntropyLoss()
    unsup_criterion = nn.MSELoss()

    optimizer = AdamW(m.parameters(), lr=2e-5, weight_decay=1e-4)
    scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                num_warmup_steps=len(train_dataloader),
                                                num_training_steps=EPOCHS*len(train_dataloader))

    train_pi_model(m, train_dataloader,
                   unlabeled_train_dataloader,
                   sup_criterion,
                   unsup_criterion,
                   dev_dataloader,
                   optimizer,
                   scheduler,
                   EPOCHS)