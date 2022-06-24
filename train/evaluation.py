# -*- coding: utf-8 -*-
# @Time    : 6/23/22 11:36 AM
# @Author  : LIANYONGXING
# @FileName: evaluation.py
# @Software: PyCharm
# @Repo    : https://github.com/lianyongxing/text-classification-nlp-pytorch
from datasets import load_metric
from sklearn.metrics import accuracy_score
import torch


def eval(model, dev_loader):
    model.eval()

    dev_metric = load_metric('f1')
    val_true = []
    val_pred = []
    for idx, batch in enumerate(dev_loader):
        inputs = {k: v for k, v in batch.items() if k != 'label'}
        y = batch['label']
        y_pred = model(**inputs)
        y_pred_lab = torch.argmax(y_pred, dim=-1).detach().cpu().numpy().tolist()
        val_true.extend(y)
        val_pred.extend(y_pred_lab)
        dev_metric.add_batch(predictions=y_pred_lab, references=y)
    dev_f1 = dev_metric.compute(average="macro")['f1']
    return accuracy_score(val_true, val_pred), dev_f1
