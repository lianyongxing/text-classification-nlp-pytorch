# -*- coding: utf-8 -*-
# @Time    : 6/22/22 3:05 PM
# @Author  : LIANYONGXING
# @FileName: train_pi_model.py
# @Software: PyCharm
# @Repo    : https://github.com/lianyongxing/text-classification-nlp-pytorch

class Pi_Model():

    def __init__(self, model, emb_name, epsilon=1.0):
        self.model = model
        self.emb_name = emb_name
        self.epsilon = epsilon
