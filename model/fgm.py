# -*- coding: utf-8 -*-
# @Time    : 6/24/22 4:06 PM
# @Author  : LIANYONGXING
# @FileName: fgm.py
# @Software: PyCharm
# @Repo    : https://github.com/lianyongxing/text-classification-nlp-pytorch
import torch
from model.bert.model import Bert


class FGM():

    def __init__(self, model, embedding_name='word_embeddings.', epsilon=1.0):
        self.model = model
        self.epsilon = epsilon
        self.embedding_name = embedding_name
        self.back_params = {}

    def attack(self):
        """
        对embedding添加扰动(根据梯度下降的反方向，epsilon控制扰动幅度)
        :return:
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.embedding_name in name:
                self.back_params[name] = param.data.clone()
                norm = torch.norm(param.grad)

                if norm != 0 and not torch.isnan(norm):
                    r_at = self.epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self):
        # 恢复正常参数
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.embedding_name in name:
                assert name in self.back_params
                param.data = self.back_params[name]
        self.back_params = {}


if __name__ == "__main__":

    # Example
    dataloader = [[(1,1,1), 1]]
    optimizer = None # 添加optimizer
    m = Bert('../resources/chinese_bert')

    fgm = FGM(m)
    for batch_input, batch_label in dataloader:
        # 正常训练
        loss = m(batch_input, batch_label)
        loss.backward() # 反向传播，得到正常的grad
        # 对抗训练
        fgm.attack() # 在embedding上添加对抗扰动
        loss_adv = m(batch_input, batch_label)
        loss_adv.backward() # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
        fgm.restore() # 恢复embedding参数
        # 梯度下降，更新参数
        optimizer.step()
        m.zero_grad()