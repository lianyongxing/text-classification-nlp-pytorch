# -*- coding: utf-8 -*-
# @Time    : 7/14/22 8:25 PM
# @Author  : LIANYONGXING
# @FileName: embedding_whitening.py
# @Software: PyCharm
# @Repo    : https://github.com/lianyongxing/text-classification-nlp-pytorch
import numpy as np

# ****************** 苏神代码 https://kexue.fm/archives/8069 ****************** #
def compute_kernel_bias(vecs):
    """计算kernel和bias
    vecs.shape = [num_samples, embedding_size]，
    最后的变换：y = (x + bias).dot(kernel)
    """
    mu = vecs.mean(axis=0, keepdims=True)
    cov = np.cov(vecs.T)
    u, s, vh = np.linalg.svd(cov)
    W = np.dot(u, np.diag(1 / np.sqrt(s)))
    return W, -mu


def compute_kernel_bias2(vecs, n_components=256):
    """计算kernel和bias
    vecs.shape = [num_samples, embedding_size]，
    最后的变换：y = (x + bias).dot(kernel)
    """
    mu = vecs.mean(axis=0, keepdims=True)
    cov = np.cov(vecs.T)
    u, s, vh = np.linalg.svd(cov)
    W = np.dot(u, np.diag(1 / np.sqrt(s)))
    return W[:, :n_components], -mu


# 应用变换，然后标准化
def transform_and_normalize(vecs, kernel, bias):
    if not (kernel is None or bias is None):
        vecs = (vecs + bias).dot(kernel)
    return vecs / (vecs**2).sum(axis=1, keepdims=True)**0.5