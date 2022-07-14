# text-classification-nlp-pytorch

[![OSCS Status](https://www.oscs1024.com/platform/badge/lianyongxing/text-classification-nlp-pytorch.svg?size=small)](https://www.oscs1024.com/project/lianyongxing/text-classification-nlp-pytorch?ref=badge_small)

Text-classification-nlp-pytorch is used for exploring text classification methods with pytorch.

文本分类、半监督、小样本学习、样本增强
## 基础文本分类-Base

基础文本分类

### 基于Bert文本分类

#### Base Bert Classification
简单bert文本分类
```bash
cd train && python train_norm.py
```

#### Add Sentence/Encoder Embedding Mixup

添加Embedding Mixup后的文本分类
```bash
cd train && python train_mixup.py
```

## 半监督学习

### PI-Model
```bash
cd train && python train_pi_model.py
```

### Temporal-Ensemble
```bash
cd train && python train_temporal_ensembling.py
```

## 参考文献

<div id="refer-anchor-1"></div>
- [1] [Augmenting Data with Mixup for Sentence Classification: An Empirical
Study](https://arxiv.org/pdf/1905.08941.pdf)
<div id="refer-anchor-2"></div>
- [2] [Temporal Ensembling for Semi-Supervised Learning](https://arxiv.org/abs/1610.02242)
