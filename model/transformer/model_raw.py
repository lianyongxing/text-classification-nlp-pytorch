# -*- coding: utf-8 -*-
# @Time    : 7/25/22 3:41 PM
# @Author  : LIANYONGXING
# @FileName: model_raw.py
# @Software: PyCharm
# @Repo    : https://github.com/lianyongxing/text-classification-nlp-pytorch

#########################################
#           手写 Transformer             #
#########################################

import torch
import torch.nn as nn
import numpy as np

d_model = 768
tgt_vocab_size = 2000
src_vocab = 1000
n_layers = 6

d_ff = 1024
d_k = 128
d_q = 128
d_v = 128
n_heads = 6
batch_size = 20


class PositionalEncoding(nn.Module):

    def __init__(self, d_model):
        super(PositionalEncoding, self).__init__()


class ScaleDotProductAttention(nn.Module):

    def __init__(self):
        super(ScaleDotProductAttention, self).__init__()

    def forward(self, q, k, v, mask):
        scores = torch.matmul(q, k.transpose(-1, -2)) / np.sqrt(d_k)
        scores.masked_fill_(mask, 1e-9)
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, v)
        return context, attn


class MultiheadAttention(nn.Module):

    def __init__(self):
        super(MultiheadAttention, self).__init__()
        self.w_q = nn.Linear(d_model, d_q * n_heads, bias=False)
        self.w_k = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.w_v = nn.Linear(d_model, d_v * n_heads, bias=False)
        self.fc = nn.Linear(d_v * n_heads, d_model, bias=False)

    def forward(self, q, k, v, mask):

        residual = q
        Q = self.w_q(q).view(batch_size, -1, n_heads, d_q).transpose(1, 2)
        K = self.w_k(k).view(batch_size, -1, n_heads, d_k).transpose(1, 2)
        V = self.w_q(v).view(batch_size, -1, n_heads, d_v).transpose(1, 2)

        attn_mask = mask.unsqueeze(1).repeat(1, n_heads, 1, 1)
        context, attn = ScaleDotProductAttention()(Q, K, V, attn_mask)
        context = context.transpose(1, 2).reshape(batch_size, -1, n_heads * d_v)
        output = self.fc(context)
        return nn.LayerNorm(d_model)(output + residual), attn


class PoswiseFeedForwardNet(nn.Module):

    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(d_ff, d_model, bias=False)
        )

    def forward(self, x):
        residual = x
        out = self.fc(x)

        return nn.LayerNorm(d_model)(out + residual)


class EncoderLayer(nn.Module):

    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiheadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()


    def forward(self, enc_inputs, enc_self_attn_mask):
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask)
        enc_outputs = self.pos_ffn(enc_outputs)
        return enc_outputs, attn


def get_attn_pad_mask(seq_q, seq_k):

    batch_size = seq_q.size(0)
    len_q, len_k = seq_q.size(1), seq_k.size(1)
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)
    return pad_attn_mask.expand(batch_size, len_q, len_k)


class Encoder(nn.Module):

    def __init__(self):
        super(Encoder, self).__init__()
        self.src_embedding = nn.Embedding(src_vocab, d_model)
        self.pos_embedding = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])

    def forward(self, enc_inputs):
        enc_outputs = self.src_embedding(enc_inputs)
        enc_outputs = self.pos_embedding(enc_outputs.transpose(0, 1)).transpose(0, 1)
        enc_self_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs)
        enc_self_attns = []
        for layer in self.layers:
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)
            enc_self_attns.append(enc_self_attn)
        return enc_outputs, enc_self_attns


class DecoderLayer(nn.Module):
    
    def __init__(self):
        super(DecoderLayer, self).__init__()
        self.dec_self_attn = MultiheadAttention()
        self.dec_enc_attn = MultiheadAttention()
        self.pos_fnn = PoswiseFeedForwardNet()

    def forward(self, dec_inputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask):
        dec_outputs, dec_self_attn = self.dec_self_attn(dec_inputs, dec_inputs, dec_inputs, dec_self_attn_mask)
        # dec_outputs: [batch_size, tgt_len, d_model], dec_enc_attn: [batch_size, h_heads, tgt_len, src_len]
        dec_outputs, dec_enc_attn = self.dec_enc_attn(dec_outputs, enc_outputs, enc_outputs, dec_enc_attn_mask)
        dec_outputs = self.pos_ffn(dec_outputs) # [batch_size, tgt_len, d_model]
        return dec_outputs, dec_self_attn, dec_enc_attn


def get_attn_subsequence_mask(dec_inputs):

    attn_shape = [dec_inputs.size(0), dec_inputs.size(1), dec_inputs.size(1)]
    sequence_mask = np.triu(np.ones(attn_shape), k=1)
    sequence_mask = torch.from_numpy(sequence_mask).byte()
    return sequence_mask


class Decoder(nn.Module):

    def __init__(self):
        super(Decoder, self).__init__()
        self.tgt_embeddings = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_embeddings = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([DecoderLayer() for _ in range(n_layers)])

    def forward(self, dec_inputs, enc_inputs, enc_outputs):
        dec_outputs = self.tgt_embeddings(dec_inputs)
        dec_outputs = self.pos_embeddings(dec_outputs.transpose(0, 1)).transpose(0, 1)
        dec_self_pad_mask = get_attn_pad_mask(dec_inputs, dec_inputs)
        dec_self_subsequence_mask = get_attn_subsequence_mask(dec_inputs)
        dec_self_attn_mask = torch.gt((dec_self_pad_mask + dec_self_subsequence_mask), 0)

        dec_enc_attn_mask = get_attn_pad_mask(dec_inputs, enc_inputs)
        dec_self_attns, dec_enc_attns = [], []

        for layer in self.layers:
            dec_outputs, dec_self_attn, dec_enc_attn = layer(dec_outputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask)
            dec_self_attns.append(dec_self_attn)
            dec_enc_attns.append(dec_enc_attn)
        return dec_outputs, dec_self_attns, dec_enc_attns

class Transformer(nn.Module):

    def __init__(self):
        super(Transformer, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.projection = nn.Linear(d_model, tgt_vocab_size, bias=False)

    def forward(self, enc_inputs, dec_inputs):
        enc_outputs, enc_self_attns = self.encoder(enc_inputs)
        dec_outputs, dec_self_attns, dec_enc_attns = self.decoder(dec_inputs, enc_inputs, enc_outputs)
        logits = self.projection(dec_outputs)
        return logits


if __name__ == "__main__":
    model = Transformer()
    print(model)