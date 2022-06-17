# -*- coding: utf-8 -*-
# @Time    : 6/17/22 11:18 AM
# @Author  : LIANYONGXING
# @FileName: text_preprocessing.py
# @Software: PyCharm
# @Repo    : https://github.com/lianyongxing/text-classification-nlp-pytorch

import re
import zhconv
from resources.emoji.emoji import emojis

def text_filter(text):
    text = text.strip()
    text = re.sub(r'@[\w.?!,]+', '', text)
    for k in emojis.keys():
        text = text.replace(k, emojis[k])

    text = text.replace("\xa0", '')  # 去除\xa0
    text = re.sub(r'[^a-zA-Z0-9\u4e00-\u9fa5]', '', text).lower()  # 去除除了汉字大小写和数字
    text = zhconv.convert(text, 'zh-cn')  # 繁体转简体
    return text



if __name__ == "__main__":
    res = text_filter("nisacnsaic你好啊撒sacsacas121擦撒撒sss")
    print(res)