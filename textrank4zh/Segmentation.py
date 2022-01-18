#-*- encoding:utf-8 -*-
"""
@author:   letian
@homepage: http://www.letiantian.me
@github:   https://github.com/someus/
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import jieba.posseg as pseg
import codecs
import os
import re

from . import util

def get_default_stop_words_file():
    d = os.path.dirname(os.path.realpath(__file__))
    return os.path.join(d, 'stopwords.txt')

class WordSegmentation(object):
    """ 分词 """
    
    def __init__(self, stop_words_file = None, allow_speech_tags = util.allow_speech_tags):
        """
        Keyword arguments:
        stop_words_file    -- 保存停止词的文件路径，utf8编码，每行一个停止词。若不是str类型，则使用默认的停止词
        allow_speech_tags  -- 词性列表，用于过滤
        """     
        
        allow_speech_tags = [util.as_text(item) for item in allow_speech_tags]

        self.default_speech_tag_filter = allow_speech_tags
        self.stop_words = set()
        self.stop_words_file = get_default_stop_words_file()
        if type(stop_words_file) is str:
            self.stop_words_file = stop_words_file
        for word in codecs.open(self.stop_words_file, 'r', 'utf-8', 'ignore'):
            self.stop_words.add(word.strip())
    
    def segment(self, text, lower = True, use_stop_words = True, use_speech_tags_filter = False):
        """对一段文本进行分词，返回list类型的分词结果

        Keyword arguments:
        lower                  -- 是否将单词小写（针对英文）
        use_stop_words         -- 若为True，则利用停止词集合来过滤（去掉停止词）
        use_speech_tags_filter -- 是否基于词性进行过滤。若为True，则使用self.default_speech_tag_filter过滤。否则，不过滤。    
        """
        text = util.as_text(text)
        jieba_result = pseg.cut(text)
        
        if use_speech_tags_filter == True:
            jieba_result = [w for w in jieba_result if w.flag in self.default_speech_tag_filter]
        else:
            jieba_result = [w for w in jieba_result]

        # 去除特殊符号
        word_list = [w.word.strip() for w in jieba_result if w.flag!='x']
        word_list = [word for word in word_list if len(word)>0]
        
        if lower:
            word_list = [word.lower() for word in word_list]

        if use_stop_words:
            word_list = [word.strip() for word in word_list if word.strip() not in self.stop_words]

        return word_list
        
    def segment_sentences(self, sentences, lower=True, use_stop_words=True, use_speech_tags_filter=False):
        """将列表sequences中的每个元素/句子转换为由单词构成的列表。
        
        sequences -- 列表，每个元素是一个句子（字符串类型）
        """
        
        res = []
        for sentence in sentences:
            res.append(self.segment(text=sentence, 
                                    lower=lower, 
                                    use_stop_words=use_stop_words, 
                                    use_speech_tags_filter=use_speech_tags_filter))
        return res
        
class SentenceSegmentation(object):
    """ 分句 """
    
    def __init__(self, delimiters=util.sentence_delimiters):
        """
        Keyword arguments:
        delimiters -- 可迭代对象，用来拆分句子
        """
        # 2022.1.6-张好-加入了空格作为分隔符
        self.delimiters = set([util.as_text(item) for item in delimiters])

    # 2021.11.10-张好-改变了textrank原有的分句方式，保留了分句结尾的分隔符
    def segment(self, text):
        # res = [util.as_text(text)]
        res = util.as_text(text)

        util.debug(res)
        util.debug(self.delimiters)

        # 转化为原始字符串
        delimiters = '(['
        for d in self.delimiters:
            delimiters += d
        delimiters += '])'
        # delimiters = "%r"%delimiters
        # delimiters = delimiters[1: -1]

        res = re.split(delimiters, res)
        res = ["".join(i) for i in zip(res[0::2], res[1::2])]
        # for sep in self.delimiters:
        #     text, res = res, []
        #     for seq in text:
        #         split_result = seq.split(sep)
        #         res += split_result
        # res = [s.strip() for s in res if len(s.strip()) > 0]

        # 2021.1.6-张好-删除长度<=1的句子、开头小于5个字的句子、重复句子
        res = sorted(set(res), key=res.index)
        clean_res = []
        stop_delete_top_short = False
        for r in res:
            if not stop_delete_top_short and len(r) > 5:
                stop_delete_top_short = True
            if stop_delete_top_short:
                if len(r) > 1:
                    clean_res.append(r)
        return clean_res
        
class Segmentation(object):
    
    def __init__(self, stop_words_file = None, 
                    allow_speech_tags = util.allow_speech_tags,
                    delimiters = util.sentence_delimiters):
        """
        Keyword arguments:
        stop_words_file -- 停止词文件
        delimiters      -- 用来拆分句子的符号集合
        """
        self.ws = WordSegmentation(stop_words_file=stop_words_file, allow_speech_tags=allow_speech_tags)
        self.ss = SentenceSegmentation(delimiters=delimiters)
        
    def segment(self, text, lower = False):
        text = util.as_text(text)
        sentences = self.ss.segment(text)
        words_no_filter = self.ws.segment_sentences(sentences=sentences, 
                                                    lower = lower, 
                                                    use_stop_words = False,
                                                    use_speech_tags_filter = False)
        words_no_stop_words = self.ws.segment_sentences(sentences=sentences, 
                                                    lower = lower, 
                                                    use_stop_words = True,
                                                    use_speech_tags_filter = False)

        words_all_filters = self.ws.segment_sentences(sentences=sentences, 
                                                    lower = lower, 
                                                    use_stop_words = True,
                                                    use_speech_tags_filter = True)

        return util.AttrDict(
                    sentences           = sentences, 
                    words_no_filter     = words_no_filter, 
                    words_no_stop_words = words_no_stop_words, 
                    words_all_filters   = words_all_filters
                )
    
        

if __name__ == '__main__':
    pass