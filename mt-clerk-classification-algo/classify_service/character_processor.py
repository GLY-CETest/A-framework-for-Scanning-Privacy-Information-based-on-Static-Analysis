#!/usr/bin/python
#coding=utf-8

import pyltp
from pyltp import Segmentor,Postagger
import os


class DataProcessor(object):
    def __init__(self, filepath_cws, filepath_pos, user_dict_filepath = None):
        super(DataProcessor, self).__init__()
        self.my_segmentor = Segmentor(filepath_cws)
        # self.my_segmentor.load(filepath_cws)
        self.postagger = Postagger(filepath_pos)
        # self.postagger.load(filepath_pos)

    def __del__(self):
        self.my_segmentor.release()
        self.postagger.release()
        
    def segmentor(self, sentence,):
        ltpword = self.my_segmentor.segment(sentence)
        ltpword_list = list(ltpword)
        return ltpword_list

    def stop_word_list(self, filepath):
        stop_word = [line.strip() for line in open(filepath, "r", encoding = "utf-8").readlines()]
        return stop_word

    def clean_word_list(self, origin_data, stop_word):
        clean_word = [word for word in origin_data if word not in stop_word]
        postags = list(self.postagger.postag(clean_word))
        clean_word_after_postagger = []
        for i in range(len(postags)):
            if postags[i] == "n" or postags[i] == "v":
                clean_word_after_postagger.append(clean_word[i]) 
        return clean_word_after_postagger

    def synonym_word_dict(self, filepath):
        synonym_dict = []
        for line in open(filepath, "r", encoding = "utf-8"):
            items = line.replace("\n", "").split(" ")
            index = items[0]
            if(index[-1] == "="):
                synonym_dict.append(items[1:])
        return synonym_dict

    def synonym_replace_word(self, word, synonym_dict):
        for each in synonym_dict:
            for w in each:
                if w == word:
                    return each[0] # 同义词替换为同义词表中的第一个词
        return word

    def synonym_replace_sentence(self, clean_word, synonym_dict):
        for i in range(len(clean_word)):
            clean_word[i] = self.synonym_replace_word(clean_word[i], synonym_dict)
        return clean_word

if __name__ == "__main__":
    processor = DataProcessor("./ltp_data_v3.4.0/cws.model", "./ltp_data_v3.4.0/pos.model")
    myList = processor.segmentor("我们是人工智能研究所，主要致力于分享人工智能方面的技术知识，欢迎大家一起学习。")
    stop_word = processor.stop_word_list("./ltp_data_v3.4.0/stop_word.txt")
    clean_word = processor.clean_word_list(myList, stop_word)
    print(clean_word)

    synonym_dict = processor.synonym_word_dict("./ltp_data_v3.4.0/HIT-IRLab-同义词词林.txt")
    clean_word_after_synonym = processor.synonym_replace_sentence(clean_word, synonym_dict)
    print(clean_word_after_synonym)
