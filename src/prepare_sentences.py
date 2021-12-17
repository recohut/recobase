#!/usr/bin/env python3

import sys
import pandas as pd
import numpy as np
import re
import os
from pathlib import Path

import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.tokenize import word_tokenize


def contraction_handle(filepath='data/bronze/dialog_data/contractions.txt'):
    contraction_dict = {}
    with open(filepath) as f:
        for key_line in f:
            (key, val) = key_line.split(':')
            contraction_dict[key] = val
    return contraction_dict

class PrepareSentences:
    def __init__(self):
        self.contraction_dict = contraction_handle()
        self.sentence_data = []

    @staticmethod
    def seeker_sentences_parser(line):
        if line:
            p = re.compile("SEEKER:(.*)").search(str(line))
            temp_line = p.group(1)
            m = re.compile('<s>(.*?)</s>').search(temp_line)
            seeker_line = m.group(1)
            seeker_line = seeker_line.lower().strip()
            return seeker_line

    @staticmethod
    def gt_sentence_parser(line):
        try:
            if not line == '\n':
                p = re.compile("GROUND TRUTH:(.*)").search(str(line))
                temp_line = p.group(1)
                m = re.compile('<s>(.*?)</s>').search(temp_line)
                gt_line = m.group(1)
                gt_line = gt_line.lower().strip()
                # gt_line = re.sub('[^A-Za-z0-9]+', ' ', gt_line)
            else:
                gt_line = ""
        except AttributeError as err:
                # print('exception accured while parsing ground truth.. \n')
                # print(line)
                # print(err)
                return gt_line

    @staticmethod
    def replace_movieIds_withPL(line):
        try:
            if "@" in line:
                ids = re.findall(r'@\S+', line)
                for id in ids:
                    line = line.replace(id,'movieid')
                    #id = re.sub('[^0-9@]+', 'movieid', id)
        except:
            lines.append(line)
            # print('exception occured here')
        return line
        # print('execution ends here')

    @staticmethod
    def remove_stopwords(line):
        text_tokens = word_tokenize(line)
        tokens_without_sw = [word for word in text_tokens if not word in stopwords.words()]
        filtered_sentence = (" ").join(tokens_without_sw)
        print(filtered_sentence)
        return filtered_sentence

    @staticmethod
    def convert_contractions(line):
        #line = "What's the best way to ensure this?"
        for word in line.split():
            if word.lower() in self.contraction_dict:
                line = line.replace(word, contraction_dict[word.lower()])
        return line

    #read and retrive dialogs from the input file
    def read_sentences(self, file_name):
        counter =0
        previous_line = ''
        counter = 0
        with open(file_name, 'r', encoding='utf-8') as input:
            for line in input:
                try:
                    #if line.__contains__('~') and line.__contains__('SKR~'):
                    if line:
                        if line.__contains__('CONVERSATION:'):
                            self.sentence_data.append(line.replace('\n',''))
                            continue
                        else:
                            previous_line = line
                            line = self.replace_movieIds_withPL(line)
                            line = line.split('~')[1].strip().lower()
                            line = self.convert_contractions(line)
                            line = re.sub('[^A-Za-z0-9]+', ' ', line)
                            line = line.replace('im','i am').strip()
                            line = self.remove_stopwords(line)
                            if len(line) < 1:
                                self.sentence_data.append('**')
                            else:
                                self.sentence_data.append(line)
                    else:
                        #print('not found')
                        #print(line)
                        #print('previous line is ...' +previous_line)
                        # print('line issue')
                        counter = counter+1
                except:
                    # print((previous_line))
                    # print(line)
                    continue

    def write_data(self, filepath):
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as filehandle:
            for line in self.sentence_data:
                filehandle.write("%s\n" % line)



if __name__ == '__main__':
    prep = PrepareSentences()
    prep.read_sentences('data/silver/dialog_data/parsed_dialogs/training_data_parsed_con.txt')
    prep.write_data('data/gold/dialog_data/dialog_sentences/training_data_plsw.txt')
    print('Dialogs have been preprocessed successfully.')