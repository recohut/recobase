#!/usr/bin/env python3

import pandas as pd
import numpy as np
from pathlib import Path
import os
import simplejson as json


class Dataset:
    def __init__(self):
        self.data = None
        self.text_messages_raw = []

    def read_input_json_file(self, filepath):
        with open(filepath, 'r', encoding='utf-8') as json_file:
            self.data = json.load(json_file)

    def parse_dialogues(self):
        dialogs = self.data['foo']
        counter = 0
        for key, d in enumerate(dialogs):
            messages = dialogs[key]['messages']
            seeker_id = dialogs[key]['initiatorWorkerId']
            recommender_id = dialogs[key]['respondentWorkerId']
            seeker_text = ''
            gt_text = ''
            counter = counter +1
            self.text_messages_raw.append('CONVERSATION:'+ str(counter))
            for msgid, msg in enumerate(messages):

                senderId = messages[msgid]['senderWorkerId']
                if senderId == seeker_id:
                    if gt_text:
                        self.text_messages_raw.append('GT~' + gt_text)
                        gt_text = ''
                        seeker_text =  seeker_text +' '+ messages[msgid]['text']
                    else:
                        seeker_text =  seeker_text +' ' + messages[msgid]['text']

                elif senderId == recommender_id:
                    if seeker_text:
                        self.text_messages_raw.append('SKR~' + seeker_text)
                        seeker_text = ''
                        gt_text = gt_text+' '  + messages[msgid]['text']
                    else:
                        gt_text = gt_text +' ' + messages[msgid]['text']

            if gt_text:
                self.text_messages_raw.append('GT~' + gt_text)
            elif seeker_text:
                self.text_messages_raw.append('SKR~' + seeker_text)

    def write_data(self, filepath):
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as filehandle:
            for line in self.text_messages_raw:
                filehandle.write("%s\n" % line)


if __name__ == '__main__':
    dataset = Dataset()
    dataset.read_input_json_file('data/bronze/dialog_data/unparsed_train_data.txt')
    dataset.parse_dialogues()
    dataset.write_data('data/silver/dialog_data/parsed_dialogs/training_data_parsed_con.txt')
    print('data exported')