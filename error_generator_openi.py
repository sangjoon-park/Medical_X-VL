import numpy as np
import io
import os
import time
from collections import defaultdict, deque
import datetime

import torch
import torch.distributed as dist
import math
import warnings
import yaml
import cv2
import random
import pandas as pd
from tqdm import tqdm
from dataset.utils import pre_caption

import json

class ErrorGenerator(object):
    def __init__(self, entire_corpus, probability=0.5):
        'probability: individual probability'
        self.probability = probability
        self.entire_corpus = entire_corpus
        # self.entire_label = entire_label
        #
        # if entire_pair:
        #     self.entire_pair = entire_pair
        # else:
        #     self.entire_pair = {}
        #
        # for study_id in tqdm(list(entire_labels.study_id)):
        #     labels = entire_labels.loc[entire_labels.study_id == study_id]
        #     study_id_ = 's' + str(study_id) + '.txt'
        #     impression = entire_corpus.loc[entire_corpus.filename == study_id_].impression.values[0]
        #     try:
        #         impression = pre_caption(impression)
        #     except:
        #         impression = impression
        #
        #     all_labels = []
        #
        #     if not np.isnan(labels['Atelectasis'].values)[0]:
        #         if labels['Atelectasis'].values[0] == 1:
        #             all_labels.append('positive_Atelectasis')
        #         elif labels['Atelectasis'].values[0] == -1:
        #             all_labels.append('probable_Atelectasis')
        #         elif labels['Atelectasis'].values[0] == -0:
        #             all_labels.append('negative_Atelectasis')
        #
        #     if not np.isnan(labels['Cardiomegaly'].values)[0]:
        #         if labels['Cardiomegaly'].values[0] == 1:
        #             all_labels.append('positive_Cardiomegaly')
        #         elif labels['Cardiomegaly'].values[0] == -1:
        #             all_labels.append('probable_Cardiomegaly')
        #         elif labels['Cardiomegaly'].values[0] == -0:
        #             all_labels.append('negative_Cardiomegaly')
        #
        #     if not np.isnan(labels['Consolidation'].values)[0]:
        #         if labels['Consolidation'].values[0] == 1:
        #             all_labels.append('positive_Consolidation')
        #         elif labels['Consolidation'].values[0] == -1:
        #             all_labels.append('probable_Consolidation')
        #         elif labels['Consolidation'].values[0] == -0:
        #             all_labels.append('negative_Consolidation')
        #
        #     if not np.isnan(labels['Edema'].values)[0]:
        #         if labels['Edema'].values[0] == 1:
        #             all_labels.append('positive_Edema')
        #         elif labels['Edema'].values[0] == -1:
        #             all_labels.append('probable_Edema')
        #         elif labels['Edema'].values[0] == -0:
        #             all_labels.append('negative_Edema')
        #
        #     if not np.isnan(labels['Enlarged Cardiomediastinum'].values)[0]:
        #         if labels['Enlarged Cardiomediastinum'].values[0] == 1:
        #             all_labels.append('positive_Enlarged Cardiomediastinum')
        #         elif labels['Enlarged Cardiomediastinum'].values[0] == -1:
        #             all_labels.append('probable_Enlarged Cardiomediastinum')
        #         elif labels['Enlarged Cardiomediastinum'].values[0] == -0:
        #             all_labels.append('negative_Enlarged Cardiomediastinum')
        #
        #     if not np.isnan(labels['Fracture'].values)[0]:
        #         if labels['Fracture'].values[0] == 1:
        #             all_labels.append('positive_Fracture')
        #         elif labels['Fracture'].values[0] == -1:
        #             all_labels.append('probable_Fracture')
        #         elif labels['Fracture'].values[0] == -0:
        #             all_labels.append('negative_Fracture')
        #
        #     if not np.isnan(labels['Lung Lesion'].values)[0]:
        #         if labels['Lung Lesion'].values[0] == 1:
        #             all_labels.append('positive_Lung Lesion')
        #         elif labels['Lung Lesion'].values[0] == -1:
        #             all_labels.append('probable_Lung Lesion')
        #         elif labels['Lung Lesion'].values[0] == -0:
        #             all_labels.append('negative_Lung Lesion')
        #
        #     if not np.isnan(labels['Lung Opacity'].values)[0]:
        #         if labels['Lung Opacity'].values[0] == 1:
        #             all_labels.append('positive_Lung Opacity')
        #         elif labels['Lung Opacity'].values[0] == -1:
        #             all_labels.append('probable_Lung Opacity')
        #         elif labels['Lung Opacity'].values[0] == -0:
        #             all_labels.append('negative_Lung Opacity')
        #
        #     if not np.isnan(labels['No Finding'].values)[0]:
        #         if labels['No Finding'].values[0] == 1:
        #             all_labels.append('positive_No Finding')
        #         elif labels['No Finding'].values[0] == -1:
        #             all_labels.append('probable_No Finding')
        #         elif labels['No Finding'].values[0] == -0:
        #             all_labels.append('negative_No Finding')
        #
        #     if not np.isnan(labels['Pleural Effusion'].values)[0]:
        #         if labels['Pleural Effusion'].values[0] == 1:
        #             all_labels.append('positive_Pleural Effusion')
        #         elif labels['Pleural Effusion'].values[0] == -1:
        #             all_labels.append('probable_Pleural Effusion')
        #         elif labels['Pleural Effusion'].values[0] == -0:
        #             all_labels.append('negative_Pleural Effusion')
        #
        #     if not np.isnan(labels['Pleural Other'].values)[0]:
        #         if labels['Pleural Other'].values[0] == 1:
        #             all_labels.append('positive_Pleural Other')
        #         elif labels['Pleural Other'].values[0] == -1:
        #             all_labels.append('probable_Pleural Other')
        #         elif labels['Pleural Other'].values[0] == -0:
        #             all_labels.append('negative_Pleural Other')
        #
        #     if not np.isnan(labels['Pneumonia'].values)[0]:
        #         if labels['Pneumonia'].values[0] == 1:
        #             all_labels.append('positive_Pneumonia')
        #         elif labels['Pneumonia'].values[0] == -1:
        #             all_labels.append('probable_Pneumonia')
        #         elif labels['Pneumonia'].values[0] == -0:
        #             all_labels.append('negative_Pneumonia')
        #
        #     if not np.isnan(labels['Pneumothorax'].values)[0]:
        #         if labels['Pneumothorax'].values[0] == 1:
        #             all_labels.append('positive_Pneumothorax')
        #         elif labels['Pneumothorax'].values[0] == -1:
        #             all_labels.append('probable_Pneumothorax')
        #         elif labels['Pneumothorax'].values[0] == -0:
        #             all_labels.append('negative_Pneumothorax')
        #
        #     if not np.isnan(labels['Support Devices'].values)[0]:
        #         if labels['Support Devices'].values[0] == 1:
        #             all_labels.append('positive_Support Devices')
        #         elif labels['Support Devices'].values[0] == -1:
        #             all_labels.append('probable_Support Devices')
        #         elif labels['Support Devices'].values[0] == -0:
        #             all_labels.append('negative_Support Devices')
        #
        #     self.entire_pair[impression] = all_labels
        #
        # with open('./entire_pair_final.json', 'w') as f:
        #     json.dump(self.entire_pair, f)

    def location(self, report):
        keywords = ['left', 'right', 'left-sided', 'right-sided', 'upper', 'lower', 'apical', 'basal', 'central', 'peripheral']
        counter_keywords = {'left': 'right', 'right': 'left', 'left-sided': 'right-sided', 'right-sided': 'left-sided', 'upper': 'lower', 'lower': 'upper', 'apical': 'basal',
                            'basal': 'apical', 'central': 'peripheral', 'peripheral': 'central'}

        text_split = report.split()
        error_report = []
        for word in text_split:
            if word in keywords:
                counter_word = counter_keywords[word]
                error_report.append(counter_word)
            else:
                error_report.append(word)
        error_report = " ".join(error_report)

        return error_report

    def extent(self, report):
        keywords = ['mild', 'slight', 'severe', 'minimal', 'extensive', 'small', 'large', 'mildly', 'severely', 'slightly']
        counter_keywords = {'mild': 'severe', 'slight': 'severe', 'severe': 'mild', 'minimal': 'extensive', 'extensive': 'minimal',
                            'small': 'large', 'large': 'small', 'severely': 'mildly', 'mildly': 'severely', 'slightly': 'severely'}

        text_split = report.split()
        error_report = []
        for word in text_split:
            if word in keywords:
                counter_word = counter_keywords[word]
                error_report.append(counter_word)
            else:
                error_report.append(word)
        error_report = " ".join(error_report)

        return error_report

    def mismatch(self, impression, id):
        'Entirely change report with not overlapping report'
        original_class = self.entire_corpus[self.entire_corpus.id == id].label.values
        while True:
            selected_id = random.sample(list(self.entire_corpus.id), 1)[0]
            selected_impression = self.entire_corpus[self.entire_corpus.id==selected_id].text.values[0]
            selected_impression = pre_caption(selected_impression)
            selected_class = self.entire_corpus[self.entire_corpus.id==selected_id].label.values
            intersection = list(set(original_class).intersection(selected_class))
            if len(intersection) == 0:
                break
        return selected_impression

    def omission(self, impression, id):
        'Set: F_T -> F_t, one sentence ommission'
        orig_class = self.entire_corpus[self.entire_corpus.id==id].label.values

        if "'No Finding'" in orig_class:
            return impression
        else:
            sentences_ = impression.split('.')
            sentences = []
            for s in sentences_:
                if len(s) < 3:
                    continue
                if s[0] == ' ':
                    s = s[1:]
                sentences.append(s)
            if len(sentences) == 1:
                return impression
            else:
                del_idx = random.randint(0, len(sentences) - 1)
                sentences.pop(del_idx)
                impression = '. '.join(sentences) + '.'
                return impression

    def confusion(self, impression, id):
        'Set: F1 -> F2 (random), one sentence change'
        orig_class = self.entire_corpus[self.entire_corpus.id==id].label.values

        if "'No Finding'" in orig_class:
            return impression
        else:
            while True:
                selected_id = random.sample(list(self.entire_corpus.id), 1)[0]
                selected_class = self.entire_corpus[self.entire_corpus.id == selected_id].label.values
                overlap_class = list(set(orig_class).intersection(selected_class))
                selected_impression = pre_caption(self.entire_corpus[self.entire_corpus.id == selected_id].text.values[0])
                if len(overlap_class) == 0:
                    break
            # replace with one sentence from candidate
            selected_sentences_ = selected_impression.split('.')
            selected_sentences = []
            for s in selected_sentences_:
                if len(s) < 3:
                    continue
                if s[0] == ' ':
                    s = s[1:]
                selected_sentences.append(s)
            sel_idx = random.randint(0, len(selected_sentences) - 1)
            selected_sentence = selected_sentences[sel_idx]

            sentences_ = impression.split('.')
            sentences = []
            for s_ in sentences_:
                if len(s_) < 3:
                    continue
                if s_[0] == ' ':
                    s_ = s_[1:]
                sentences.append(s_)
            idx = random.randint(0, len(sentences) - 1)
            sentences[idx] = selected_sentence

            impression = '. '.join(sentences) + '.'
            return impression

    def positive(self, impression, id):
        'Set: N -> F (random), false positive'

        orig_class = self.entire_corpus[self.entire_corpus.id==id].label.values
        if "'No Finding'" in orig_class:
            while True:
                selected_id = random.sample(list(self.entire_corpus.id), 1)[0]
                selected_impression = pre_caption(
                    self.entire_corpus[self.entire_corpus.id == selected_id].text.values[0])
                selected_class = self.entire_corpus[self.entire_corpus.id == selected_id].label.values
                if not "'No Finding'" in selected_class:
                    break
            return selected_impression
        else:
            return impression

    def negative(self, impression):
        'Set: F -> N (random), false positive'

        orig_class = self.entire_corpus[self.entire_corpus.id==id].label.values
        if not "'No Finding'" in orig_class:
            while True:
                selected_id = random.sample(list(self.entire_corpus.id), 1)[0]
                selected_impression = pre_caption(
                    self.entire_corpus[self.entire_corpus.id == selected_id].text.values[0])
                selected_class = self.entire_corpus[self.entire_corpus.id == selected_id].label.values
                if "'No Finding'" in selected_class:
                    break
            return selected_impression
        else:
            return impression

    def __call__(self, impression, id, O=False, C=False, FP=False, FN=True, L=False, E=False, M=False):
        """
        Realistic error generation
        input: report (text)
        output: report with error (text)
        """

        if O:
            if random.random() <= self.probability:
                impression = self.omission(impression, id)
        if C:
            if random.random() <= self.probability:
                impression = self.confusion(impression, id)
        if FP:
            if random.random() <= self.probability:
                impression = self.positive(impression, id)
        if FN:
            if random.random() <= self.probability:
                impression = self.negative(impression)
        if L:
            if random.random() <= self.probability:
                impression = self.location(impression)
        if E:
            if random.random() <= self.probability:
                impression = self.extent(impression)
        if M:
            if random.random() <= self.probability:
                impression = self.mismatch(impression, id)

        return impression


if __name__ == '__main__':
    corpus_dir = './data/openi/Test_selected.jsonl'

    entire_corpus = pd.read_json(corpus_dir, lines=True)

    # entire_texts = pd.read_csv(corpus_dir)
    # entire_labels = pd.read_csv(label_dir)

    error_gen = ErrorGenerator(entire_corpus)

    # error_gen = ErrorGenerator(entire_pair, probability=0.5)

    for id in list(entire_corpus.id):
        sentence = entire_corpus[entire_corpus.id == id].text.values[0]
        sentence = pre_caption(sentence)
        print('original sentence: {}'.format(sentence))
        try:
            error_report = error_gen(sentence, id)
        except:
            error_report = sentence
        print('error sentence: {}'.format(error_report))
        print('-----------------------------------------------------')