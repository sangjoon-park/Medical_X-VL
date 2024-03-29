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
    def __init__(self, entire_corpus, entire_labels, entire_pair=None, probability=0.25, random_state=1234):
        'probability: individual probability'
        self.probability = probability
        self.entire_corpus = entire_corpus
        self.entire_label = entire_labels

        random.seed(random_state)
        np.random.seed(random_state)

        if entire_pair:
            self.entire_pair = entire_pair
        else:
            self.entire_pair = {}

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

    def mismatch(self, impression):
        'Entirely change report with not overlapping report'
        original_class = self.entire_pair[impression]
        while True:
            selected_idx = np.random.randint(0, len(list(self.entire_corpus.keys()))-1)
            selected_impression = self.entire_corpus.iloc[selected_idx].impression
            selected_impression = pre_caption(selected_impression)
            selected_class = self.entire_pair[selected_impression]
            intersection = list(set(original_class).intersection(selected_class))
            if len(intersection) == 0:
                break
        return selected_impression

    def omission(self, impression):
        'Set: F_T -> F_t, one sentence ommission'
        orig_class = self.entire_pair[impression]

        if 'positive_No Finding' in orig_class:
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

    def confusion(self, impression):
        'Set: F1 -> F2 (random), one sentence change'
        orig_class = self.entire_pair[impression]

        if 'positive_No Finding' in orig_class:
            return impression
        else:
            while True:
                selected_impression = random.choice(list(self.entire_pair.keys()))
                selected_class = self.entire_pair[selected_impression]
                overlap_class = list(set(orig_class).intersection(selected_class))
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

    def positive(self, impression):
        'Set: N -> F (random), false positive'

        original_class = self.entire_pair[impression]
        if 'positive_No Finding' in original_class:
            while True:
                selected_impression = random.choice(list(self.entire_pair.keys()))
                selected_class = self.entire_pair[selected_impression]
                if not 'positive_No Finding' in selected_class:
                    break
            return selected_impression
        else:
            return impression

    def negative(self, impression):
        'Set: F -> N (random), false positive'

        original_class = self.entire_pair[impression]
        if not 'positive_No Finding' in original_class:
            while True:
                selected_impression = random.choice(list(self.entire_pair.keys()))
                selected_class = self.entire_pair[selected_impression]
                if 'positive_No Finding' in selected_class:
                    break
            return selected_impression
        else:
            return impression

    def __call__(self, impression, FP=False, FN=False, L=False, E=False, M=False, train=True):
        """
        Realistic error generation
        input: report (text)
        output: report with error (text)
        """

        if train:
            if random.random() <= self.probability:
                selected = random.randint(0, 4)
                # if O:
                #     if random.random() <= self.probability:
                #         impression = self.omission(impression)
                # if C:
                #     if random.random() <= self.probability:
                #         impression = self.confusion(impression)
                if selected == 0:
                    impression = self.positive(impression)
                if selected == 1:
                    impression = self.negative(impression)
                if selected == 2:
                    impression = self.location(impression)
                if selected == 3:
                    impression = self.extent(impression)
                if selected == 4:
                    impression = self.mismatch(impression)
        else:
            if random.random() <= self.probability:
                if FP:
                    impression = self.positive(impression)
                elif FN:
                    impression = self.negative(impression)
                elif L:
                    impression = self.location(impression)
                elif E:
                    impression = self.extent(impression)
                elif M:
                    impression = self.mismatch(impression)
            else:
                pass

        return impression


if __name__ == '__main__':
    corpus_dir = '/COVID_8TB/sangjoon/mimic_CXR/mimic_impressions_final.csv'
    label_dir = '/COVID_8TB/sangjoon/mimic-cxr/mimic-cxr-2.0.0-chexpert.csv'
    pair_dir = './entire_pair_final.json'

    with open(pair_dir, 'r') as f:
        entire_pair = json.load(f, encoding='cp949')

    entire_texts = pd.read_csv(corpus_dir)
    entire_labels = pd.read_csv(label_dir)

    error_gen = ErrorGenerator(entire_texts, entire_labels, entire_pair)

    # error_gen = ErrorGenerator(entire_pair, probability=0.5)

    for sentence in list(entire_texts.impression):
        sentence = pre_caption(sentence)
        print('original sentence: {}'.format(sentence))
        try:
            error_report = error_gen(sentence, M=True)
        except:
            error_report = sentence
        print('error sentence: {}'.format(error_report))
        print('-----------------------------------------------------')