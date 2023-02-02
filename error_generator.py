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

class ErrorGenerator(object):
    def __init__(self, probability=0.05):
        self.probability = probability
        self.entire_text = pd.read_csv('')

    def localization(self, report):
        keywords = ['left', 'right', 'upper', 'lower', 'high', 'low', 'big', 'small']
        counter_keywords = {'left': 'right', 'right': 'left', 'upper':'lower', 'lower': 'upper', 'high': 'low',
                            'low': 'high', 'big': 'small', 'small': 'big'}

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

    def numerical(self, report):

        text_split = report.split()
        error_report = []
        for word in text_split:
            if 'cm' in word or 'mm' in word:
                if random.random() <= 0.5:
                    previous_word = error_report[-1]
                    if random.random() <= 0.5:
                        previous_word = str(int(previous_word) * 10.)
                    else:
                        previous_word = str(int(previous_word) * 0.1)
                    error_report = error_report[:-1]
                    error_report.append(previous_word)
                else:
                    if 'cm' in word:
                        if random.random() <= 0.5:
                            word = 'm'
                        else:
                            word = 'mm'
                    elif 'mm' in word:
                        if random.random() <= 0.5:
                            word = 'm'
                        else:
                            word = 'cm'
            error_report.append(word)
        error_report = " ".join(error_report)

        return error_report

    def reasoning(self, impression):
        original_class = self.entire_texts[impression]
        if original_class != ['No Finding']:
            while True:
                selected_impression = random.choice(list(self.entire_texts.keys()))
                selected_class = self.entire_texts[selected_impression]
                intersection = list(set(original_class).intersection(selected_class))
                if len(intersection) == 0 and selected_class != ['No Finding']:
                    break
            return selected_impression
        else:
            return impression

    def absence(self, impression):
        original_class = self.entire_texts[impression]
        if original_class != ['No Finding']:
            while True:
                selected_impression = random.choice(list(self.entire_texts.keys()))
                selected_class = self.entire_texts[selected_impression]
                if selected_class == ['No Finding']:
                    break
            return selected_impression
        else:
            return impression

    def presence(self, impression):
        original_class = self.entire_texts[impression]
        if original_class == ['No Finding']:
            while True:
                selected_impression = random.choice(list(self.entire_texts.keys()))
                selected_class = self.entire_texts[selected_impression]
                if selected_class != ['No Finding']:
                    break
            return selected_impression
        else:
            return impression

    def mismatch(self, findings, impression):
        original_class = self.entire_texts[impression]
        while True:
            selected_idx = np.random.randint(0, len(list(self.entire_text.keys()))-1)
            selected_impression = self.entire_text.iloc[selected_idx].impression
            selected_findings = self.entire_text.iloc[selected_idx].findings
            selected_class = self.entire_texts[selected_impression]
            intersection = list(set(original_class).intersection(selected_class))
            if len(intersection):
                break
        return selected_findings, selected_impression

    def forward(self, findings=None, impression=None):
        """
        Realistic error generation
        input: report (text)
        output: report with error (text)
        """

        # Localization error
        if random.random() <= self.probability:
            findings = self.localization(findings)
            impression = self.localization(impression)

        # Numerical measurement error
        if random.random() <= self.probability:
            findings = self.numerical(findings)
            impression = self.numerical(impression)

        # Interpretation error
        error_selection = random.random()
        if error_selection < 0.3333:
            # Faulty reasoning
            if random.random() <= self.probability:
                impression = self.reasoning(impression)
        elif error_selection >= 0.3333 and error_selection < 0.6666:
            # Absence of abnormality
            if random.random() <= self.probability:
                impression = self.absence(impression)
        elif error_selection >= 0.6666:
            # Presence of incorrect finding
            if random.random() <= self.probability:
                impression = self.presence(impression)

        # Registration error (Patient-report mismatch)
        if random.random() <= self.probability:
            findings, impression = self.mismatch(findings, impression)

        return findings, impression