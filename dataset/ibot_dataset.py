# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import random
import math
import numpy as np
import json

import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
from dataset.utils import pre_caption, pre_question, shuffle
import torch
import glob
from utils import post_process
import cv2
import h5py

class ImageFolderMask(Dataset):
    def __init__(self, ann_file, patch_size, pred_ratio, pred_ratio_var, pred_aspect_ratio,
                 pred_shape='block', mode='train', transforms_1=None, transforms_2=None, pred_start_epoch=0, max_words=120):
        self.psz = patch_size
        self.pred_ratio = pred_ratio[0] if isinstance(pred_ratio, list) and \
                                           len(pred_ratio) == 1 else pred_ratio
        self.pred_ratio_var = pred_ratio_var[0] if isinstance(pred_ratio_var, list) and \
                                                   len(pred_ratio_var) == 1 else pred_ratio_var
        if isinstance(self.pred_ratio, list) and not isinstance(self.pred_ratio_var, list):
            self.pred_ratio_var = [self.pred_ratio_var] * len(self.pred_ratio)
        self.log_aspect_ratio = tuple(map(lambda x: math.log(x), pred_aspect_ratio))
        self.pred_shape = pred_shape
        self.pred_start_epoch = pred_start_epoch

        # caption label and etc.
        self.img_dset = h5py.File(ann_file[0], 'r')['cxr']
        self.df = pd.read_csv(ann_file[1])

        self.index_mapping = []

        for i in range(len(self.df)):
            split = self.df.iloc[i].split
            findings = self.df.iloc[i].findings
            impression = self.df.iloc[i].impression
            views = self.df.iloc[i].views
            # if split == mode and findings != 'NONE' and impression != 'NONE' and type(findings) != float and type(impression) != float:
            if split == mode and impression != 'NONE' and type(impression) != float:
                check_impression = shuffle(pre_caption(impression, 90))
                if len(check_impression) > 1:
                    if views == 'AP' or views == 'PA':
                        self.index_mapping.append(i)

        if len(self.img_dset) != len(self.df):
            raise AssertionError()

        self.max_words = max_words
        self.transforms_1 = transforms_1
        self.transforms_2 = transforms_2

    def __len__(self):
        return len(self.index_mapping)

    def _resize_img(self, img, scale):
        """
        Args:
            img - image as numpy array (cv2)
            scale - desired output image-size as scale x scale
        Return:
            image resized to scale x scale with shortest dimension 0-padded
        """
        size = img.shape
        max_dim = max(size)
        max_ind = size.index(max_dim)

        # Resizing
        if max_ind == 0:
            # image is heigher
            wpercent = scale / float(size[0])
            hsize = int((float(size[1]) * float(wpercent)))
            desireable_size = (scale, hsize)
        else:
            # image is wider
            hpercent = scale / float(size[1])
            wsize = int((float(size[0]) * float(hpercent)))
            desireable_size = (wsize, scale)
        resized_img = cv2.resize(
            img, desireable_size[::-1], interpolation=cv2.INTER_AREA
        )  # this flips the desireable_size vector

        # Padding
        if max_ind == 0:
            # height fixed at scale, pad the width
            pad_size = scale - resized_img.shape[1]
            left = int(np.floor(pad_size / 2))
            right = int(np.ceil(pad_size / 2))
            top = int(0)
            bottom = int(0)
        else:
            # width fixed at scale, pad the height
            pad_size = scale - resized_img.shape[0]
            top = int(np.floor(pad_size / 2))
            bottom = int(np.ceil(pad_size / 2))
            left = int(0)
            right = int(0)
        resized_img = np.pad(
            resized_img, [(top, bottom), (left, right)], "constant", constant_values=0
        )

        return resized_img

    def get_pred_ratio(self):
        if hasattr(self, 'epoch') and self.epoch < self.pred_start_epoch:
            return 0

        if isinstance(self.pred_ratio, list):
            pred_ratio = []
            for prm, prv in zip(self.pred_ratio, self.pred_ratio_var):
                assert prm >= prv
                pr = random.uniform(prm - prv, prm + prv) if prv > 0 else prm
                pred_ratio.append(pr)
            pred_ratio = random.choice(pred_ratio)
        else:
            assert self.pred_ratio >= self.pred_ratio_var
            pred_ratio = random.uniform(self.pred_ratio - self.pred_ratio_var, self.pred_ratio + \
                                        self.pred_ratio_var) if self.pred_ratio_var > 0 else self.pred_ratio

        return pred_ratio

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __getitem__(self, mapping_idx):
        # ann = self.ann[index]
        index = self.index_mapping[mapping_idx]

        image = self.img_dset[index]

        # image_path = self.ann[index]['image']
        # image = cv2.imread(str(image_path), 0)

        image = self._resize_img(image, 224)
        image = Image.fromarray(image).convert('RGB')

        # Image augmentation
        output_1 = self.transforms_1(image)  # jinyu
        output_2 = self.transforms_2(image)

        # findings = self.df.iloc[index].findings
        impression = self.df.iloc[index].impression
        # overall = findings + ' ' + impression

        # findings = pre_caption(findings, self.max_words)
        impression = pre_caption(impression, self.max_words)
        # overall = pre_caption(overall, self.max_words)

        # Text augmentation
        # findings = shuffle(findings)
        impression = shuffle(impression)
        # overall = shuffle(overall)

        return output_1, output_2, impression


class Retrieval_dataset(Dataset):
    def __init__(self, ann_file, query_file, transforms=None, max_words=120):

        self.candidates = pd.read_csv(ann_file)
        self.queries = pd.read_csv(query_file)

        self.max_words = max_words
        self.transforms = transforms

        self.labels = {}
        self.text_queries = []
        for i in range(len(self.queries)):
            query = pre_caption(self.queries.iloc[i].Text, self.max_words)
            label = self.queries.iloc[i].Variable
            self.labels[query] = label
            self.text_queries.append(query)

        self.findings = ["Pneumothorax", "Pneumonia", "Fracture", "Cardiomegaly",
                         "Pleural Effusion", "Edema", "Atelectasis", "No Finding"]

    def __len__(self):
        return len(self.candidates)

    def _resize_img(self, img, scale):
        """
        Args:
            img - image as numpy array (cv2)
            scale - desired output image-size as scale x scale
        Return:
            image resized to scale x scale with shortest dimension 0-padded
        """
        size = img.shape
        max_dim = max(size)
        max_ind = size.index(max_dim)

        # Resizing
        if max_ind == 0:
            # image is heigher
            wpercent = scale / float(size[0])
            hsize = int((float(size[1]) * float(wpercent)))
            desireable_size = (scale, hsize)
        else:
            # image is wider
            hpercent = scale / float(size[1])
            wsize = int((float(size[0]) * float(hpercent)))
            desireable_size = (wsize, scale)
        resized_img = cv2.resize(
            img, desireable_size[::-1], interpolation=cv2.INTER_AREA
        )  # this flips the desireable_size vector

        # Padding
        if max_ind == 0:
            # height fixed at scale, pad the width
            pad_size = scale - resized_img.shape[1]
            left = int(np.floor(pad_size / 2))
            right = int(np.ceil(pad_size / 2))
            top = int(0)
            bottom = int(0)
        else:
            # width fixed at scale, pad the height
            pad_size = scale - resized_img.shape[0]
            top = int(np.floor(pad_size / 2))
            bottom = int(np.ceil(pad_size / 2))
            left = int(0)
            right = int(0)
        resized_img = np.pad(
            resized_img, [(top, bottom), (left, right)], "constant", constant_values=0
        )

        return resized_img

    def __getitem__(self, index):
        image_id = self.candidates.iloc[index].Path
        image_id = image_id.replace('CheXpert-v1.0-small', '/COVID_8TB/1_data/OpenCXR/CheXpert-v1.0')
        image = cv2.imread(image_id, 0)

        # Preprocess same to the MIMIC-jpg
        image = cv2.resize(image, dsize=(320, 320))
        image = image - image.min()
        image = image / image.max()

        image = image * 255
        image = image.astype(np.uint8)
        image = cv2.equalizeHist(image)

        image = self._resize_img(image, 224)
        image = Image.fromarray(image).convert('RGB')
        output = self.transforms(image)

        target_index = np.zeros(8)
        if self.candidates.iloc[index]["Pneumothorax"]:
            target_index[0] = 1
        if self.candidates.iloc[index]["Pneumonia"]:
            target_index[1] = 1
        if self.candidates.iloc[index]["Fracture"]:
            target_index[2] = 1
        if self.candidates.iloc[index]["Cardiomegaly"]:
            target_index[3] = 1
        if self.candidates.iloc[index]["Pleural Effusion"]:
            target_index[4] = 1
        if self.candidates.iloc[index]["Edema"]:
            target_index[5] = 1
        if self.candidates.iloc[index]["Atelectasis"]:
            target_index[6] = 1
        if self.candidates.iloc[index]["No Finding"]:
            target_index[7] = 1

        if target_index.sum() > 1:
            raise AssertionError()

        target = np.nonzero(target_index)[0][0]

        return output, self.findings[target], image_id

class Vqa_dataset(Dataset):
    def __init__(self, ann_file, transform, vqa_root, vg_root, eos='[SEP]', split="train", max_ques_words=120, chest_only=True,
                 answer_list=''):
        self.split = split
        self.ann = []
        for f in ann_file:
            self.ann += json.load(open(f, 'r'))

        self.transform = transform
        self.vqa_root = vqa_root
        self.vg_root = vg_root
        self.max_ques_words = max_ques_words
        self.eos = eos

        self.selected_ann = []
        for data in self.ann:
            if chest_only:
                if data['image_organ'] == 'CHEST':
                    self.selected_ann.append(data)
                else:
                    pass
            else:
                self.selected_ann.append(data)
        self.ann = self.selected_ann

        if split == 'test':
            self.max_ques_words = 120  # do not limit question length during test
            self.answer_list = []
            for ans in self.ann:
                answer = pre_question(ans['answer'], self.max_ques_words)
                answer = post_process(answer)

                if not answer in self.answer_list:
                    self.answer_list.append(answer)

    def __len__(self):
        return len(self.ann)

    def _resize_img(self, img, scale):
        """
        Args:
            img - image as numpy array (cv2)
            scale - desired output image-size as scale x scale
        Return:
            image resized to scale x scale with shortest dimension 0-padded
        """
        size = img.shape
        max_dim = max(size)
        max_ind = size.index(max_dim)

        # Resizing
        if max_ind == 0:
            # image is heigher
            wpercent = scale / float(size[0])
            hsize = int((float(size[1]) * float(wpercent)))
            desireable_size = (scale, hsize)
        else:
            # image is wider
            hpercent = scale / float(size[1])
            wsize = int((float(size[0]) * float(hpercent)))
            desireable_size = (wsize, scale)
        resized_img = cv2.resize(
            img, desireable_size[::-1], interpolation=cv2.INTER_AREA
        )  # this flips the desireable_size vector

        # Padding
        if max_ind == 0:
            # height fixed at scale, pad the width
            pad_size = scale - resized_img.shape[1]
            left = int(np.floor(pad_size / 2))
            right = int(np.ceil(pad_size / 2))
            top = int(0)
            bottom = int(0)
        else:
            # width fixed at scale, pad the height
            pad_size = scale - resized_img.shape[0]
            top = int(np.floor(pad_size / 2))
            bottom = int(np.ceil(pad_size / 2))
            left = int(0)
            right = int(0)
        resized_img = np.pad(
            resized_img, [(top, bottom), (left, right)], "constant", constant_values=0
        )

        return resized_img

    def __getitem__(self, index):

        ann = self.ann[index]
        image_id = '/COVID_8TB/sangjoon/vision_language/data_RAD/home/mimic-cxr/dataset/data_RAD/images/' + ann['image_name']
        image = cv2.imread(image_id, 0)

        # Preprocess same to the MIMIC-jpg
        image = cv2.resize(image, dsize=(320, 320))
        image = image - image.min()
        image = image / image.max()

        image = image * 255
        image = image.astype(np.uint8)
        image = cv2.equalizeHist(image)

        image = self._resize_img(image, 224)
        image = Image.fromarray(image).convert('RGB')
        image = self.transform(image)

        if self.split == 'test':
            question = pre_question(str(ann['question']), self.max_ques_words)
            question_id = ann['qid']
            return image, question, question_id


        elif self.split == 'train':

            question = pre_question(str(ann['question']), self.max_ques_words)

            # if ann['dataset']=='vqa':

            processed_answer = pre_question(str(ann['answer']), self.max_ques_words)
            ann_all = [processed_answer for x in range(10)]

            answer_weight = {}
            for answer in ann_all:
                if answer in answer_weight.keys():
                    answer_weight[answer] += 1 / len(ann_all)
                else:
                    answer_weight[answer] = 1 / len(ann_all)

            answers = list(answer_weight.keys())
            weights = list(answer_weight.values())

            answers = [str(answer) + self.eos for answer in answers]

            return image, question, answers, weights


class Gen_dataset(Dataset):
    def __init__(self, ann_file, transforms=None, eos='[SEP]', mode='train', max_words=120):
        self.mode = mode

        # caption label and etc.
        self.img_dset = h5py.File(ann_file[0], 'r')['cxr']
        self.df = pd.read_csv(ann_file[1])

        self.index_mapping = []

        for i in range(len(self.df)):
            split = self.df.iloc[i].split
            findings = self.df.iloc[i].findings
            impression = self.df.iloc[i].impression
            # view = self.df.iloc[i].view
            if split == mode and findings != 'NONE' and type(findings) != float:
                self.index_mapping.append(i)

        if len(self.img_dset) != len(self.df):
            raise AssertionError()

        print('Total dataset: {}'.format(len(self.index_mapping)))

        self.max_words = max_words
        self.transforms = transforms

        self.transforms = transforms
        self.img_ids = {}

    def __len__(self):
        return len(self.index_mapping)

    def _resize_img(self, img, scale):
        """
        Args:
            img - image as numpy array (cv2)
            scale - desired output image-size as scale x scale
        Return:
            image resized to scale x scale with shortest dimension 0-padded
        """
        size = img.shape
        max_dim = max(size)
        max_ind = size.index(max_dim)

        # Resizing
        if max_ind == 0:
            # image is heigher
            wpercent = scale / float(size[0])
            hsize = int((float(size[1]) * float(wpercent)))
            desireable_size = (scale, hsize)
        else:
            # image is wider
            hpercent = scale / float(size[1])
            wsize = int((float(size[0]) * float(hpercent)))
            desireable_size = (wsize, scale)
        resized_img = cv2.resize(
            img, desireable_size[::-1], interpolation=cv2.INTER_AREA
        )  # this flips the desireable_size vector

        # Padding
        if max_ind == 0:
            # height fixed at scale, pad the width
            pad_size = scale - resized_img.shape[1]
            left = int(np.floor(pad_size / 2))
            right = int(np.ceil(pad_size / 2))
            top = int(0)
            bottom = int(0)
        else:
            # width fixed at scale, pad the height
            pad_size = scale - resized_img.shape[0]
            top = int(np.floor(pad_size / 2))
            bottom = int(np.ceil(pad_size / 2))
            left = int(0)
            right = int(0)
        resized_img = np.pad(
            resized_img, [(top, bottom), (left, right)], "constant", constant_values=0
        )

        return resized_img

    def __getitem__(self, mapping_idx):
        # ann = self.ann[index]
        index = self.index_mapping[mapping_idx]
        image_id = self.df.iloc[index].dicom_id

        image = self.img_dset[index]

        # image_path = self.ann[index]['image']
        # image = cv2.imread(str(image_path), 0)

        image = self._resize_img(image, 224)
        image = Image.fromarray(image).convert('RGB')
        output = self.transforms(image)  # jinyu

        findings = self.df.iloc[index].findings
        findings = pre_caption(findings, self.max_words)

        return output, findings, image_id

class Cls_dataset(Dataset):
    def __init__(self, ann_file, transforms=None, max_words=100):

        self.df = pd.read_csv(ann_file)

        self.max_words = max_words
        self.transforms = transforms

        print('number of data: {}'.format(len(self.df)))

    def __len__(self):
        return len(self.df)

    def _resize_img(self, img, scale):
        """
        Args:
            img - image as numpy array (cv2)
            scale - desired output image-size as scale x scale
        Return:
            image resized to scale x scale with shortest dimension 0-padded
        """
        size = img.shape
        max_dim = max(size)
        max_ind = size.index(max_dim)

        # Resizing
        if max_ind == 0:
            # image is heigher
            wpercent = scale / float(size[0])
            hsize = int((float(size[1]) * float(wpercent)))
            desireable_size = (scale, hsize)
        else:
            # image is wider
            hpercent = scale / float(size[1])
            wsize = int((float(size[0]) * float(hpercent)))
            desireable_size = (wsize, scale)
        resized_img = cv2.resize(
            img, desireable_size[::-1], interpolation=cv2.INTER_AREA
        )  # this flips the desireable_size vector

        # Padding
        if max_ind == 0:
            # height fixed at scale, pad the width
            pad_size = scale - resized_img.shape[1]
            left = int(np.floor(pad_size / 2))
            right = int(np.ceil(pad_size / 2))
            top = int(0)
            bottom = int(0)
        else:
            # width fixed at scale, pad the height
            pad_size = scale - resized_img.shape[0]
            top = int(np.floor(pad_size / 2))
            bottom = int(np.ceil(pad_size / 2))
            left = int(0)
            right = int(0)
        resized_img = np.pad(
            resized_img, [(top, bottom), (left, right)], "constant", constant_values=0
        )

        return resized_img

    def __getitem__(self, index):
        image_id = self.df.iloc[index].Path
        image_id = image_id.replace('.dcm', '.jpg')
        image = cv2.imread(image_id, 0)
        image = self._resize_img(image, 224)
        image = Image.fromarray(image).convert('RGB')
        output = self.transforms(image)

        target = self.df.iloc[index].Target

        # if target == 0:
        #     report = 'No evidence of pneumonia.'
        # elif target == 1:
        #     report = 'Findings suggesting pneumonia.'

        return output, target, image_id

