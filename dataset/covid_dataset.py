import logging
import numpy as np
import pandas as pd
import os, os.path
from PIL import Image
from torch.utils import data
from torch.utils.data.sampler import WeightedRandomSampler
import torch
import glob
import random
import os
import cv2
import json
from dataset.utils import pre_caption

import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler
import matplotlib.pyplot as plt

from utils import transform

logger = logging.getLogger(__name__)


def get_loader(args, trainset, testset):
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    if args.local_rank == 0:
        torch.distributed.barrier()

    train_sampler = RandomSampler(trainset) if args.local_rank == -1 else DistributedSampler(trainset)

    test_sampler = SequentialSampler(testset)
    train_loader = DataLoader(trainset,
                              sampler=train_sampler,
                              batch_size=args.sev_train_batch_size,
                              num_workers=args.num_workers,
                              drop_last=True,
                              pin_memory=True)
    test_loader = DataLoader(testset,
                             sampler=test_sampler,
                             batch_size=args.eval_batch_size,
                             num_workers=args.num_workers,
                             pin_memory=True) if testset is not None else None

    return train_loader, test_loader


def get_dataset(gen_file, args, cfg):
    trainset = COVID_Dataset_K_fold(gen_file, args, cfg, mode='train')
    testset = COVID_Dataset_K_fold(gen_file, args, cfg, mode='test')

    return trainset, testset


class COVID_Dataset_K_fold(data.Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self, gen_file, args, cfg, mode='train'):
        'Initialization'
        # 5-fold
        self.dim = (args.img_size, args.img_size)
        self.n_channels = 3
        self.n_classes = 3
        self.fold = args.fold
        self.mode = mode
        self.image_standard = args.image_std
        self.args =args
        self.positions = ['v7_210129_PA_AP/split_5_folds']
        self.cfg = cfg

        self.crf = pd.read_csv(args.crf_path)
        self.brixia_crf = pd.read_csv(args.brixia_crf_path)
        self.brixia_flag = []


        if self.mode == 'train':
            self.fold_ids = list(range(5)) # use testset as validation set ##+ ['test']   # Not use testset as validation set.
            # self.brixia_fold_ids = list(range(self.self_train_level))
        elif self.mode == 'val':
            self.fold_ids = ['test']
        elif self.mode == 'test':
            if self.args.test_brixia:
                self.fold_ids = ['test_brixia']
                self.brixia_crf = pd.read_csv('./consensus_testset_v1_extended.csv', sep=';')
            else:
                self.fold_ids = ['test'] + list(range(5))
                self.brixia_crf = pd.read_csv(args.brixia_crf_path)

        self.total_images_dic = {}

        if self.mode == 'train' or self.mode == 'val':

            for position in self.positions:
                self.data_position = os.path.join(args.data_dir, position)
                print(self.data_position)

                for idx in self.fold_ids:
                    if 'test' in str(idx):
                        self.data_dir = os.path.join(self.data_position, idx)
                    else:
                        self.data_dir = os.path.join(self.data_position, f'fold_{idx}')

                    self.labels = os.listdir(self.data_dir)
                    # self.labels = args.labels
                    # self.labels = ["COVID-19_pos_KNUH", "COVID-19_pos_CNUH","COVID-19_pos_YNU"]
                    for label in self.labels:

                        if self.mode == 'val' and label == 'COVID-19_Brixia':
                            continue

                        npy_dir = os.path.join(self.data_dir, label)


                        if args.idx_client == 7:
                            if label == 'COVID-19_pos_KNUH':
                                y_label = 2

                            elif label == 'COVID-19_pos_YNU':
                                y_label = 2

                            elif label == 'COVID-19_Brixia':
                                y_label = 2
                            else:  # if there are other folder which we don't want, just pass.
                                continue

                            if self.image_standard:
                                images_list = glob.glob(os.path.join(npy_dir, '*_orig.png'))
                            else:
                                images_list = glob.glob(os.path.join(npy_dir, '*_image.png'))
                            for image in images_list:
                                self.total_images_dic[image] = y_label
                                if label == 'COVID-19_Brixia':
                                    self.brixia_flag.append(1)
                                else:
                                    self.brixia_flag.append(0)

                        if args.idx_client == 8:
                            if label == 'COVID-19_pos_CNUH':
                                y_label = 2

                            elif label == 'COVID-19_pos_YNU':
                                y_label = 2

                            elif label == 'COVID-19_Brixia':
                                y_label = 2
                            else:  # if there are other folder which we don't want, just pass.
                                continue

                            if self.image_standard:
                                images_list = glob.glob(os.path.join(npy_dir, '*_orig.png'))
                            else:
                                images_list = glob.glob(os.path.join(npy_dir, '*_image.png'))
                            for image in images_list:
                                self.total_images_dic[image] = y_label
                                if label == 'COVID-19_Brixia':
                                    self.brixia_flag.append(1)
                                else:
                                    self.brixia_flag.append(0)

                        if args.idx_client == 9:
                            if label == 'COVID-19_pos_CNUH':
                                y_label = 2

                            elif label == 'COVID-19_pos_KNUH':
                                y_label = 2

                            elif label == 'COVID-19_Brixia':
                                y_label = 2
                            else:  # if there are other folder which we don't want, just pass.
                                continue

                            if self.image_standard:
                                images_list = glob.glob(os.path.join(npy_dir, '*_orig.png'))
                            else:
                                images_list = glob.glob(os.path.join(npy_dir, '*_image.png'))
                            for image in images_list:
                                self.total_images_dic[image] = y_label
                                if label == 'COVID-19_Brixia':
                                    self.brixia_flag.append(1)
                                else:
                                    self.brixia_flag.append(0)

                        if args.idx_client == 10: # Cohen External Test (영남대랑 같음)
                            if label == 'COVID-19_pos_YNU':
                                y_label = 2
                            else:  # if there are other folder which we don't want, just pass.
                                continue

                            if self.image_standard:
                                images_list = glob.glob(os.path.join(npy_dir, '*_orig.png'))
                            else:
                                images_list = glob.glob(os.path.join(npy_dir, '*_image.png'))
                            for image in images_list:
                                self.total_images_dic[image] = y_label
                                if label == 'COVID-19_Brixia':
                                    self.brixia_flag.append(1)
                                else:
                                    self.brixia_flag.append(0)

        elif self.mode == 'test':

            for position in self.positions:
                self.data_position = os.path.join(args.data_dir, position)

                for idx in self.fold_ids:
                    if 'test' in str(idx):
                        self.data_dir = os.path.join(self.data_position, idx)
                    else:
                        self.data_dir = os.path.join(self.data_position, f'fold_{idx}')

                    self.labels = os.listdir(self.data_dir)
                    # self.labels = args.labels
                    # self.labels = ["COVID-19_pos_KNUH", "COVID-19_pos_CNUH","COVID-19_pos_YNU"]
                    for label in self.labels:
                        # if self.mode == 'train' and \
                        #     label == 'COVID-19_Brixia' and \
                        #         idx not in self.brixia_fold_ids:
                        #     print(f'Brixia fold_{idx} pass!!!')
                        #     continue

                        # if args.labels != ['COVID-19_Brixia'] and \
                        #         self.mode == 'val' and \
                        #         label == 'COVID-19_Brixia':
                        #     continue

                        npy_dir = os.path.join(self.data_dir, label)

                        if args.idx_client == 7:
                            if label == 'COVID-19_pos_CNUH':
                                y_label = 2

                            else:  # if there are other folder which we don't want, just pass.
                                continue

                            if self.image_standard:
                                images_list = glob.glob(os.path.join(npy_dir, '*_orig.png'))
                            else:
                                images_list = glob.glob(os.path.join(npy_dir, '*_image.png'))
                            for image in images_list:
                                self.total_images_dic[image] = y_label
                                if label == 'COVID-19_Brixia':
                                    self.brixia_flag.append(1)
                                else:
                                    self.brixia_flag.append(0)

                        if args.idx_client == 8:
                            if label == 'COVID-19_pos_KNUH':
                                y_label = 2

                            else:  # if there are other folder which we don't want, just pass.
                                continue

                            if self.image_standard:
                                images_list = glob.glob(os.path.join(npy_dir, '*_orig.png'))
                            else:
                                images_list = glob.glob(os.path.join(npy_dir, '*_image.png'))
                            for image in images_list:
                                self.total_images_dic[image] = y_label
                                if label == 'COVID-19_Brixia':
                                    self.brixia_flag.append(1)
                                else:
                                    self.brixia_flag.append(0)

                        if args.idx_client == 9:
                            if label == 'COVID-19_pos_YNU':
                                y_label = 2

                            else:  # if there are other folder which we don't want, just pass.
                                continue

                            if self.image_standard:
                                images_list = glob.glob(os.path.join(npy_dir, '*_orig.png'))
                            else:
                                images_list = glob.glob(os.path.join(npy_dir, '*_image.png'))
                            for image in images_list:
                                self.total_images_dic[image] = y_label
                                if label == 'COVID-19_Brixia':
                                    self.brixia_flag.append(1)
                                else:
                                    self.brixia_flag.append(0)

                        if args.idx_client == 10: # Cohen External Test
                            if label == 'COVID-19_pos_Cohen':
                                y_label = 2

                            else:  # if there are other folder which we don't want, just pass.
                                continue

                            if self.image_standard:
                                images_list = glob.glob(os.path.join(npy_dir, '*_orig.png'))
                            else:
                                images_list = glob.glob(os.path.join(npy_dir, '*_image.png'))
                            for image in images_list:
                                self.total_images_dic[image] = y_label
                                if label == 'COVID-19_Brixia' or 'COVID-19_pos_Cohen':
                                    self.brixia_flag.append(1)
                                else:
                                    self.brixia_flag.append(0)


        else:
            raise AssertionError()

        print('Generator: %s' % self.mode)
        print('A total of %d image data were generated.' % len(self.total_images_dic))
        # print(f'Self-Train Level: {self.self_train_level}')
        # print(f'Brixia folds: {self.brixia_fold_ids}')
        print(f'# of Brixia data: {sum(self.brixia_flag)}')
        print()
        self.n_data = len(self.total_images_dic)
        self.classes = [i for i in range(self.n_classes)]

        # self.images_list = sorted(self.total_images_dic.keys())
        self.images_list = list(self.total_images_dic.keys())

        # calculate class weights
        y = []
        if args.idx_client == 10: # Cohen External Test
            self.brixia_crf = pd.read_csv('./public-annotations_v2.csv')
        for index, img in enumerate(self.images_list):
            img_path = self.images_list[index]
            img_id = os.path.split(img_path)[-1].split('_image')[0]
            if self.brixia_flag[index] == 1:
                if args.idx_client == 6:
                    sev = gen_brixia_sev_arr(img_id, self.brixia_crf, 'c')
                elif self.args.test_brixia:
                    sev = gen_brixia_sev_arr(img_id, self.brixia_crf, 'b_mode')
                else:
                    sev = gen_brixia_sev_arr(img_id, self.brixia_crf)
            else:
                sev = gen_sev_arr(img_id, self.crf)
            sev = torch.sum(torch.from_numpy(sev).float())
            y.append(sev)
        y = np.array(y).astype(np.int)

        # counts = np.bincount(y)
        # labels_weights = 1. / counts
        # weights = labels_weights[y]
        #
        # self.weighted_sampler = WeightedRandomSampler(weights, len(weights))

        self.gen_dic = [json.loads(l) for l in open(gen_file)][0]
        self.max_words = 120

    def __len__(self):
        'Denotes the total number of samples'
        return self.n_data

    def __getitem__(self, index):
        'Generates one sample of data'

        'Generates data containing batch_size samples'  # X : (n_samples, *dims. n_channels)
        # Generate data & Store sample
        # Assign probablity and parameters

        img_path = self.images_list[index]
        image = cv2.imread(img_path, 0)
        # image = cv2.resize(image, dsize=(1024, 1024), interpolation=cv2.INTER_LINEAR)
        # if not len(image.shape) == 2:
        #     print('find error.')
        # mask = cv2.imread(img_path.replace('orig.png', 'mask.png'), 0)
        # mask = cv2.resize(mask, dsize=(224,224))
        # mask = mask / 255

        image = Image.fromarray(image)
        # if self.mode == 'train':
        #     image = GetTransforms(image, type=self.cfg.use_transforms_type) # Aug out
        image = np.array(image)
        image = transform(image, self.cfg)  # -2~2

        # Normalize
        image = (image - image.min()) / (image.max() - image.min())

        label = self.total_images_dic[self.images_list[index]]

        img_id = os.path.split(img_path)[-1].split('_image')[0]
        if self.brixia_flag[index] == 1:
            if self.args.idx_client == 6:
                sev = gen_brixia_sev_arr(img_id, self.brixia_crf, 'c')
            elif self.args.test_brixia:
                sev = gen_brixia_sev_arr(img_id, self.brixia_crf, 'b_mode')
            else:
                sev = gen_brixia_sev_arr(img_id, self.brixia_crf)
        else:
            sev = gen_sev_arr(img_id, self.crf)
        sev = torch.from_numpy(sev).float()
        sev = sev.sum()

        # Get caption for image
        caption = self.gen_dic[img_path]['predicted']
        caption = caption.replace('[CLS]', '').replace('[SEP]', '')
        caption = pre_caption(caption, self.max_words)

        return image, caption, sev

        # return image, label, sev


def gen_sev_arr(idx, crf):
    idx = idx.replace('_orig.png', '')
    sev_arr = np.zeros((3, 2))

    loc_dic = {1: (0, 0), 2: (1, 0), 3: (2, 0), 4: (0, 1), 5: (1, 1), 6: (2, 1)}

    loc_list = crf.loc[crf.file_name == idx + '.dcm', ["CXR_location"]].values
    if len(loc_list) == 0:
        return -np.ones((3, 2))
    else:
        loc = loc_list[0][0]
        loc = list(map(int, str(loc)))
        for x in loc:
            if x != 0:
                sev_arr[loc_dic[x]] = 1
        return sev_arr


def gen_brixia_sev_arr(idx, brixia_crf, src='a'):
    # Load brixia score # No overlap

    brix_arr = np.zeros((3, 2))
    idx = idx.replace('_orig.png', '')
    if src == 'a':
        dcm = idx + '.dcm'
        df = brixia_crf[brixia_crf.Filename == dcm]
        brix = df.BrixiaScore.values.item()
        str_brix = list(str(brix).zfill(6))

        brix_arr = np.array(list(map(int, str_brix))).reshape(2, 3).T

    elif src == 'b_mean':
        dcm = idx + '.dcm'
        df = brixia_crf[brixia_crf.Filename == dcm]
        a = df.MeanA.values.item()
        b = df.MeanB.values.item()
        c = df.MeanC.values.item()
        d = df.MeanD.values.item()
        e = df.MeanE.values.item()
        f = df.MeanF.values.item()
        brix_arr = np.array([[a, d],
                             [b, e],
                             [c, f]])

    elif src == 'b_mode':
        dcm = idx + '.dcm'
        df = brixia_crf[brixia_crf.Filename == dcm]
        a = df.ModeA.values.item()
        b = df.ModeB.values.item()
        c = df.ModeC.values.item()
        d = df.ModeD.values.item()
        e = df.ModeE.values.item()
        f = df.ModeF.values.item()
        brix_arr = np.array([[a, d],
                             [b, e],
                             [c, f]])

    elif src == 'c':
        file = idx
        df = brixia_crf[brixia_crf.filename == file]
        a = df['S-A'].values.item() + df['J-A'].values.item()
        b = df['S-B'].values.item() + df['J-B'].values.item()
        c = df['S-C'].values.item() + df['J-C'].values.item()
        d = df['S-D'].values.item() + df['J-D'].values.item()
        e = df['S-E'].values.item() + df['J-E'].values.item()
        f = df['S-F'].values.item() + df['J-F'].values.item()

        brix_arr = np.array([[a, d],
                             [b, e],
                             [c, f]]) / 2

    """
    brixia_arr
    1st idx: 0=upper, 1=middle, 2=lower
    2nd idx: 0=left, 1=right
    """
    brix_arr[brix_arr > 0] = 1

    return brix_arr


def gen_sev_map(sev_arr, m):
    # sev_label : (3,2), m: (256,256)
    if sev_arr.max() < 0:
        return np.zeros_like(m).astype(float)

    # Load mask
    both, left, right = only_lung(m)

    # Calculate the line
    both_max = both.nonzero()[0][-1]
    both_min = both.nonzero()[0][0]

    both_3_12 = int(3 * (both_max - both_min) // 12)
    both_4_12 = int(4 * (both_max - both_min) // 12)
    both_5_12 = int(5 * (both_max - both_min) // 12)

    both_7_12 = int(7 * (both_max - both_min) // 12)
    both_8_12 = int(8 * (both_max - both_min) // 12)
    both_9_12 = int(9 * (both_max - both_min) // 12)

    left_map = np.zeros_like(both).astype(np.float32)
    right_map = np.zeros_like(both).astype(np.float32)
    if len(left.nonzero()[0]) > 0:
        left_max = left.nonzero()[0][-1]
        left_min = left.nonzero()[0][0]

        # Label on the mask
        left_map[left_min:both_5_12] = sev_arr[0][0]
        left_map[both_5_12:both_8_12] = sev_arr[1][0]
        left_map[both_8_12:left_max] = sev_arr[2][0]
        left_map *= left

    if len(right.nonzero()[0]) > 0:
        right_max = right.nonzero()[0][-1]
        right_min = right.nonzero()[0][0]

        right_map[right_min:both_5_12] = sev_arr[0][1]
        right_map[both_5_12:both_8_12] = sev_arr[1][1]
        right_map[both_8_12:right_max] = sev_arr[2][1]
        right_map *= right

    sev_map = left_map + right_map

    return sev_map


def only_lung(mask, blur=False):
    if blur:
        m = cv2.blur(mask.copy(), ksize=(200, 200))
    else:
        m = mask.copy()
    ret, markers = cv2.connectedComponents(m)
    npoints = []
    for i in range(ret):
        npoints.append(len(markers[markers == i]))

    sorted_labels = np.argsort(npoints)

    half1 = np.zeros_like(m)
    half2 = np.zeros_like(m)
    if len(sorted_labels) >= 3:
        half1[markers == sorted_labels[-2]] = 1
        half2[markers == sorted_labels[-3]] = 1

        flag1 = half1.nonzero()[1].max()
        flag2 = half2.nonzero()[1].max()

        if flag1 > flag2:
            left = half2  # right lung in reality
            right = half1  # left lung in reality
        else:
            left = half1
            right = half2

    else:
        half1[markers == sorted_labels[-2]] = 1
        flag_max = half1.nonzero()[1].max()
        flag_min = half1.nonzero()[1].min()
        if half1[:, flag_min].argmax() > half1[:, flag_max].argmax():
            left = half1
            right = half2
        else:
            left = half2
            right = half1

    both = left + right

    return both, left, right