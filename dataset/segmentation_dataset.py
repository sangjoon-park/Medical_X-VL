
import logging
import pandas as pd
import os, os.path
from torch.utils import data
import os
import cv2
import albumentations as albu
import numpy as np

import torch
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler
from albumentations.pytorch.transforms import ToTensor
import json
from dataset.utils import pre_caption, pre_question

logger = logging.getLogger(__name__)

def get_train_val_loader_SIIM(args, config, train_config, fold_id=0):

    train_transform = albu.load(train_config['TRAIN_TRANSFORMS'])
    valid_transform = albu.load(train_config['VALID_TRANSFORMS'])

    non_empty_mask_proba = train_config.get('NON_EMPTY_MASK_PROBA', 0)
    use_sampler = train_config['USE_SAMPLER']

    folds_distr_path = train_config['FOLD']['FILE']
    n_folds = train_config['FOLD']['NUMBER']

    # dataset_folder = train_config['DATA_DIRECTORY']
    dataset_folder = os.path.join(os.path.join(args.data_dir, 'SIIM_ACR'), 'dataset512')
    num_workers = args.num_workers
    train_batch_size = args.train_batch_size
    eval_batch_size = args.eval_batch_size

    train_dataset = PneumothoraxDataset(gen_file=config['train_gen'],
        data_folder=dataset_folder, mode='train', args=args,
        transform=train_transform, fold_index=fold_id,
        folds_distr_path=folds_distr_path,
    )
    train_sampler = PneumoSampler(folds_distr_path, fold_id, non_empty_mask_proba, args=args)
    if use_sampler:
        train_dataloader = DataLoader(
            dataset=train_dataset, batch_size=train_batch_size,
            num_workers=num_workers, sampler=train_sampler, drop_last=True
        )
    else:
        train_dataloader = DataLoader(
            dataset=train_dataset, batch_size=train_batch_size,
            num_workers=num_workers, shuffle=True, drop_last=True
        )

    valid_dataset = PneumothoraxDataset(gen_file=config['val_gen'],
        data_folder=dataset_folder, mode='val', args=args,
        transform=valid_transform, fold_index=str(fold_id),
        folds_distr_path=folds_distr_path,
    )
    valid_dataloader = DataLoader(
        dataset=valid_dataset, batch_size=eval_batch_size,
        num_workers=num_workers, shuffle=False, drop_last=True
    )


    return train_dataloader, valid_dataloader


def get_test_loader_SIIM(args, inference_config):
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    if args.local_rank == 0:
        torch.distributed.barrier()

    # dataset_folder = train_config['DATA_DIRECTORY']
    dataset_folder = os.path.join(os.path.join(args.data_dir, 'SIIM_ACR'), 'dataset512')
    num_workers = args.num_workers
    eval_batch_size = args.eval_batch_size

    transform = albu.load(inference_config['TEST_TRANSFORMS'])

    dataset = PneumothoraxDataset(
        data_folder=dataset_folder, mode='test',
        transform=transform,
        args=args
    )
    dataloader = DataLoader(
        dataset=dataset, batch_size=eval_batch_size,
        num_workers=num_workers, shuffle=False
    )
    return dataloader, len(dataset)


class PneumothoraxDataset(data.Dataset):
    def __init__(self, gen_file, data_folder, mode, args, transform=None,
                 fold_index=None, folds_distr_path=None):

        self.transform = transform
        self.mode = mode
        self.args = args

        # change to your path
        self.train_image_path = '{}/train/'.format(data_folder)
        self.train_mask_path = '{}/mask/'.format(data_folder)
        self.test_image_path = '{}/test/'.format(data_folder)

        self.fold_index = None
        self.folds_distr_path = folds_distr_path
        self.set_mode(mode, fold_index)
        self.to_tensor = ToTensor()

        self.gen_dic = [json.loads(l) for l in open(gen_file)][0]

    def set_mode(self, mode, fold_index):
        self.mode = mode
        self.fold_index = fold_index

        if self.mode == 'train':
            folds = pd.read_csv(self.folds_distr_path)
            folds.fold = folds.fold.astype(str)
            folds = folds[folds.fold != fold_index]

            self.total_samples = folds.fname.values.tolist()

            self.train_list = folds.fname.values.tolist()
            self.exist_labels = folds.exist_labels.values.tolist()

            # if self.args.idx_client == 0:
            #     self.train_list = folds.fname.values.tolist()[:len(self.total_samples) // 2]
            #     self.exist_labels = folds.exist_labels.values.tolist()[:len(self.total_samples) // 2]
            # elif self.args.idx_client == 1:
            #     self.train_list = folds.fname.values.tolist()[len(self.total_samples) // 2:]
            #     self.exist_labels = folds.exist_labels.values.tolist()[len(self.total_samples) // 2:]

            self.num_data = len(self.train_list)
            print(f'Total training data : {self.num_data}')

        elif self.mode == 'val':
            folds = pd.read_csv(self.folds_distr_path)
            folds.fold = folds.fold.astype(str)
            folds = folds[folds.fold == fold_index]

            self.val_list = folds.fname.values.tolist()
            self.num_data = len(self.val_list)
            print(f'Total validation data : {self.num_data}')

        elif self.mode == 'test':
            self.test_list = sorted(os.listdir(self.test_image_path))
            self.num_data = len(self.test_list)

    def __getitem__(self, index):
        if self.fold_index is None and self.mode != 'test':
            print('WRONG!!!!!!! fold index is NONE!!!!!!!!!!!!!!!!!')
            return

        if self.mode == 'test':
            img_path = os.path.join(self.test_image_path, self.test_list[index])
            image = cv2.imread(img_path, 1)
            if self.transform:
                sample = {"image": image}
                sample = self.transform(**sample)
                sample = self.to_tensor(**sample)
                image = sample['image']
            image_id = self.test_list[index].replace('.png', '')
            return image_id, image

        elif self.mode == 'train':
            img_path = os.path.join(self.train_image_path, self.train_list[index])
            image = cv2.imread(img_path, 1)
            if self.exist_labels[index] == 0:
                label = np.zeros((512, 512))  # from (1024, 1024) to (512,512)
            else:
                label = cv2.imread(os.path.join(self.train_mask_path, self.train_list[index]), 0)

        elif self.mode == 'val':
            img_path = os.path.join(self.train_image_path, self.val_list[index])
            image = cv2.imread(img_path, 1)
            label = cv2.imread(os.path.join(self.train_mask_path, self.val_list[index]), 0)

        if self.transform:
            sample = {"image": image, "mask": label}
            sample = self.transform(**sample)
            sample = self.to_tensor(**sample)
            image, label = sample['image'], sample['mask']

        caption = self.gen_dic[img_path]['predicted']
        caption = caption.replace('[CLS]', '').replace('[SEP]', '')
        caption = pre_caption(caption, self.max_words)

        return image, caption, label

    def __len__(self):
        return self.num_data



class PneumoSampler(data.sampler.Sampler):
    def __init__(self, folds_distr_path, fold_index, demand_non_empty_proba, args):
        assert demand_non_empty_proba > 0, 'frequensy of non-empty images must be greater then zero'
        self.fold_index = fold_index
        self.positive_proba = demand_non_empty_proba

        self.folds = pd.read_csv(folds_distr_path)
        self.folds.fold = self.folds.fold.astype(str)
        self.folds = self.folds[self.folds.fold != fold_index].reset_index(drop=True)

        self.positive_idxs = self.folds[self.folds.exist_labels == 1].index.values
        self.negative_idxs = self.folds[self.folds.exist_labels == 0].index.values

        # if args.idx_client == 0:
        #     self.folds = self.folds.iloc[: len(self.folds) // 2]
        #     self.positive_idxs = self.folds[self.folds.exist_labels == 1].index.values
        #     self.negative_idxs = self.folds[self.folds.exist_labels == 0].index.values
        # elif args.idx_client == 1:
        #     self.folds = self.folds.iloc[: len(self.folds) // 2]
        #     self.positive_idxs = self.folds[self.folds.exist_labels == 1].index.values
        #     self.negative_idxs = self.folds[self.folds.exist_labels == 0].index.values

        self.n_positive = self.positive_idxs.shape[0]
        self.n_negative = int(self.n_positive * (1 - self.positive_proba) / self.positive_proba)

    def __iter__(self):
        negative_sample = np.random.choice(self.negative_idxs, size=self.n_negative)
        shuffled = np.random.permutation(np.hstack((negative_sample, self.positive_idxs)))
        return iter(shuffled.tolist())

    def __len__(self):
        return self.n_positive + self.n_negative