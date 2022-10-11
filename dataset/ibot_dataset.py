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
from dataset.utils import pre_caption, pre_question
import torch
import glob


class ImageFolderMask(Dataset):
    def __init__(self, task, ann_file, patch_size, pred_ratio, pred_ratio_var, pred_aspect_ratio,
                 pred_shape='block', transforms=None, pred_start_epoch=0, max_words=120):
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
        train_data = [json.loads(l) for l in open(ann_file[0])]

        # Train data
        if 'openi' in ann_file[0]:
            train_list = []
            for data in train_data:
                _, label, txt, img = data.keys()
                d_txt = data[txt]
                d_img = data[img].replace('/home/data_storage/mimic-cxr/dataset/',
                                          '/COVID_8TB/sangjoon/vision_language/')
                d_label = data[label].replace("'", "").split(',')
                train_list.append([d_img, d_txt, d_label])
            self.train_df = pd.DataFrame(train_list, columns=['image', 'caption', 'label'])
        else:
            train_list = []
            for data in train_data:
                _, _, label, txt, img = data.keys()
                d_txt = data[txt]
                d_img = data[img].replace('/home/mimic-cxr/dataset/image_preprocessing/re_512_3ch/',
                                          '/COVID_8TB/sangjoon/vision_language/mimic_dset/re_512_3ch/')
                d_label = data[label].replace("'", "").split(',')
                train_list.append([d_img, d_txt, d_label])
            self.train_df = pd.DataFrame(train_list, columns=['image', 'caption', 'label'])

        # Make into dictionary
        self.ann = []
        for sub_idx in range(len(self.train_df)):
            subject = self.train_df.iloc[sub_idx]
            label = subject['label']
            if task == 'pretrain':
                # captions = [subject['caption']]
                captions = subject['caption'].split('.')
                total_captions = []
                for caption in captions:
                    if len(caption) < 3:
                        continue
                    total_captions.append(caption)
                len_sentence = len(total_captions)
                len_choice = min(len_sentence, 10)
                selected_captions = random.sample(total_captions, len_choice)
                selected_captions = '.'.join(selected_captions)
            elif task == 'generation':
                selected_captions = subject['caption']

            if len(selected_captions) == 0:
                continue
            self.ann.append({'image': subject['image'], 'caption': selected_captions})

        self.max_words = max_words
        self.transforms = transforms

    def __len__(self):
        return len(self.ann)

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

    def __getitem__(self, index):
        ann = self.ann[index]

        image_path = self.ann[index]['image']
        image = Image.open(image_path).convert('RGB')
        output = self.transforms(image)  # jinyu

        caption = ann['caption'].replace('.', '. ')
        caption = pre_caption(caption, self.max_words) + '.'

        masks = []
        for idx, img in enumerate(output):
            if idx == 0:
                try:
                    H, W = img.shape[1] // self.psz, img.shape[2] // self.psz
                except:
                    # skip non-image
                    continue
                mask = np.zeros((H, W), dtype=bool)
            else:
                try:
                    H, W = img.shape[1] // self.psz, img.shape[2] // self.psz
                except:
                    # skip non-image
                    continue

                high = self.get_pred_ratio() * H * W

                if self.pred_shape == 'block':
                    # following BEiT (https://arxiv.org/abs/2106.08254), see at
                    # https://github.com/microsoft/unilm/blob/b94ec76c36f02fb2b0bf0dcb0b8554a2185173cd/beit/masking_generator.py#L55
                    mask = np.zeros((H, W), dtype=bool)
                    mask_count = 0
                    while mask_count < high:
                        max_mask_patches = high - mask_count

                        delta = 0
                        for attempt in range(10):
                            low = (min(H, W) // 3) ** 2
                            target_area = random.uniform(low, max_mask_patches)
                            aspect_ratio = math.exp(random.uniform(*self.log_aspect_ratio))
                            h = int(round(math.sqrt(target_area * aspect_ratio)))
                            w = int(round(math.sqrt(target_area / aspect_ratio)))
                            if w < W and h < H:
                                top = random.randint(0, H - h)
                                left = random.randint(0, W - w)

                                num_masked = mask[top: top + h, left: left + w].sum()
                                if 0 < h * w - num_masked <= max_mask_patches:
                                    for i in range(top, top + h):
                                        for j in range(left, left + w):
                                            if mask[i, j] == 0:
                                                mask[i, j] = 1
                                                delta += 1

                            if delta > 0:
                                break

                        if delta == 0:
                            break
                        else:
                            mask_count += delta

                elif self.pred_shape == 'rand':
                    mask = np.hstack([
                        np.zeros(H * W - int(high)),
                        np.ones(int(high)),
                    ]).astype(bool)
                    np.random.shuffle(mask)
                    mask = mask.reshape(H, W)

                else:
                    # no implementation
                    assert False

            masks.append(mask)

        return output, caption, masks, ann['image']


class Re_eval_ImageFolder(Dataset):
    def __init__(self, ann_file, patch_size, transforms=None, max_words=120):
        self.psz = patch_size

        # caption label and etc.
        train_data = [json.loads(l) for l in open(ann_file)]

        # Train data
        if 'openi' in ann_file:
            train_list = []
            for data in train_data:
                _, label, txt, img = data.keys()
                d_txt = data[txt]
                d_img = data[img].replace('/home/data_storage/mimic-cxr/dataset/',
                                          '/COVID_8TB/sangjoon/vision_language/')
                d_label = data[label].replace("'", "").split(',')
                train_list.append([d_img, d_txt, d_label])
            self.train_df = pd.DataFrame(train_list, columns=['image', 'caption', 'label'])
        else:
            train_list = []
            for data in train_data:
                _, _, label, txt, img = data.keys()
                d_txt = data[txt]
                d_img = data[img].replace('/home/mimic-cxr/dataset/image_preprocessing/re_512_3ch/',
                                          '/COVID_8TB/sangjoon/vision_language/mimic_dset/re_512_3ch/')
                d_label = data[label].replace("'", "").split(',')
                train_list.append([d_img, d_txt, d_label])
            self.train_df = pd.DataFrame(train_list, columns=['image', 'caption', 'label'])

        # Make into dictionary
        self.ann = []
        for sub_idx in range(len(self.train_df)):
            subject = self.train_df.iloc[sub_idx]
            captions = subject['caption'].split('.')
            label = subject['label']
            # captions = [subject['caption']]
            total_captions = []
            for caption in captions:
                if len(caption) < 3:
                    continue
                total_captions.append(caption)
            len_sentence = len(total_captions)
            len_choice = min(len_sentence, 10)
            selected_captions = random.sample(total_captions, len_choice)
            selected_captions = '.'.join(selected_captions)

            if len(selected_captions) == 0:
                continue
            self.ann.append({'image': subject['image'], 'caption': selected_captions, 'label': label})

        self.max_words = max_words
        self.transforms = transforms

        self.text = []
        self.image = []
        self.label = []
        self.txt2img = {}
        self.img2txt = {}

        txt_id = 0
        for img_id, ann in enumerate(self.ann):
            self.image.append(ann['image'])
            self.label.append(ann['label'])
            self.img2txt[img_id] = []
            caption = ann['caption']
            caption = caption.replace('.', '. ')
            self.text.append(pre_caption(caption, self.max_words) + '.')
            self.img2txt[img_id].append(txt_id)
            self.txt2img[txt_id] = img_id
            txt_id += 1

    def __len__(self):
        return len(self.image)

    def __getitem__(self, index):
        image_path = self.ann[index]['image']
        image = Image.open(image_path).convert('RGB')
        output = self.transforms(image)  # jinyu

        caption = self.ann[index]['caption'].replace('.', '. ')
        caption = pre_caption(caption, self.max_words) + '.'

        return output, index, self.ann[index]['image'], caption


class Re_train_ImageFolder(Dataset):
    def __init__(self, ann_file, patch_size, transforms=None, max_words=120):
        self.psz = patch_size

        # caption label and etc.
        train_data = [json.loads(l) for l in open(ann_file[0])]

        # Train data
        if 'openi' in ann_file[0]:
            train_list = []
            for data in train_data:
                _, label, txt, img = data.keys()
                d_txt = data[txt]
                d_img = data[img].replace('/home/data_storage/mimic-cxr/dataset/',
                                          '/COVID_8TB/sangjoon/vision_language/')
                d_label = data[label].replace("'", "").split(',')
                train_list.append([d_img, d_txt, d_label])
            self.train_df = pd.DataFrame(train_list, columns=['image', 'caption', 'label'])
        else:
            train_list = []
            for data in train_data:
                _, _, label, txt, img = data.keys()
                d_txt = data[txt]
                d_img = data[img].replace('/home/mimic-cxr/dataset/image_preprocessing/re_512_3ch/',
                                          '/COVID_8TB/sangjoon/vision_language/mimic_dset/re_512_3ch/')
                d_label = data[label].replace("'", "").split(',')
                train_list.append([d_img, d_txt, d_label])
            self.train_df = pd.DataFrame(train_list, columns=['image', 'caption', 'label'])

        # Make into dictionary
        self.ann = []
        for sub_idx in range(len(self.train_df)):
            subject = self.train_df.iloc[sub_idx]
            captions = subject['caption'].split('.')
            label = subject['label']
            # captions = [subject['caption']]
            total_captions = []
            for caption in captions:
                if len(caption) < 3:
                    continue
                total_captions.append(caption)
            len_sentence = len(total_captions)
            len_choice = min(len_sentence, 10)
            selected_captions = random.sample(total_captions, len_choice)
            selected_captions = '.'.join(selected_captions)

            if len(selected_captions) == 0:
                continue
            self.ann.append({'image': subject['image'], 'caption': selected_captions, 'image_id': sub_idx, 'label': label})

        self.transforms = transforms
        self.max_words = max_words
        self.img_ids = {}

        n = 0
        for ann in self.ann:
            img_id = ann['image_id']
            if img_id not in self.img_ids.keys():
                self.img_ids[img_id] = n
                n += 1

        txt_id = 0
        self.text = []
        for img_id, ann in enumerate(self.ann):
            caption = ann['caption']
            caption = caption.replace('.', '. ')
            self.text.append(pre_caption(caption, self.max_words) + '.')
            txt_id += 1

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):
        image_path = self.ann[index]['image']
        image = Image.open(image_path).convert('RGB')
        output = self.transforms(image)  # jinyu

        caption = self.ann[index]['caption'].replace('.', '. ')
        caption = pre_caption(caption, self.max_words) + '.'

        return output, caption, self.img_ids[self.ann[index]['image_id']], self.ann[index]['image']


class Gen_eval_ImageFolder(Dataset):
    def __init__(self, ann_file, patch_size, transforms=None, max_words=120):
        self.psz = patch_size

        # caption label and etc.
        train_data = [json.loads(l) for l in open(ann_file)]

        # Train data
        train_list = []
        for data in train_data:
            _, _, label, txt, img = data.keys()
            d_txt = data[txt]
            d_img = data[img].replace('/home/mimic-cxr/dataset/image_preprocessing/re_512_3ch/',
                                      '/COVID_8TB/sangjoon/vision_language/mimic_dset/re_512_3ch/')
            d_label = data[label].replace("'", "").split(',')
            train_list.append([d_img, d_txt, d_label])
        self.train_df = pd.DataFrame(train_list, columns=['image', 'caption', 'label'])

        # Make into dictionary
        self.ann = []
        for sub_idx in range(len(self.train_df)):
            subject = self.train_df.iloc[sub_idx]
            captions = subject['caption'].split('.')
            label = subject['label']
            # captions = [subject['caption']]
            total_captions = []
            for caption in captions:
                if len(caption) < 3:
                    continue
                total_captions.append(caption)
            len_sentence = len(total_captions)
            len_choice = min(len_sentence, 10)
            selected_captions = random.sample(total_captions, len_choice)
            selected_captions = '.'.join(selected_captions)

            if len(selected_captions) == 0:
                continue
            self.ann.append({'image': subject['image'], 'caption': selected_captions, 'label': label})

        self.max_words = max_words
        self.transforms = transforms

        self.text = []
        self.image = []
        self.label = []
        self.txt2img = {}
        self.img2txt = {}

        txt_id = 0
        for img_id, ann in enumerate(self.ann):
            self.image.append(ann['image'])
            self.label.append(ann['label'])
            self.img2txt[img_id] = []
            caption = ann['caption']
            caption = caption.replace('.', '. ')
            self.text.append(pre_caption(caption, self.max_words))
            self.img2txt[img_id].append(txt_id)
            self.txt2img[txt_id] = img_id
            txt_id += 1

    def __len__(self):
        return len(self.image)

    def __getitem__(self, index):
        image_path = self.ann[index]['image']
        image = Image.open(image_path).convert('RGB')
        output = self.transforms(image)  # jinyu

        caption = self.ann['caption'][index].replace('.', '. ')
        caption = pre_caption(caption, self.max_words)

        return output, caption


class Gen_train_ImageFolder(Dataset):
    def __init__(self, ann_file, patch_size, transforms=None, max_words=120):
        self.psz = patch_size

        # caption label and etc.
        train_data = [json.loads(l) for l in open(ann_file[0])]

        # Train data
        train_list = []
        for data in train_data:
            _, _, label, txt, img = data.keys()
            d_txt = data[txt]
            d_img = data[img].replace('/home/mimic-cxr/dataset/image_preprocessing/re_512_3ch/',
                                      '/COVID_8TB/sangjoon/vision_language/mimic_dset/re_512_3ch/')
            d_label = data[label].replace("'", "").split(',')
            train_list.append([d_img, d_txt, d_label])
        self.train_df = pd.DataFrame(train_list, columns=['image', 'caption', 'label'])

        # Make into dictionary
        self.ann = []
        for sub_idx in range(len(self.train_df)):
            subject = self.train_df.iloc[sub_idx]
            captions = subject['caption'].split('.')
            label = subject['label']
            # captions = [subject['caption']]
            total_captions = []
            for caption in captions:
                if len(caption) < 3:
                    continue
                total_captions.append(caption)
            len_sentence = len(total_captions)
            len_choice = min(len_sentence, 10)
            selected_captions = random.sample(total_captions, len_choice)
            selected_captions = '.'.join(selected_captions)

            if len(selected_captions) == 0:
                continue
            self.ann.append({'image': subject['image'], 'caption': selected_captions, 'image_id': sub_idx, 'label': label})

        self.transforms = transforms
        self.max_words = max_words
        self.img_ids = {}

        n = 0
        for ann in self.ann:
            img_id = ann['image_id']
            if img_id not in self.img_ids.keys():
                self.img_ids[img_id] = n
                n += 1

        txt_id = 0
        self.text = []
        for img_id, ann in enumerate(self.ann):
            caption = ann['caption']
            caption = caption.replace('.', '. ')
            self.text.append(pre_caption(caption, self.max_words))
            txt_id += 1

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):
        image_path = self.ann[index]['image']
        image = Image.open(image_path).convert('RGB')
        output = self.transforms(image)  # jinyu

        caption = self.ann[index]['caption'].replace('.', '. ')
        caption = pre_caption(caption, self.max_words)

        return output, caption


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

                # Unify the words with same meaning
                if answer == 'pa':
                    answer = 'posterior anterior'
                elif answer == 'x ray':
                    answer = 'x-ray'
                elif answer == 'xray':
                    answer = 'x-ray'
                elif answer == 'xr':
                    answer = 'x-ray'
                elif answer == 'plain film x ray':
                    answer = 'x-ray'
                elif answer == 'plain film':
                    answer = 'x-ray'
                elif answer == 't2 mri':
                    answer = 't2 weighted'
                elif answer == 'mr flair':
                    answer = 'flair'

                if not answer in self.answer_list:
                    self.answer_list.append(answer)
            print('fond')

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):

        ann = self.ann[index]

        image = Image.open(
            '/COVID_8TB/sangjoon/vision_language/data_RAD/home/mimic-cxr/dataset/data_RAD/images/' + ann['image_name']).convert('RGB')
        image = image.resize((224, 224))
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


class Gen_Vqa_dataset(Dataset):
    def __init__(self, ann_file, transforms=None, eos='[SEP]', split='train', max_ques_words=120):
        self.split = split
        self.max_ques_words = max_ques_words

        # caption label and etc.
        train_data = [json.loads(l) for l in open(ann_file[0])]

        # Train data
        if 'openi' in ann_file[0]:
            train_list = []
            for data in train_data:
                _, label, txt, img = data.keys()
                d_txt = data[txt]
                d_img = data[img].replace('/home/data_storage/mimic-cxr/dataset/',
                                          '/COVID_8TB/sangjoon/vision_language/')
                d_label = data[label].replace("'", "").split(',')
                train_list.append([d_img, d_txt, d_label])
            self.train_df = pd.DataFrame(train_list, columns=['image', 'caption', 'label'])
        else:
            train_list = []
            for data in train_data:
                _, _, label, txt, img = data.keys()
                d_txt = data[txt]
                d_img = data[img].replace('/home/mimic-cxr/dataset/image_preprocessing/re_512_3ch/',
                                          '/COVID_8TB/sangjoon/vision_language/mimic_dset/re_512_3ch/')
                d_label = data[label].replace("'", "").split(',')
                train_list.append([d_img, d_txt, d_label])
            self.train_df = pd.DataFrame(train_list, columns=['image', 'caption', 'label'])

        # Make into dictionary
        self.ann = []
        for sub_idx in range(len(self.train_df)):
            subject = self.train_df.iloc[sub_idx]
            label = subject['label']
            selected_captions = subject['caption']
            self.ann.append({'image': subject['image'], 'caption': selected_captions, 'image_id': sub_idx, 'label': label})

        if self.split == 'test':
            self.answer_list = []
            for ans in self.ann:
                answer = pre_caption(ans['caption'], self.max_ques_words) + '.'
                if not answer in self.answer_list:
                    self.answer_list.append(answer)

        self.transforms = transforms
        self.img_ids = {}

        self.eos = eos

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):

        ann = self.ann[index]
        image_path = ann['image']
        image = Image.open(image_path).convert('RGB')
        output = self.transforms(image)  # jinyu

        if self.split == 'test':
            caption = ann['caption'].replace('.', '. ')
            caption = pre_caption(caption, self.max_ques_words)

            return output, caption, image_path

        elif self.split == 'train':

            caption = ann['caption'].replace('.', '. ')
            caption = pre_caption(caption, self.max_ques_words)

            ann_all = [caption for x in range(10)]

            answer_weight = {}
            for answer in ann_all:
                if answer in answer_weight.keys():
                    answer_weight[answer] += 1 / len(ann_all)
                else:
                    answer_weight[answer] = 1 / len(ann_all)

            answers = list(answer_weight.keys())
            weights = list(answer_weight.values())

            return output, answers, weights

class Cls_COVID(Dataset):
    def __init__(self, ann_file, partial, patch_size, transforms=None, max_words=100):
        self.psz = patch_size
        self.partial = partial

        self.train_data = glob.glob(ann_file + '**/*_orig.png', recursive=True)

        if self.partial and 'train' in ann_file:
            len_data = int(0.04 * len(self.train_data))
            self.train_data = random.sample(self.train_data, len_data)
        else:
            pass

        self.max_words = max_words
        self.transforms = transforms

        print('number of data: {}'.format(len(self.train_data)))

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, index):
        image_path = self.train_data[index]
        image = Image.open(image_path).convert('RGB')
        output = self.transforms(image)  # jinyu

        # make label
        labels = [0]

        if 'COVID-19' in image_path:
            labels[0] = 1

        labels = np.array(labels)
        labels = torch.from_numpy(labels)

        return output, labels.float()


class Det_eval_ImageFolder(Dataset):
    def __init__(self, mode, ann_file, patch_size, transforms=None, max_words=120):
        self.psz = patch_size
        self.mode = mode

        # caption label and etc.
        train_data = [json.loads(l) for l in open(ann_file)]

        # Train data
        train_list = []
        for data in train_data:
            _, _, label, txt, img = data.keys()
            d_txt = data[txt]
            d_img = data[img].replace('/home/mimic-cxr/dataset/image_preprocessing/re_512_3ch/',
                                      '/COVID_8TB/sangjoon/vision_language/mimic_dset/re_512_3ch/')
            d_label = data[label].replace("'", "").split(',')
            train_list.append([d_img, d_txt, d_label])
        self.train_df = pd.DataFrame(train_list, columns=['image', 'caption', 'label'])

        # Make into dictionary
        self.ann = []
        for sub_idx in range(len(self.train_df)):
            subject = self.train_df.iloc[sub_idx]
            captions = subject['caption'].split('.')
            label = subject['label']
            selected_captions = subject['caption']

            if len(selected_captions) == 0:
                continue
            self.ann.append({'image': subject['image'], 'caption': selected_captions, 'label': label})

        self.max_words = max_words
        self.transforms = transforms

        self.text = []
        self.image = []
        self.label = []
        self.txt2img = {}
        self.img2txt = {}

        txt_id = 0
        for img_id, ann in enumerate(self.ann):
            self.image.append(ann['image'])
            self.label.append(ann['label'])
            self.img2txt[img_id] = []
            caption = ann['caption']
            caption = caption.replace('.', '. ')
            self.text.append(pre_caption(caption, self.max_words) + '.')
            self.img2txt[img_id].append(txt_id)
            self.txt2img[txt_id] = img_id
            txt_id += 1

    def __len__(self):
        return len(self.image)

    def __getitem__(self, index):

        image_path = self.ann[index]['image']
        image = Image.open(image_path).convert('RGB')
        output = self.transforms(image)

        caption = self.ann[index]['caption'].replace('.', '. ')
        caption = pre_caption(caption, self.max_words)
        true_caption = caption
        label = 0

        if self.mode == 'patient_mismatch':
            # Detection mode: different patient
            cor = random.random()
            if cor >= 0.99: # corruption probability (1%)
                selected_idx = random.randint(0, len(self.ann))
                caption = self.ann[selected_idx]['caption'].replace('.', '. ')
                caption = pre_caption(caption, self.max_words) + '.'
                label = 1

        elif self.mode == 'orientation_error':
            # Detection mode: right-left confusion
            error_list = ['right', 'left']
            available_errors = []
            for err in error_list:
                if err in caption:
                    # if not 'no ' + err in caption:
                    available_errors.append(err)
            if len(available_errors) == 0:
                error = None
            else:
                cor = random.random()
                if cor >= 0.95: # corruption probability (5%)
                    error = random.sample(available_errors, 1)[0]
                else:
                    error = None

            if not error == None:
                if error == 'right':
                    caption = caption.replace('right', 'left')
                    label = 1
                elif error == 'left':
                    caption = caption.replace('left', 'right')
                    label = 1

        return output, image_path, label, caption, true_caption


class Det_train_ImageFolder(Dataset):
    def __init__(self, ann_file, patch_size, transforms=None, max_words=120):
        self.psz = patch_size

        # caption label and etc.
        train_data = [json.loads(l) for l in open(ann_file[0])]

        # Train data
        train_list = []
        for data in train_data:
            _, _, label, txt, img = data.keys()
            d_txt = data[txt]
            d_img = data[img].replace('/home/mimic-cxr/dataset/image_preprocessing/re_512_3ch/',
                                      '/COVID_8TB/sangjoon/vision_language/mimic_dset/re_512_3ch/')
            d_label = data[label].replace("'", "").split(',')
            train_list.append([d_img, d_txt, d_label])
        self.train_df = pd.DataFrame(train_list, columns=['image', 'caption', 'label'])

        # Make into dictionary
        self.ann = []
        for sub_idx in range(len(self.train_df)):
            subject = self.train_df.iloc[sub_idx]
            captions = subject['caption'].split('.')
            label = subject['label']
            # captions = [subject['caption']]
            total_captions = []
            for caption in captions:
                if len(caption) < 3:
                    continue
                total_captions.append(caption)
            len_sentence = len(total_captions)
            len_choice = min(len_sentence, 10)
            selected_captions = random.sample(total_captions, len_choice)
            selected_captions = '.'.join(selected_captions)

            if len(selected_captions) == 0:
                continue
            self.ann.append({'image': subject['image'], 'caption': selected_captions, 'image_id': sub_idx, 'label': label})

        self.transforms = transforms
        self.max_words = max_words
        self.img_ids = {}

        n = 0
        for ann in self.ann:
            img_id = ann['image_id']
            if img_id not in self.img_ids.keys():
                self.img_ids[img_id] = n
                n += 1

        txt_id = 0
        self.text = []
        for img_id, ann in enumerate(self.ann):
            caption = ann['caption']
            caption = caption.replace('.', '. ')
            self.text.append(pre_caption(caption, self.max_words) + '.')
            txt_id += 1

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):
        image_path = self.ann[index]['image']
        image = Image.open(image_path).convert('RGB')
        output = self.transforms(image)  # jinyu

        caption = self.ann[index]['caption'].replace('.', '. ')
        caption = pre_caption(caption, self.max_words) + '.'

        return output, caption, self.img_ids[self.ann[index]['image_id']], self.ann[index]['image']

