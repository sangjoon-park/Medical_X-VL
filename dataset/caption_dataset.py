import json
import os
import random

from torch.utils.data import Dataset

from PIL import Image
from PIL import ImageFile
import pandas as pd

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

from dataset.utils import pre_caption


class re_train_dataset(Dataset):
    def __init__(self, ann_file, transform, image_root, max_words=60):
        train_data = [json.loads(l) for l in open(ann_file[0])]

        # Train data
        train_list = []
        for data in train_data:
            _, _, label, txt, img = data.keys()
            d_txt = data[txt]
            d_img = data[img].replace('/home/mimic-cxr/dataset/image_preprocessing/',
                                      '/4TB_hdd/vision_language/mimic_dset/')
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
                if '___' in caption:  # Remove if raise an error.
                    continue
                total_captions.append(caption)
            len_sentence = len(total_captions)
            len_choice = min(len_sentence, 3)   # 3개만 고르기 (혹은 더 적으면 그 숫자가 최대)
            selected_captions = random.sample(total_captions, len_choice)
            selected_captions = '.'.join(selected_captions)

            if len(selected_captions) == 0:
                continue
            self.ann.append({'image': subject['image'], 'caption': selected_captions, 'image_id': sub_idx, 'label': label})

        self.transform = transform
        self.max_words = max_words
        self.img_ids = {}

        n = 0
        for ann in self.ann:
            img_id = ann['image_id']
            if img_id not in self.img_ids.keys():
                self.img_ids[img_id] = n
                n += 1

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):

        ann = self.ann[index]

        image_path = self.ann[index]['image']
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)  # jinyu

        caption = ann['caption'].replace('.', ' [SEP] ')
        caption = pre_caption(caption, self.max_words)

        return image, caption, self.img_ids[ann['image_id']], ann['image']


class re_eval_dataset(Dataset):
    def __init__(self, ann_file, transform, image_root, max_words=60):
        train_data = [json.loads(l) for l in open(ann_file)]

        # Train data
        train_list = []
        for data in train_data:
            _, _, label, txt, img = data.keys()
            d_txt = data[txt]
            d_img = data[img].replace('/home/mimic-cxr/dataset/image_preprocessing/',
                                      '/4TB_hdd/vision_language/mimic_dset/')
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
                if '___' in caption:  # Remove if raise an error.
                    continue
                total_captions.append(caption)
            len_sentence = len(total_captions)
            len_choice = min(len_sentence, 3)   # 3개만 고르기 (혹은 더 적으면 그 숫자가 최대)
            selected_captions = random.sample(total_captions, len_choice)
            selected_captions = '.'.join(selected_captions)

            if len(selected_captions) == 0:
                continue
            self.ann.append({'image': subject['image'], 'caption': selected_captions, 'label': label})

        self.transform = transform
        self.max_words = max_words

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
            caption = caption.replace('.', ' [SEP] ')
            self.text.append(pre_caption(caption, self.max_words))
            self.img2txt[img_id].append(txt_id)
            self.txt2img[txt_id] = img_id
            txt_id += 1

    def __len__(self):
        return len(self.image)

    def __getitem__(self, index):

        image_path = self.ann[index]['image']
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        return image, index, self.ann[index]['image']


class pretrain_dataset(Dataset):
    def __init__(self, ann_file, transform, max_words=60):
        train_data = [json.loads(l) for l in open(ann_file[0])]

        # Train data
        train_list = []
        for data in train_data:
            _, _, label, txt, img = data.keys()
            d_txt = data[txt]
            d_img = data[img].replace('/home/mimic-cxr/dataset/image_preprocessing/',
                                      '/4TB_hdd/vision_language/mimic_dset/')
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
                if '___' in caption:  # Remove if raise an error.
                    continue
                total_captions.append(caption)
            len_sentence = len(total_captions)
            len_choice = min(len_sentence, 3)   # 3개만 고르기 (혹은 더 적으면 그 숫자가 최대)
            selected_captions = random.sample(total_captions, len_choice)
            selected_captions = '.'.join(selected_captions)

            if len(selected_captions) == 0:
                continue
            self.ann.append({'image': subject['image'], 'caption': selected_captions})

        self.transform = transform
        self.max_words = max_words

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):

        ann = self.ann[index]
        caption = ann['caption'].replace('.', ' [SEP] ')

        if type(ann['caption']) == list:
            caption = pre_caption(random.choice(caption), self.max_words)
        else:
            caption = pre_caption(caption, self.max_words)

        image = Image.open(ann['image']).convert('RGB')
        images = self.transform(image)

        return images, caption, ann['image']