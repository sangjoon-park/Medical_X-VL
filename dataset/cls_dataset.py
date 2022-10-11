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


class cls_train_dataset(Dataset):
    def __init__(self, ann_file, transform, image_root, max_words=30):
        train_data = [json.loads(l) for l in open(ann_file[0])]

        # Train data
        train_list = []
        for data in train_data:
            _, _, label, txt, img = data.keys()
            d_txt = data[txt]
            d_img = data[img].replace('/home/mimic-cxr/dataset/image_preprocessing/',
                                      '/4TB_hdd/vision_language/mimic_dset/')
            train_list.append([d_img, d_txt])
        train_df = pd.DataFrame(train_list, columns=['image', 'caption'])

        # Make into dictionary
        self.ann = []
        for sub_idx in range(len(train_df)):
            subject = train_df.iloc[sub_idx]
            captions = subject['caption'].split('.')
            # captions = [subject['caption']]
            for caption in captions:
                if len(caption) < 4:
                    continue
                if '___' in caption:  # Remove if raise an error.
                    continue
                self.ann.append({'image': subject['image'], 'caption': caption, 'image_id': sub_idx})

        self.transform = transform
        self.image_root = image_root
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
        image1 = self.transform(image)  # jinyu
        image2 = self.transform(image)  # jinyu

        caption = pre_caption(ann['caption'], self.max_words)

        return image1, image2, caption, self.img_ids[ann['image_id']]


class cls_eval_dataset(Dataset):
    def __init__(self, ann_file, transform, image_root, max_words=30):
        train_data = [json.loads(l) for l in open(ann_file)]

        # Train data
        train_list = []
        for data in train_data:
            _, _, label, txt, img = data.keys()
            d_txt = data[txt]
            d_img = data[img].replace('/home/mimic-cxr/dataset/image_preprocessing/',
                                      '/4TB_hdd/vision_language/mimic_dset/')
            train_list.append([d_img, d_txt])
        train_df = pd.DataFrame(train_list, columns=['image', 'caption'])

        # Make into dictionary
        self.ann = []
        for sub_idx in range(len(train_df)):
            subject = train_df.iloc[sub_idx]
            # captions = [subject['caption']]
            captions = subject['caption'].split('.')
            selected_captions = []
            for caption in captions:
                if len(caption) < 4:
                    continue
                if '___' in caption:  # Remove if raise an error.
                    continue
                selected_captions.append(caption)
            self.ann.append({'image': subject['image'], 'caption': selected_captions})

        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words

        self.text = []
        self.image = []
        self.txt2img = {}
        self.img2txt = {}

        txt_id = 0
        for img_id, ann in enumerate(self.ann):
            self.image.append(ann['image'])
            self.img2txt[img_id] = []
            for i, caption in enumerate(ann['caption']):
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

        return image, index