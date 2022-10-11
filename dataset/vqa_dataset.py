import os
import json
import random
from PIL import Image
from torch.utils.data import Dataset
from dataset.utils import pre_question
import _pickle as cPickle


class vqa_dataset(Dataset):
    def __init__(self, ann_file, transform, vqa_root, vg_root, eos='[SEP]', split="train", max_ques_words=30,
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
            if data['image_organ'] == 'CHEST':
                self.selected_ann.append(data)
            else:
                pass
        self.ann = self.selected_ann

        if split == 'test':
            self.max_ques_words = 50  # do not limit question length during test
            self.answer_list = []
            for ans in self.ann:
                answer = pre_question(ans['answer'], self.max_ques_words)
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

                if not answer in self.answer_list:
                    self.answer_list.append(answer)

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):

        ann = self.ann[index]

        # if ann['dataset']=='vqa':
        #     image_path = os.path.join(self.vqa_root,ann['image'])
        # elif ann['dataset']=='vg':
        #     image_path = os.path.join(self.vg_root,ann['image'])

        image = Image.open(
            '/4TB_hdd/downloads/data_RAD/home/mimic-cxr/dataset/data_RAD/images/' + ann['image_name']).convert('RGB')
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

            # elif ann['dataset']=='vg':
            #     answers = [ann['answer']]
            #     weights = [0.5]

            answers = [str(answer) + self.eos for answer in answers]

            return image, question, answers, weights