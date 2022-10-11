import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image

from dataset.caption_dataset import re_train_dataset, re_eval_dataset, pretrain_dataset
from dataset.nlvr_dataset import nlvr_dataset
from dataset.ve_dataset import ve_dataset
from dataset.vqa_dataset import vqa_dataset
from dataset.grounding_dataset import grounding_dataset

from dataset.randaugment import RandomAugment
from dataset.utils import GaussianBlur

from ibot_utils import DataAugmentationiBOT, NoAugmentationiBOT, FineAugmentationiBOT
from dataset.ibot_dataset import ImageFolderMask, Re_eval_ImageFolder, Re_train_ImageFolder, Gen_eval_ImageFolder, Gen_train_ImageFolder, Vqa_dataset, Gen_Vqa_dataset
from dataset.ibot_dataset import Cls_COVID
from dataset.ibot_dataset import Det_train_ImageFolder, Det_eval_ImageFolder


def create_dataset(dataset, config):
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(config['image_res'], scale=(0.85, 1.0), interpolation=Image.BICUBIC),
        transforms.RandomApply([GaussianBlur([.1, 1.])], p=0.5),
        transforms.RandomHorizontalFlip(),
        RandomAugment(1, 1, isPIL=True, augs=['Identity', 'AutoContrast', 'Equalize', 'Brightness', 'Sharpness',
                                              'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),
        transforms.ToTensor(),
    ])

    ibot_transform = DataAugmentationiBOT(
        config['global_crops_scale'],
        config['local_crops_scale'],
        config['global_crops_number'],
        config['local_crops_number'],
    )
    finetune_transform = FineAugmentationiBOT()
    test_transform = NoAugmentationiBOT(
    )

    if dataset == 'pretrain':
        patch_size = config['patch_size']
        dataset= ImageFolderMask(
            config['task'],
            config['train_file'],
            patch_size=patch_size,
            pred_ratio=config['pred_ratio'],
            pred_ratio_var=config['pred_ratio_var'],
            pred_aspect_ratio=(0.3, 1 / 0.3),
            pred_shape=config['pred_shape'],
            pred_start_epoch=config['pred_start_epoch'],
            transforms=ibot_transform
            )
        return dataset

    elif dataset == 'gen':
        train_dataset = Gen_Vqa_dataset(config['train_file'], finetune_transform, split='train')
        vqa_test_dataset = Gen_Vqa_dataset(config['test_file'], test_transform, split='test')
        return train_dataset, vqa_test_dataset

    elif dataset == 're':
        patch_size = config['patch_size']
        train_dataset = Re_train_ImageFolder(config['train_file'], patch_size, transforms=finetune_transform)
        val_dataset = Re_eval_ImageFolder(config['val_file'], patch_size, transforms=test_transform)
        test_dataset = Re_eval_ImageFolder(config['test_file'], patch_size, transforms=test_transform)
        return train_dataset, val_dataset, test_dataset

    elif dataset == 'det':
        patch_size = config['patch_size']
        train_dataset = Det_train_ImageFolder(config['train_file'], patch_size, transforms=finetune_transform)
        val_dataset = Det_eval_ImageFolder(config['mode'], config['val_file'], patch_size, transforms=test_transform)
        test_dataset = Det_eval_ImageFolder(config['mode'], config['test_file'], patch_size, transforms=test_transform)
        return train_dataset, val_dataset, test_dataset

    elif dataset == 'vqa':
        train_dataset = Vqa_dataset(config['train_file'], finetune_transform, config['vqa_root'], config['vg_root'],
                                    split='train', chest_only=config['chest_only'])
        vqa_test_dataset = Vqa_dataset(config['test_file'], test_transform, config['vqa_root'], config['vg_root'],
                                       split='test', answer_list=config['answer_list'], chest_only=config['chest_only'])
        return train_dataset, vqa_test_dataset

    elif dataset == 'cls':
        patch_size = config['patch_size']
        train_dataset = Cls_COVID(config['covid_train'], config['partial'], patch_size, transforms=finetune_transform)
        val_dataset = Cls_COVID(config['covid_test'], False, patch_size, transforms=test_transform)
        test_dataset = Cls_COVID(config['covid_test'], False, patch_size, transforms=test_transform)
        return train_dataset, val_dataset, test_dataset


def gen_collate_fn(batch):
    image_list, answer_list, weight_list, n = [], [], [], []
    for image, answer, weights in batch:
        image_list.append(image)
        weight_list += weights
        answer_list += answer
        n.append(len(answer))
    return torch.stack(image_list, dim=0), answer_list, torch.Tensor(weight_list), n


def vqa_collate_fn(batch):
    image_list, question_list, answer_list, weight_list, n = [], [], [], [], []
    for image, question, answer, weights in batch:
        image_list.append(image)
        question_list.append(question)
        weight_list += weights
        answer_list += answer
        n.append(len(answer))
    return torch.stack(image_list, dim=0), question_list, answer_list, torch.Tensor(weight_list), n


def create_sampler(datasets, shuffles, num_tasks, global_rank):
    samplers = []
    for dataset, shuffle in zip(datasets, shuffles):
        sampler = torch.utils.data.DistributedSampler(dataset, num_replicas=num_tasks, rank=global_rank,
                                                      shuffle=shuffle)
        samplers.append(sampler)
    return samplers


def create_loader(datasets, samplers, batch_size, num_workers, is_trains, collate_fns):
    loaders = []
    for dataset, sampler, bs, n_worker, is_train, collate_fn in zip(datasets, samplers, batch_size, num_workers,
                                                                    is_trains, collate_fns):
        if is_train:
            shuffle = (sampler is None)
            drop_last = True
        else:
            sampler = None
            shuffle = True
            drop_last = True
        loader = DataLoader(
            dataset,
            batch_size=bs,
            num_workers=n_worker,
            pin_memory=True,
            sampler=sampler,
            shuffle=shuffle,
            collate_fn=collate_fn,
            drop_last=drop_last,
        )
        loaders.append(loader)
    return loaders


def create_VQA_loader(datasets, samplers, batch_size, num_workers, is_trains, collate_fns):
    loaders = []
    for dataset, sampler, bs, n_worker, is_train, collate_fn in zip(datasets, samplers, batch_size, num_workers,
                                                                    is_trains, collate_fns):
        if is_train:
            shuffle = (sampler is None)
            drop_last = True
        else:
            shuffle = False
            drop_last = False
        loader = DataLoader(
            dataset,
            batch_size=bs,
            num_workers=n_worker,
            pin_memory=True,
            sampler=sampler,
            shuffle=shuffle,
            collate_fn=collate_fn,
            drop_last=drop_last,
        )
        loaders.append(loader)
    return loaders