import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image

from dataset.caption_dataset import re_train_dataset, re_eval_dataset, pretrain_dataset
from dataset.grounding_dataset import grounding_dataset

from dataset.randaugment import RandomAugment
from dataset.utils import GaussianBlur

from ibot_utils import DataAugmentationiBOT, NoAugmentationiBOT, FineAugmentationiBOT
from dataset.ibot_dataset import ImageFolderMask, Retrieval_dataset, Gen_dataset, Cls_dataset, Vqa_dataset, CXRTrainDataset, CXRTestDataset
from health_multimodal.image.data.transforms import create_chest_xray_transform_for_inference, create_chest_xray_transform_for_train


def create_dataset(dataset, config):
    # # ibot_transform = DataAugmentationiBOT(
    # #     config['global_crops_scale'],
    # #     config['local_crops_scale'],
    # #     config['global_crops_number'],
    # #     config['local_crops_number'],
    # # )
    # finetune_transform = FineAugmentationiBOT()
    # test_transform = NoAugmentationiBOT(
    # )


    # jinyu: add augmentation
    TRANSFORM_RESIZE = 240
    TRANSFORM_CENTER_CROP_SIZE = 224

    resize = TRANSFORM_RESIZE,
    center_crop_size = TRANSFORM_CENTER_CROP_SIZE,
    test_transform = create_chest_xray_transform_for_inference(resize, center_crop_size)
    train_transform = create_chest_xray_transform_for_train(center_crop_size, scale=(0.7, 1.0))

    if dataset == 'pretrain':
        patch_size = config['patch_size']
        dataset= ImageFolderMask(
            config['train_file'],
            patch_size=patch_size,
            pred_ratio=config['pred_ratio'],
            pred_ratio_var=config['pred_ratio_var'],
            pred_aspect_ratio=(0.3, 1 / 0.3),
            pred_shape=config['pred_shape'],
            pred_start_epoch=config['pred_start_epoch'],
            transforms_1=test_transform,
            transforms_2=train_transform
            )
        return dataset

    elif dataset == 'cls':
        train_dataset = Cls_dataset(config['train'], transforms=train_transform)
        # val_dataset = Cls_dataset(config['validation'], transforms=test_transform)
        test_dataset = Cls_dataset(config['test'], transforms=test_transform)
        return train_dataset, test_dataset

    elif dataset == 'retrieval':
        test_dataset = Retrieval_dataset(config['candidate_file'], config['query_file'], transforms=test_transform)
        return test_dataset

    elif dataset == 'generation':
        train_dataset = Gen_dataset(config['train_file'], train_transform, mode='train')
        # val_dataset = Gen_dataset(config['val_file'], test_transform, mode='test')
        test_dataset = Gen_dataset(config['train_file'], test_transform, mode='test')
        return train_dataset, test_dataset

    elif dataset == 'vqa':
        train_dataset = Vqa_dataset(config['train_file'], train_transform, config['vqa_root'], config['vg_root'],
                                    split='train', chest_only=config['chest_only'])
        vqa_test_dataset = Vqa_dataset(config['test_file'], test_transform, config['vqa_root'], config['vg_root'],
                                       split='test', answer_list=config['answer_list'], chest_only=config['chest_only'])
        return train_dataset, vqa_test_dataset

    elif dataset == 'error_correction':
        train_set = CXRTrainDataset(config['train_file'], transform=train_transform)
        test_set = CXRTestDataset(config['train_file'], transform=test_transform)
        return train_set, test_set


def gen_collate_fn(batch):
    image_list, answer_list, weight_list, n = [], [], [], []
    for image, answer, weights in batch:
        image_list.append(image)
        weight_list += weights
        answer_list += answer
        n.append(len(answer))
    return torch.stack(image_list, dim=0), answer_list, torch.Tensor(weight_list), n


def re_collate_fn(batch):
    image_list, index_list, id_list, caption_list, label_list = [], [], [], [], []
    for image, index, id, caption, label in batch:
        image_list.append(image)
        index_list.append(index)
        id_list.append(id)
        caption_list.append(caption)
        label_list.append(label)
    return torch.stack(image_list, dim=0), index_list, id_list, caption_list, label_list


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



def create_error_loader(datasets, samplers, batch_size, num_workers, is_trains, collate_fns):
    loaders = []
    for dataset, sampler, bs, n_worker, is_train, collate_fn in zip(datasets, samplers, batch_size, num_workers,
                                                                    is_trains, collate_fns):
        if is_train:
            shuffle = (sampler is None)
            drop_last = True
        else:
            sampler = None
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