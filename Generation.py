'''
 * Copyright (c) 2021, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see LICENSE.txt file in the repo root or https://opensource.org/licenses/BSD-3-Clause
'''

import argparse
import os

os.environ["TOKENIZERS_PARALLELISM"] = "true"
import ruamel.yaml as yaml
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from models.model_generation import XVLModel, XVLModel_infer
from models.vit import interpolate_pos_embed
from models.tokenization_bert import BertTokenizer

import utils
from dataset import create_dataset, create_sampler, create_loader, gen_collate_fn
from scheduler import create_scheduler
from optim import create_optimizer
from ibot_utils import iBOTLoss
from transformers import AutoTokenizer
import ibot_utils


def train(model, data_loader, optimizer, tokenizer, epoch, warmup_steps, device, scheduler, config, fp16_scaler):
    # train
    model.train()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
    metric_logger.add_meter('loss_mlm', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_ita', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_itm', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_ibot', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))

    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 50
    step_size = 100
    warmup_iterations = warmup_steps * step_size

    # define iBOT loss
    same_dim = config['shared_head'] or config['shared_head_teacher']
    ibot_loss = iBOTLoss(
        config['out_dim'],
        config['out_dim'] if same_dim else config['patch_out_dim'],
        config['global_crops_number'],
        config['local_crops_number'],
        config['warmup_teacher_temp'],
        config['teacher_temp'],
        config['warmup_teacher_patch_temp'],
        config['teacher_patch_temp'],
        config['warmup_teacher_temp_epochs'],
        config['schedular']['epochs'],
        lambda1=config['lambda1'],
        lambda2=config['lambda2'],
        mim_start_epoch=config['pred_start_epoch'],
    ).cuda()

    if args.distributed:
        data_loader.sampler.set_epoch(epoch)

    for i, (images, text, masks, label) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        optimizer.zero_grad()

        images = [im.cuda(non_blocking=True) for im in images]
        masks = [msk.cuda(non_blocking=True) for msk in masks]

        text_input = tokenizer(text, padding='longest', truncation=True, max_length=120, return_tensors="pt").to(device)

        if epoch > 0:
            alpha = config['alpha']
        else:
            alpha = config['alpha'] * min(1, i / len(data_loader))

            # calculate iteration
        it = len(data_loader) * epoch + i

        loss_mlm, loss_ita, loss_itm, loss_ibot = model(it, images, text_input, masks, label, ibot_loss, epoch,
                                                        fp16_scaler, alpha=alpha)

        loss = loss_mlm + loss_ita + loss_itm + loss_ibot

        param_norms = None
        if fp16_scaler is None:
            loss.backward()
            if config['clip_grad']:
                param_norms = ibot_utils.clip_gradients(model.module.visual_encoder, config['clip_grad'])
            ibot_utils.cancel_gradients_last_layer(epoch, model.module.visual_encoder,
                                                   config['freeze_last_layer'])
            optimizer.step()
        else:
            fp16_scaler.scale(loss).backward()
            if args.clip_grad:
                fp16_scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                param_norms = ibot_utils.clip_gradients(model.module.visual_encoder, config['clip_grad'])
            ibot_utils.cancel_gradients_last_layer(epoch, model.module.visual_encoder,
                                                   config['freeze_last_layer'])
            fp16_scaler.step(optimizer)
            fp16_scaler.update()

        metric_logger.update(loss_mlm=loss_mlm.item())
        metric_logger.update(loss_ita=loss_ita.item())
        metric_logger.update(loss_itm=loss_itm.item())
        metric_logger.update(loss_ibot=loss_ibot.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        if epoch == 0 and i % step_size == 0 and i <= warmup_iterations:
            scheduler.step(i // step_size)

            # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())
    return {k: "{:.3f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}

def evaluation(model, data_loader, tokenizer, device, config):
    # test
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Generate Generation test result:'
    print_freq = 50

    results = {}

    bos_token = config['bos_token']

    for n, (image, caption, image_path) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        image = image.to(device, non_blocking=True)
        candidates = model(device, image, bos_token, train=False, k=config['k_test'])

        if config['search'] == 'greedy':
            sentences = tokenizer.decode(candidates[0])
            results[image_path[0]] = {'caption': caption, 'predicted': sentences}

        elif config['search'] == 'beam':
            sentences = []
            for candidate in candidates[0]:
                candidate = tokenizer.decode(candidate[0])
                sentences.append(candidate)

            results[image_path[0]] = {'caption': caption, 'predicted': sentences[-1]}

    return results

def main(args, config):
    utils.init_distributed_mode(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    start_epoch = 0
    max_epoch = config['schedular']['epochs']
    warmup_steps = config['schedular']['warmup_epochs']

    fp16_scaler = None
    if config['use_fp16']:
        fp16_scaler = torch.cuda.amp.GradScaler()

    #### Dataset ####
    print("Creating Generation dataset")
    datasets = [create_dataset('pretrain', config)]

    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        samplers = create_sampler(datasets, [True], num_tasks, global_rank)
    else:
        samplers = [None]

    data_loader = \
    create_loader(datasets, samplers, batch_size=[config['batch_size_train']], num_workers=[4], is_trains=[True],
                  collate_fns=[None])[0]

    datasets = create_dataset('gen', config)

    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        samplers = create_sampler(datasets, [True, False], num_tasks, global_rank)
    else:
        samplers = [None, None]

    _, test_loader = create_loader(datasets, samplers,
                                              batch_size=[config['batch_size_train'], config['batch_size_test']],
                                              num_workers=[4, 4], is_trains=[True, False],
                                              collate_fns=[gen_collate_fn, None])

    # tokenizer = BertTokenizer.from_pretrained(args.text_encoder)
    tokenizer = AutoTokenizer.from_pretrained("./my_tokenizer/")

    #### Model ####
    print("Creating model")
    if not args.evaluate:
        model = XVLModel(data_loader, config=config, text_encoder=args.text_encoder, tokenizer=tokenizer, init_deit=True)
    else:
        model = XVLModel_infer(config=config, text_encoder=args.text_encoder, tokenizer=tokenizer)

    model = model.to(device)

    arg_opt = utils.AttrDict(config['optimizer'])
    optimizer = create_optimizer(arg_opt, model)
    arg_sche = utils.AttrDict(config['schedular'])
    lr_scheduler, _ = create_scheduler(arg_sche, optimizer)

    if args.checkpoint:
        if not args.evaluate:
            checkpoint = torch.load(args.checkpoint, map_location='cpu')
            state_dict = checkpoint['model']
            model.load_state_dict(state_dict)
            print('load checkpoint from %s' % args.checkpoint)
        else:
            checkpoint = torch.load(args.checkpoint, map_location='cpu')
            state_dict = checkpoint['model']
            pos_embed_reshaped = interpolate_pos_embed(state_dict['visual_encoder.backbone.pos_embed'],
                                                       model.visual_encoder)
            state_dict['visual_encoder.backbone.pos_embed'] = pos_embed_reshaped
            state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}

            if config['distill']:
                m_pos_embed_reshaped = interpolate_pos_embed(state_dict['visual_encoder_m.pos_embed'],
                                                             model.visual_encoder_m)
                state_dict['visual_encoder_m.pos_embed'] = m_pos_embed_reshaped

            msg = model.load_state_dict(state_dict, strict=False)
            print('load checkpoint from %s' % args.checkpoint)
            print(msg)

    if config['distill']:
        model.copy_params()
        print('model parameters are copied.')

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    print("Start training")
    start_time = time.time()

    for epoch in range(start_epoch, max_epoch):
        if args.evaluate:
            break

        if epoch > 0:
            lr_scheduler.step(epoch + warmup_steps)

        train_stats = train(model, data_loader, optimizer, tokenizer, epoch, warmup_steps, device, lr_scheduler, config,
                            fp16_scaler)
        if utils.is_main_process():
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         'epoch': epoch,
                         }
            save_obj = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'config': config,
                'epoch': epoch,
            }
            torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint_%02d.pth' % epoch))

            with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
                f.write(json.dumps(log_stats) + "\n")

        dist.barrier()

    generation_results = evaluation(model, test_loader, tokenizer, device, config)
    with open('./gen_{}.json'.format(args.output_dir.split('/')[-3], config['dataset']), 'w') as f:
        json.dump(generation_results, f)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/Pretrain.yaml')
    parser.add_argument('--checkpoint', default='')
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--resume', default=False, type=bool)
    parser.add_argument('--output_dir', default='Pretrain/')
    parser.add_argument('--text_encoder', default='bert-base-uncased')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=True, type=bool)

    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))

    main(args, config)