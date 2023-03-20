import argparse
import os

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

from models.model_vqa import XVLModel
from models.vit import interpolate_pos_embed
import pandas as pd

import funcs
from dataset.utils import save_result
from dataset import create_dataset, create_sampler, create_error_loader
from dataset.utils import pre_question

from scheduler import create_scheduler
from optim import create_optimizer
from transformers import AutoTokenizer
from funcs import post_process


def train(model, data_loader, optimizer, epoch, warmup_steps, device, scheduler, config):
    # train
    model.train()

    metric_logger = funcs.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', funcs.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('loss', funcs.SmoothedValue(window_size=1, fmt='{value:.4f}'))

    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 50
    step_size = 100
    warmup_iterations = warmup_steps * step_size

    for i, (image, error_text, answer, label) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        image = image.to(device, non_blocking=True)

        if epoch > 0 or not config['warm_up']:
            alpha = config['alpha']
        else:
            alpha = config['alpha'] * min(1, i / len(data_loader))

        loss = model(image, error_text, answer, train=True)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        if epoch == 0 and i % step_size == 0 and i <= warmup_iterations:
            scheduler.step(i // step_size)

            # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())
    return {k: "{:.3f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluation(model, data_loader, device, config):
    # test
    model.eval()

    metric_logger = funcs.MetricLogger(delimiter="  ")
    header = 'Error correction test result:'
    print_freq = 50

    result = []
    bos_token = config['bos_token']
    for n, (image, error_text, answer, label) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        image = image.to(device, non_blocking=True)
        stt_text_input = model.module.tokenizer.batch_encode_plus(batch_text_or_text_pairs=error_text,
                                                    add_special_tokens=True,
                                                    padding='longest',
                                                    return_tensors='pt').to(device)
        candidates = model(image, stt_text_input, bos_token, train=False)
        corrected_text = model.module.tokenizer.decode(candidates[0])
        result.append({"error": error_text, "corrected": corrected_text, "label": answer})
    return result


def main(args, config):
    funcs.init_distributed_mode(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + funcs.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    start_epoch = 0
    max_epoch = config['schedular']['epochs']
    warmup_steps = config['schedular']['warmup_epochs']

    #### Dataset ####
    print("Creating error correction datasets")
    datasets = create_dataset('error_correction', config)

    if args.distributed:
        num_tasks = funcs.get_world_size()
        global_rank = funcs.get_rank()
        samplers = create_sampler(datasets, [True, False, False], num_tasks, global_rank)
    else:
        samplers = [None, None, None]

    train_loader, test_loader = create_error_loader(datasets, samplers,
                                              batch_size=[config['batch_size_train'], config['batch_size_test']],
                                              num_workers=[4, 4], is_trains=[True, False],
                                              collate_fns=[None, None])

    #### Model ####
    print("Creating model")
    model = XVLModel(config=config)
    model = model.to(device)

    arg_opt = funcs.AttrDict(config['optimizer'])
    optimizer = create_optimizer(arg_opt, model)
    arg_sche = funcs.AttrDict(config['schedular'])
    lr_scheduler, _ = create_scheduler(arg_sche, optimizer)

    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        state_dict = checkpoint['model']

        if not args.resume:
            if not args.evaluate:
                # reshape positional embedding to accomodate for image resolution change
                pos_embed_reshaped  = interpolate_pos_embed(state_dict['visual_encoder.pos_embed'],model.visual_encoder)
                state_dict['visual_encoder.pos_embed'] = pos_embed_reshaped

                if config['distill']:
                    m_pos_embed_reshaped = interpolate_pos_embed(state_dict['visual_encoder_m.pos_embed'],model.visual_encoder_m)
                    state_dict['visual_encoder_m.pos_embed'] = m_pos_embed_reshaped
                # state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}

                # if config['distill']:
                #     m_pos_embed_reshaped = interpolate_pos_embed(state_dict['visual_encoder_m.pos_embed'],model.visual_encoder_m)
                #     state_dict['visual_encoder_m.pos_embed'] = m_pos_embed_reshaped

                # for key in list(state_dict.keys()):
                #     encoder_key = key.replace('.bert', '')
                #     state_dict[encoder_key] = state_dict[key]
                    # if 'text_encoder.bert' in key:
                    #     encoder_key = key.replace('.bert', '')
                    #     state_dict[encoder_key] = state_dict[key]
                    #     del state_dict[key]
                    # if 'text_encoder_m.bert' in key:
                    #     encoder_key = key.replace('text_encoder_m.bert', 'text_encoder_m')
                    #     state_dict[encoder_key] = state_dict[key]
                    #     del state_dict[key]
                    #
                    # if 'fusion_encoder.bert' in key:
                    #     encoder_key = key.replace('fusion_encoder.bert', 'fusion_encoder')
                    #     state_dict[encoder_key] = state_dict[key]
                    #     del state_dict[key]
                    # if 'fusion_encoder_m.bert' in key:
                    #     encoder_key = key.replace('fusion_encoder_m.bert', 'fusion_encoder_m')
                    #     state_dict[encoder_key] = state_dict[key]
                    #     del state_dict[key]

                for key in list(state_dict.keys()):
                    if 'bert' in key:
                        encoder_key = key.replace('bert.', '')
                        state_dict[encoder_key] = state_dict[key]
                        # intialize text decoder as multimodal encoder (last 6 layers of model.text_encoder)
                    if 'fusion_encoder' in key:
                        if 'layer' in key:
                            encoder_keys = key.split('.')
                            layer_num = int(encoder_keys[4])
                            if layer_num < 8:
                                del state_dict[key]
                                continue
                            else:
                                decoder_layer_num = (layer_num - 8)
                                encoder_keys[4] = str(decoder_layer_num)
                                encoder_key = '.'.join(encoder_keys)
                        else:
                            encoder_key = key
                        decoder_key = encoder_key.replace('fusion_encoder', 'fusion_decoder')
                        state_dict[decoder_key] = state_dict[key]

                        del state_dict[key]

        msg = model.load_state_dict(state_dict, strict=False)
        print('load checkpoint from %s' % args.checkpoint)
        print(msg)

    if config['distill']:
        model.copy_params()
        print('model parameters are copied.')

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module


    print("Start training")
    start_time = time.time()

    with torch.no_grad():
        corrected_results = evaluation(model, test_loader, device, config)
        result_file = save_result(corrected_results, args.result_dir, 'Test_corrected_result')

    # if args.evaluate:
    #     with torch.no_grad():
    #         corrected_results   = evaluation(model, test_loader, tokenizer, device, config)
    #         result_file         = save_result(corrected_results, args.result_dir, 'Test_corrected_result')
    # else:
    for epoch in range(start_epoch, max_epoch):
        if epoch > 0:
            lr_scheduler.step(epoch + warmup_steps)

        # if not args.evaluate:
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)
        train_stats = train(model, train_loader, optimizer, epoch, warmup_steps, device, lr_scheduler,config)
        # else:
        #     break

        if funcs.is_main_process():
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},'epoch': epoch}
            with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
                f.write(json.dumps(log_stats) + "\n")

            save_obj = {
                            'model': model_without_ddp.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'lr_scheduler': lr_scheduler.state_dict(),
                            'config': config,
                            'epoch': epoch,
                        }
            torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint.pth'))

        corrected_results = evaluation(model, test_loader, device, config)
        result_file = save_result(corrected_results, args.result_dir, 'Valid_corrected_result_epoch%d' % epoch)

        dist.barrier()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/correction.yaml')
    parser.add_argument('--checkpoint', default='assets_withvision/correction/checkpoint.pth')
    parser.add_argument('--output_dir', default='assets_withvision/correction')
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--text_encoder', default='bert-base-uncased')
    parser.add_argument('--text_decoder', default='bert-base-uncased')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=True, type=bool)
    parser.add_argument('--resume', default=False, type=bool)
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    args.result_dir = os.path.join(args.output_dir, 'result')

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.result_dir).mkdir(parents=True, exist_ok=True)

    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))
    main(args, config)