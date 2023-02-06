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

import language_evaluation
from dataset.utils import save_result

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from models.model_generation import XVLModel
from models.vit import interpolate_pos_embed
from models.tokenization_bert import BertTokenizer
from dataset.utils import pre_caption, pre_question

import utils
from dataset import create_dataset, create_sampler, create_loader, gen_collate_fn
from scheduler import create_scheduler
from optim import create_optimizer
from ibot_utils import iBOTLoss
from transformers import AutoTokenizer
import ibot_utils
from eval_function import COCOEvalCapDirect


def train(model, data_loader, optimizer, tokenizer, epoch, warmup_steps, device, scheduler, config, fp16_scaler):
    # train
    model.train()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
    metric_logger.add_meter('loss_mlm', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))

    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 50
    step_size = 100
    warmup_iterations = warmup_steps * step_size

    if args.distributed:
        data_loader.sampler.set_epoch(epoch)

    for i, (image, findings, image_id) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        optimizer.zero_grad()

        image = image.to(device)
        text = findings

        text_input = tokenizer(text, padding='longest', truncation=True, max_length=180, return_tensors="pt").to(device)

        if epoch > 0:
            alpha = config['alpha']
        else:
            alpha = config['alpha'] * min(1, i / len(data_loader))

        loss_mlm = model(image, text_input, train=True, alpha=alpha)

        loss = loss_mlm

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
    header = 'Generate test set result:'
    print_freq = 10

    result = {}

    answer_input = None
    for n, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        image = batch[0].to(device)
        report = batch[1]
        ID = batch[2][0]
        text = report

        # load caption
        caption = []
        for txt in text:
            caption.append(pre_caption(txt, max_words=120))

        object_labels = ["" for i in range(len(image))]
        gold_caption = caption

        topk_ids, topk_probs = model(image, train=False)

        for topk_id, topk_prob, gold_caption_list in zip(topk_ids, topk_probs, gold_caption):
            ans = tokenizer.decode(topk_id[0]).replace("[SEP]", "").replace("[CLS]", "").replace("[PAD]", "").strip()
            result[ID] = {"predicted": ans, "caption": gold_caption_list}

            if n % 20 == 0:
                print("pred_caption : {} / gold_caption: {}".format(ans, gold_caption_list))
    return result

def cal_metric(result_file):
    result_list = json.load(open(result_file, "r"))
    predicts = []
    answers = []
    for each in result_list:
        predicts.append(each["pred_caption"])
        answers.append(each["gold_caption"])
    evaluator = language_evaluation.CocoEvaluator(verbose=False)
    results = evaluator.run_evaluation(predicts, answers)
    print (len(result_list), results)
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
    datasets = create_dataset('generation', config)

    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        samplers = create_sampler(datasets, [True, False], num_tasks, global_rank)
    else:
        samplers = [None, None]

    data_loader, test_loader = create_loader(datasets, samplers,
                                              batch_size=[config['batch_size_train'], config['batch_size_test']],
                                              num_workers=[4, 4], is_trains=[True, False],
                                              collate_fns=[None, None])

    url = "microsoft/BiomedVLP-CXR-BERT-specialized"
    tokenizer = AutoTokenizer.from_pretrained(url, trust_remote_code=True)

    #### Model ####
    print("Creating model")
    model = XVLModel(config=config, tokenizer=tokenizer)

    model = model.to(device)

    arg_opt = utils.AttrDict(config['optimizer'])
    optimizer = create_optimizer(arg_opt, model)
    arg_sche = utils.AttrDict(config['schedular'])
    lr_scheduler, _ = create_scheduler(arg_sche, optimizer)

    if args.checkpoint:
        if not args.evaluate:
            checkpoint = torch.load(args.checkpoint, map_location='cpu')
            state_dict = checkpoint['model']
            # model.load_state_dict(state_dict)
            # print('load checkpoint from %s' % args.checkpoint)

            pos_embed_reshaped = interpolate_pos_embed(state_dict['visual_encoder.backbone.pos_embed'],
                                                       model.visual_encoder)
            state_dict['visual_encoder.backbone.pos_embed'] = pos_embed_reshaped
            if config['distill']:
                m_pos_embed_reshaped = interpolate_pos_embed(state_dict['visual_encoder_m.backbone.pos_embed'],
                                                             model.visual_encoder_m)
                state_dict['visual_encoder_m.backbone.pos_embed'] = m_pos_embed_reshaped
            state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}

        else:
            checkpoint = torch.load(args.checkpoint, map_location='cpu')
            state_dict = checkpoint['model']
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
    # vqa_result = evaluation(model, test_loader, tokenizer, device, config)
    # result_file = save_result(vqa_result, args.result_dir, 'vqa_result_epoch10')
    #
    # # Eval metrics
    # cocoEval = COCOEvalCapDirect(vqa_result)
    # cocoEval.evaluate()
    #
    # for metric, score in cocoEval.eval.items():
    #     print('%s: %.3f' % (metric, score))

    for epoch in range(start_epoch, max_epoch):
        if epoch > 0:
            lr_scheduler.step(epoch + warmup_steps)

        if not args.evaluate:
            if args.distributed:
                data_loader.sampler.set_epoch(epoch)

            train_stats = train(model, data_loader, optimizer, tokenizer, epoch, warmup_steps, device, lr_scheduler, config,
                                fp16_scaler)

        if args.evaluate:
            break

        # ########## Val results ##########
        # print('>>>>> Validation results')
        # vqa_result = evaluation(model, test_loader, tokenizer, device, config)
        # result_file = save_result(vqa_result, args.result_dir, 'vqa_result_epoch%d' % epoch)
        #
        # # Eval metrics
        # cocoEval = COCOEvalCapDirect(vqa_result)
        # cocoEval.evaluate()
        #
        # for metric, score in cocoEval.eval.items():
        #     print('%s: %.3f' % (metric, score))

        ########## Test results ##########
        print('>>>>> Test results')
        vqa_result = evaluation(model, test_loader, tokenizer, device, config)
        result_file = save_result(vqa_result, args.result_dir, 'vqa_result_epoch%d' % epoch)

        # Eval metrics
        cocoEval = COCOEvalCapDirect(vqa_result)
        cocoEval.evaluate()

        for metric, score in cocoEval.eval.items():
            print('%s: %.3f' % (metric, score))

        if utils.is_main_process():
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         'epoch': epoch,
                         }
            with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
                f.write(json.dumps(log_stats) + "\n")

            torch.save({
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'config': config,
                'epoch': epoch,
            }, os.path.join(args.output_dir, 'checkpoint.pth'))

        dist.barrier()

    generation_results = evaluation(model, test_loader, tokenizer, device, config)
    with open('./gen_val_{}.json'.format(args.output_dir.split('/')[-3], config['dataset']), 'w') as f:
        json.dump(generation_results, f)

    generation_results = evaluation(model, test_loader, tokenizer, device, config)
    with open('./gen_test_{}.json'.format(args.output_dir.split('/')[-3], config['dataset']), 'w') as f:
        json.dump(generation_results, f)

    # Eval metrics
    cocoEval = COCOEvalCapDirect(generation_results)
    cocoEval.evaluate()

    for metric, score in cocoEval.eval.items():
        print('%s: %.3f' % (metric, score))


    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/Generation.yaml')
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

    args.result_dir = os.path.join(args.output_dir, 'result')
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.result_dir).mkdir(parents=True, exist_ok=True)

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))

    main(args, config)