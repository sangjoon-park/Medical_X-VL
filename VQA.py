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

from models.model_vqa import XVLModel
from models.vit import interpolate_pos_embed
from models.tokenization_bert import BertTokenizer
import pandas as pd

import utils
from dataset.utils import save_result
from dataset import create_dataset, create_sampler, create_VQA_loader, vqa_collate_fn
from dataset.utils import pre_question

from scheduler import create_scheduler
from optim import create_optimizer
from transformers import AutoTokenizer
from utils import post_process


def train(model, data_loader, optimizer, tokenizer, epoch, warmup_steps, device, scheduler, config):
    # train
    model.train()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))

    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 50
    step_size = 100
    warmup_iterations = warmup_steps * step_size

    for i, (image, question, answer, weights, n) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        image, weights = image.to(device, non_blocking=True), weights.to(device, non_blocking=True)
        question_input = tokenizer(question, padding='longest', truncation=True, max_length=120, return_tensors="pt").to(
            device)
        answer_input = tokenizer(answer, padding='longest', return_tensors="pt").to(device)

        if epoch > 0 or not config['warm_up']:
            alpha = config['alpha']
        else:
            alpha = config['alpha'] * min(1, i / len(data_loader))

        loss = model(image, question_input, answer_input, train=True, alpha=alpha, k=n, weights=weights)

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
def evaluation(model, data_loader, tokenizer, device, config):
    # test
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'VQA test result:'
    print_freq = 50

    result = []

    answer_list = [answer + config['eos'] for answer in data_loader.dataset.answer_list]
    answer_input = tokenizer(answer_list, padding='longest', return_tensors='pt').to(device)

    for n, (image, question, question_id) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        image = image.to(device, non_blocking=True)
        question_input = tokenizer(question, padding='longest', return_tensors="pt").to(device)

        topk_ids, topk_probs = model(image, question_input, answer_input, train=False, k=config['k_test'])

        for ques_id, topk_id, topk_prob in zip(question_id, topk_ids, topk_probs):
            ques_id = int(ques_id.item())
            _, pred = topk_prob.max(dim=0)
            result.append({"question_id": ques_id, "answer": data_loader.dataset.answer_list[topk_id[pred]]})

    return result


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

    #### Dataset ####
    print("Creating vqa datasets")
    datasets = create_dataset('vqa', config)

    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        samplers = create_sampler(datasets, [True, False], num_tasks, global_rank)
    else:
        samplers = [None, None]

    train_loader, test_loader = create_VQA_loader(datasets, samplers,
                                              batch_size=[config['batch_size_train'], config['batch_size_test']],
                                              num_workers=[4, 4], is_trains=[True, False],
                                              collate_fns=[vqa_collate_fn, None])

    tokenizer = AutoTokenizer.from_pretrained("./my_tokenizer/")

    #### Model ####
    print("Creating model")
    model = XVLModel(config=config, text_encoder=args.text_encoder, text_decoder=args.text_decoder, tokenizer=tokenizer)
    model = model.to(device)

    arg_opt = utils.AttrDict(config['optimizer'])
    optimizer = create_optimizer(arg_opt, model)
    arg_sche = utils.AttrDict(config['schedular'])
    lr_scheduler, _ = create_scheduler(arg_sche, optimizer)

    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        state_dict = checkpoint['model']

        if not args.resume:
            if not args.evaluate:
                # reshape positional embedding to accomodate for image resolution change
                pos_embed_reshaped = interpolate_pos_embed(state_dict['visual_encoder.backbone.pos_embed'],
                                                           model.visual_encoder)
                state_dict['visual_encoder.backbone.pos_embed'] = pos_embed_reshaped
                if config['distill']:
                    m_pos_embed_reshaped = interpolate_pos_embed(state_dict['visual_encoder_m.backbone.pos_embed'],
                                                                 model.visual_encoder_m)
                    state_dict['visual_encoder_m.backbone.pos_embed'] = m_pos_embed_reshaped
                state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}

                if not args.evaluate:
                    if config['distill']:
                        m_pos_embed_reshaped = interpolate_pos_embed(state_dict['visual_encoder_m.pos_embed'],
                                                                     model.visual_encoder_m)
                        state_dict['visual_encoder_m.pos_embed'] = m_pos_embed_reshaped

                    for key in list(state_dict.keys()):
                        if 'bert' in key:
                            encoder_key = key.replace('bert.', '')
                            state_dict[encoder_key] = state_dict[key]
                            # intialize text decoder as multimodal encoder (last 6 layers of model.text_encoder)
                        if 'text_encoder' in key:
                            if 'layer' in key:
                                encoder_keys = key.split('.')
                                layer_num = int(encoder_keys[4])
                                if layer_num < 6:
                                    del state_dict[key]
                                    continue
                                else:
                                    decoder_layer_num = (layer_num - 6)
                                    encoder_keys[4] = str(decoder_layer_num)
                                    encoder_key = '.'.join(encoder_keys)
                            else:
                                encoder_key = key
                            decoder_key = encoder_key.replace('text_encoder', 'text_decoder')
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

    for epoch in range(start_epoch, max_epoch):
        if epoch > 0:
            lr_scheduler.step(epoch + warmup_steps)

        if not args.evaluate:
            if args.distributed:
                train_loader.sampler.set_epoch(epoch)

            train_stats = train(model, train_loader, optimizer, tokenizer, epoch, warmup_steps, device, lr_scheduler,
                                config)

        if args.evaluate:
            break

        if utils.is_main_process():
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         'epoch': epoch,
                         }
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

        dist.barrier()

    vqa_result = evaluation(model, test_loader, tokenizer, device, config)
    result_file = save_result(vqa_result, args.result_dir, 'vqa_result_epoch%d' % epoch)

    # Load all answer list
    ann = json.load(open('/COVID_8TB/sangjoon/vision_language/data_RAD/home/mimic-cxr/dataset/data_RAD/testset.json', 'r'))
    test_list = []
    for data in ann:
        qid, img_name, _, answer, answer_type, _, question, _ = data.keys()
        d_qid = data[qid]
        d_ans = pre_question(data[answer], 50)
        d_type = data[answer_type]
        d_ans = post_process(d_ans)
        d_iid = data[img_name]
        d_que = data[question]

        test_list.append([d_qid, d_ans, d_type, d_iid, d_que])
    df = pd.DataFrame(test_list, columns=['qid', 'answer', 'type', 'iid', 'question'])

    o_acc = 0
    c_acc = 0
    t_acc = 0
    o_total = 0
    c_total = 0
    t_total = 0

    result_list = []

    for vqa in vqa_result:
        qid_r, ans_r = vqa.keys()
        pred_qid = vqa[qid_r]
        pred_ans = vqa[ans_r]
        label_ans = df.loc[df.qid == pred_qid].answer.iloc[0]
        type = df.loc[df.qid == pred_qid].type.iloc[0]
        iid = df.loc[df.qid == pred_qid].iid.iloc[0]
        que = df.loc[df.qid == pred_qid].question.iloc[0]

        if type == 'OPEN':
            if pred_ans == label_ans:
                o_acc += 1
            o_total += 1
        elif type == 'CLOSED':
            if pred_ans == label_ans:
                c_acc += 1
            c_total += 1
        else:
            raise AssertionError()
        if pred_ans == label_ans:
            t_acc += 1
        t_total += 1

        result_list.append([iid, que, type, label_ans, pred_ans])
    rf = pd.DataFrame(result_list, columns=['iid', 'question', 'type', 'label', 'pred'])
    rf.to_csv("VQA.csv", mode='w')

    o_accuracy = o_acc / o_total
    c_accuracy = c_acc / c_total
    t_accuracy = t_acc / t_total
    print('OPEN accuracy: {}'.format(o_accuracy))
    print('CLOSED accuracy: {}'.format(c_accuracy))
    print('TOTAL accuracy: {}'.format(t_accuracy))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/VQA.yaml')
    parser.add_argument('--checkpoint', default='')
    parser.add_argument('--output_dir', default='output/vqa')
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