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
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.utils.data import DataLoader

from models.model_retrieval import ALBEF
from models.vit import interpolate_pos_embed
from models.tokenization_bert import BertTokenizer
from transformers import AutoTokenizer

import utils
from dataset import create_dataset, create_sampler, create_loader
from scheduler import create_scheduler
from optim import create_optimizer
from fuzzywuzzy import fuzz
from sklearn.metrics import roc_auc_score


def train(model, data_loader, optimizer, tokenizer, epoch, warmup_steps, device, scheduler, config):
    # train
    model.train()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('loss_itm', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_ita', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 50
    step_size = 100
    warmup_iterations = warmup_steps * step_size

    for i, (image, image_aug, text, idx, label) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        image = image.to(device, non_blocking=True)
        image_aug = image_aug.to(device, non_blocking=True)
        idx = idx.to(device, non_blocking=True)
        text_input = tokenizer(text, padding='longest', max_length=120, return_tensors="pt").to(device)

        if epoch > 0 or not config['warm_up']:
            alpha = config['alpha']
        else:
            alpha = config['alpha'] * min(1, i / len(data_loader))

        loss_ita, loss_itm = model(image, image_aug, text_input, label, alpha=alpha, idx=idx)
        loss = loss_ita + loss_itm

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        metric_logger.update(loss_itm=loss_itm.item())
        metric_logger.update(loss_ita=loss_ita.item())
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
    header = 'Evaluation:'

    print('Computing features for evaluation...')
    start_time = time.time()

    scores = []
    labels = []

    # eval every batch
    for image, img_id, label, caption, true_caption in data_loader:
        image = image.to(device)
        image_feat = model.visual_encoder(image)

        image_feats = image_feat

        text = caption
        text_input = tokenizer(text, padding='max_length', truncation=True, max_length=120, return_tensors="pt").to(
            device)
        with torch.no_grad():
            text_output = model.text_encoder(text_input.input_ids, attention_mask=text_input.attention_mask, mode='text')
        text_feat = text_output.last_hidden_state

        text_feats = text_feat
        text_atts = text_input.attention_mask

        with torch.no_grad():
            output_t = model.text_encoder(encoder_embeds=text_feats,
                                          attention_mask=text_atts,
                                          encoder_hidden_states=image_feats,
                                          encoder_attention_mask=torch.ones(image_feats.size()[:-1], dtype=torch.long).to(device),
                                          return_dict=True,
                                          mode='fusion'
                                          )
            output_v = model.text_encoder(encoder_embeds=image_feats,
                                          attention_mask=torch.ones(image_feats.size()[:-1], dtype=torch.long).to(device),
                                          encoder_hidden_states=text_feats,
                                          encoder_attention_mask=text_atts,
                                          return_dict=True,
                                          mode='fusion')
            score_t = model.itm_head_t(output_t.last_hidden_state[:, 0, :])[:, 1]
            score_v = model.itm_head_v(output_v.last_hidden_state[:, 0, :])[:, 1]
            score = (score_t + score_v) / 2

        scores.append(score.detach().cpu().numpy()[0])
        labels.extend(label.detach().cpu().numpy())

    scores = np.array(scores)
    labels = np.array(labels)

    return labels, scores


def main(args, config):
    utils.init_distributed_mode(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    #### Dataset ####
    print("Creating retrieval dataset")
    train_dataset, val_dataset, test_dataset = create_dataset('det', config)

    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        samplers = create_sampler([train_dataset], [True], num_tasks, global_rank) + [None, None]
    else:
        samplers = [None, None, None]

    train_loader, val_loader, test_loader = create_loader([train_dataset, val_dataset, test_dataset], samplers,
                                                          batch_size=[config['batch_size_train']] + [
                                                              config['batch_size_test']] * 2,
                                                          num_workers=[4, 4, 4],
                                                          is_trains=[True, False, False],
                                                          collate_fns=[None, None, None])

    tokenizer = AutoTokenizer.from_pretrained("./my_tokenizer/")

    #### Model ####
    print("Creating model")
    model = ALBEF(train_loader, config=config, text_encoder=args.text_encoder, tokenizer=tokenizer)

    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        state_dict = checkpoint['model']

        # reshape positional embedding to accomodate for image resolution change
        pos_embed_reshaped = interpolate_pos_embed(state_dict['visual_encoder.backbone.pos_embed'],
                                                   model.visual_encoder)
        state_dict['visual_encoder.backbone.pos_embed'] = pos_embed_reshaped
        m_pos_embed_reshaped = interpolate_pos_embed(state_dict['visual_encoder_m.backbone.pos_embed'],
                                                     model.visual_encoder_m)
        state_dict['visual_encoder_m.backbone.pos_embed'] = m_pos_embed_reshaped
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}

        for key in list(state_dict.keys()):
            if 'bert' in key:
                encoder_key = key.replace('bert.', '')
                state_dict[encoder_key] = state_dict[key]
                del state_dict[key]
        msg = model.load_state_dict(state_dict, strict=False)

        print('load checkpoint from %s' % args.checkpoint)
        print(msg)

    model = model.to(device)

    model.copy_params()
    print('model parameters are copied.')

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    arg_opt = utils.AttrDict(config['optimizer'])
    optimizer = create_optimizer(arg_opt, model)
    arg_sche = utils.AttrDict(config['schedular'])
    lr_scheduler, _ = create_scheduler(arg_sche, optimizer)

    max_epoch = config['schedular']['epochs']
    warmup_steps = config['schedular']['warmup_epochs']
    best = 0
    best_epoch = 0

    print("Start training")
    start_time = time.time()
    for epoch in range(0, max_epoch):
        if not args.evaluate:
            if args.distributed:
                train_loader.sampler.set_epoch(epoch)
            train_stats = train(model, train_loader, optimizer, tokenizer, epoch, warmup_steps, device, lr_scheduler,
                                config)

        # val_labels, val_scores = evaluation(model_without_ddp, val_loader, tokenizer, device, config)
        test_labels, test_scores = evaluation(model_without_ddp, test_loader, tokenizer, device, config)

        # AUC 가지고 계산하기 [Progression도 같은 요령으로?]
        # val_auc = roc_auc_score(y_true=val_labels, y_score=val_scores)
        test_auc = roc_auc_score(y_true=test_labels, y_score=test_scores)
        print(1 - test_auc)

        if args.evaluate:
            break

    lr_scheduler.step(epoch + warmup_steps + 1)
    dist.barrier()
    torch.cuda.empty_cache()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

    if utils.is_main_process():
        with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
            f.write("best epoch: %d" % best_epoch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/Detection.yaml')
    parser.add_argument('--output_dir', default='output/Detection')
    parser.add_argument('--checkpoint', default='')
    parser.add_argument('--text_encoder', default='bert-base-uncased')
    parser.add_argument('--evaluate', action='store_true')
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
