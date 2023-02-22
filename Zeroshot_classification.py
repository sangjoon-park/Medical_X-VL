import argparse
import os

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"
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

from models.model_retrieval import XVLModel
from models.vit import interpolate_pos_embed
from models.tokenization_bert import BertTokenizer
from transformers import AutoTokenizer

import utils
from dataset import create_dataset, create_sampler, create_loader, re_collate_fn
from scheduler import create_scheduler
from optim import create_optimizer
from fuzzywuzzy import fuzz
from dataset.utils import pre_caption, pre_question

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

    total_scores = []
    total_labels = []

    if config['data'] == 'pneumonia':
        reports = ['No evidence of pneumonia.', 'Findings suggesting pneumonia.']
    elif config['data'] == 'pneumothorax':
        reports = ['No pneumothorax.', 'Mediastinal shift, absent lung marking, collapsed lung is noted, findings suggesting pneumothorax.']
    elif config['data'] == 'covid':
        reports = ['No evidence of opacity or consolidation.', 'Bilateral peripheral reticular pattern, ground-glass opacities and consolidations, with rounded morphology and a confluent or patchy multifocal distribution with a predominance in the lower fields is observed.']
        # reports = ['No evidence of covid-19 infection.', 'Findings suggesting covid-19 infection.']

    # eval every batch
    for image, label, image_id in data_loader:
        image = image.to(device)

        with torch.no_grad():
            image_feat = model.visual_encoder(image)
            image_embed = model.vision_proj(image_feat[:, 0, :])
            image_embed = F.normalize(image_embed, dim=-1)

        image_feats = image_feat
        image_embeds = image_embed

        text_neg = pre_caption(reports[0], max_words=120)
        text_pos = pre_caption(reports[1], max_words=120)

        # Negative score
        text_input = tokenizer(text_neg, padding='max_length', truncation=True, max_length=90, return_tensors="pt").to(
            device)
        with torch.no_grad():
            text_output = model.text_encoder(text_input.input_ids, attention_mask=text_input.attention_mask,
                                             return_dict=True, mode='text')
            text_feat = text_output.last_hidden_state
            text_embed = F.normalize(model.text_proj(text_feat[:, 0, :]), dim=-1)

            text_embeds = text_embed
            text_feats = text_feat
            text_atts = text_input.attention_mask

            encoder_output = image_feats
            encoder_att = torch.ones(encoder_output.size()[:-1], dtype=torch.long).to(device)

            output_t = model.text_encoder(encoder_embeds=text_feats,
                                          attention_mask=text_atts,
                                          encoder_hidden_states=encoder_output,
                                          encoder_attention_mask=encoder_att,
                                          return_dict=True,
                                          mode='fusion'
                                          )
            output_v = model.text_encoder(encoder_embeds=encoder_output,
                                          attention_mask=encoder_att,
                                          encoder_hidden_states=text_feats,
                                          encoder_attention_mask=text_atts,
                                          return_dict=True,
                                          mode='fusion'
                                          )
            score_neg_t = model.itm_head_t(output_t.last_hidden_state[:, 0, :])[:, 1]
            score_neg_v = model.itm_head_v(output_v.last_hidden_state[:, 0, :])[:, 1]
            score_neg = (score_neg_t + score_neg_v) / 2.

        # Positive score
        text_input = tokenizer(text_pos, padding='max_length', truncation=True, max_length=90, return_tensors="pt").to(
            device)
        with torch.no_grad():
            text_output = model.text_encoder(text_input.input_ids, attention_mask=text_input.attention_mask,
                                             return_dict=True, mode='text')
            text_feat = text_output.last_hidden_state
            text_embed = F.normalize(model.text_proj(text_feat[:, 0, :]), dim=-1)

            text_embeds = text_embed
            text_feats = text_feat
            text_atts = text_input.attention_mask

            encoder_output = image_feats
            encoder_att = torch.ones(encoder_output.size()[:-1], dtype=torch.long).to(device)

            output_t = model.text_encoder(encoder_embeds=text_feats,
                                          attention_mask=text_atts,
                                          encoder_hidden_states=encoder_output,
                                          encoder_attention_mask=encoder_att,
                                          return_dict=True,
                                          mode='fusion'
                                          )
            output_v = model.text_encoder(encoder_embeds=encoder_output,
                                          attention_mask=encoder_att,
                                          encoder_hidden_states=text_feats,
                                          encoder_attention_mask=text_atts,
                                          return_dict=True,
                                          mode='fusion'
                                          )
            score_pos_t = model.itm_head_t(output_t.last_hidden_state[:, 0, :])[:, 1]
            score_pos_v = model.itm_head_v(output_v.last_hidden_state[:, 0, :])[:, 1]
            score_pos = (score_pos_t + score_pos_v) / 2.

        score = torch.cat([score_neg, score_pos], dim=-1)
        score = F.softmax(score)
        total_scores.append(score.cpu().detach().numpy())
        total_labels.append(label.cpu().detach().numpy())

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Evaluation time {}'.format(total_time_str))

    AUC = roc_auc_score(y_true=np.array(total_labels), y_score=np.array(total_scores)[:,1])
    print('AUC:{}'.format(AUC))

    return AUC


def main(args, config):
    utils.init_distributed_mode(args)

    device = torch.device(args.device)

    cudnn.benchmark = True

    #### Dataset ####
    print("Creating retrieval dataset")
    train_dataset, test_dataset = create_dataset('cls', config)

    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        samplers = create_sampler([train_dataset], [True], num_tasks, global_rank) + [None]
    else:
        samplers = [None, None]

    train_loader, test_loader = create_loader([train_dataset, test_dataset], samplers,
                                                          batch_size=[config['batch_size_train']] + [
                                                              config['batch_size_test']],
                                                          num_workers=[4, 4],
                                                          is_trains=[True, False],
                                                          collate_fns=[None, None])

    url = "microsoft/BiomedVLP-CXR-BERT-specialized"
    tokenizer = AutoTokenizer.from_pretrained(url, trust_remote_code=True)

    #### Model ####
    print("Creating model")
    model = XVLModel(config=config, tokenizer=tokenizer)

    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        state_dict = checkpoint['model']

        # reshape positional embedding to accomodate for image resolution change
        pos_embed_reshaped = interpolate_pos_embed(state_dict['visual_encoder.pos_embed'],
                                                   model.visual_encoder)
        state_dict['visual_encoder.pos_embed'] = pos_embed_reshaped
        m_pos_embed_reshaped = interpolate_pos_embed(state_dict['visual_encoder_m.pos_embed'],
                                                     model.visual_encoder_m)
        state_dict['visual_encoder_m.pos_embed'] = m_pos_embed_reshaped
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

    print("Zero-shot classification")
    start_time = time.time()

    auc = evaluation(model_without_ddp, test_loader, tokenizer, device, config)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Evaluation time {}'.format(total_time_str))

    print('pneumonia classification AUC: %.3f' % (auc))
    print('-------------------------------------------')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/Classification.yaml')
    parser.add_argument('--output_dir', default='output/Retrieval')
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
