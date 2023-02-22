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
import torchmetrics


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
def evaluation(model, data_loader, dataset, tokenizer, device, config):
    # test
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Evaluation:'

    print('Computing features for evaluation...')
    start_time = time.time()

    label_mapping = dataset.labels

    # Calculate entire text features (len = 40)
    with torch.no_grad():
        text_queries = dataset.text_queries
        text = text_queries
        text_input = tokenizer(text, padding='max_length', truncation=True, max_length=120, return_tensors="pt").to(
            device)
        text_output = model.text_encoder(text_input.input_ids, attention_mask=text_input.attention_mask, return_dict=True,
                                         mode='text')
        text_feat = text_output.last_hidden_state
        text_embed = F.normalize(model.text_proj(text_feat[:, 0, :]), dim=-1)

        text_embeds = text_embed
        text_feats = text_feat
        text_atts = text_input.attention_mask

    image_feat_total = []
    image_embed_total = []
    batch_labels = []

    # calculate entire image features (len = 1600)
    for image, label, img_id in data_loader:
        image = image.to(device)

        with torch.no_grad():
            image_feat = model.visual_encoder(image)
            image_embed = model.vision_proj(image_feat[:, 0, :])
            image_embed = F.normalize(image_embed, dim=-1)

            image_feat_total.append(image_feat)
            image_embed_total.append(image_embed)
            batch_labels.append(label[0])

    image_feat_total = torch.cat(image_feat_total, dim=0)
    image_embed_total = torch.cat(image_embed_total, dim=0)

    image_feats = image_feat_total
    image_embeds = image_embed_total

    sims_matrix = text_embeds @ image_embeds.t()

    num_tasks = utils.get_world_size()
    rank = utils.get_rank()

    score_matrix_t2i = torch.full((len(text), len(image_feat_total)), -100.0).to(device)

    step = sims_matrix.size(0) // num_tasks + 1
    start = rank * step
    end = min(sims_matrix.size(0), start + step)

    for i, sims in enumerate(metric_logger.log_every(sims_matrix[start:end], 50, header)):
        topk_sim, topk_idx = sims.topk(k=config['k_test'], dim=0)
        encoder_output = image_feats[topk_idx]
        encoder_att = torch.ones(encoder_output.size()[:-1], dtype=torch.long).to(device)
        with torch.no_grad():
            output_t = model.text_encoder(encoder_embeds=text_feats[start + i].repeat(config['k_test'], 1, 1),
                                          attention_mask=text_atts[start + i].repeat(config['k_test'], 1),
                                          encoder_hidden_states=encoder_output,
                                          encoder_attention_mask=encoder_att,
                                          return_dict=True,
                                          mode='fusion'
                                          )
            output_v = model.text_encoder(encoder_embeds=encoder_output,
                                          attention_mask=encoder_att,
                                          encoder_hidden_states=text_feats[start + i].repeat(config['k_test'], 1, 1),
                                          encoder_attention_mask=text_atts[start + i].repeat(config['k_test'], 1),
                                          return_dict=True,
                                          mode='fusion'
                                          )
        score_t = model.itm_head_t(output_t.last_hidden_state[:, 0, :])[:, 1]
        score_v = model.itm_head_v(output_v.last_hidden_state[:, 0, :])[:, 1]
        score = (score_t + score_v) / 2.
        score_matrix_t2i[start + i, topk_idx] = score

        if args.distributed:
            dist.barrier()
            torch.distributed.all_reduce(score_matrix_t2i, op=torch.distributed.ReduceOp.SUM)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Evaluation time {}'.format(total_time_str))

    score_t2i = score_matrix_t2i.cpu().numpy()

    eval_result = itm_eval(score_t2i, text_queries, label_mapping, batch_labels)

    prec_5 = eval_result['prec@5']  # text retrieval (i2t)
    prec_10 = eval_result['prec@10']  # image retrieval (t2i)
    prec_50 = eval_result['prec@50']  # text retrieval (i2t)

    return prec_5, prec_10, prec_50


@torch.no_grad()
def itm_eval(scores_t2i, text_queries, label_mapping, batch_labels):

    # Text->Images
    total_p5 = []
    total_p10 = []
    total_p50 = []

    batch_labels = np.array(batch_labels)

    # total len = 40
    for index, score in enumerate(scores_t2i):

        # total len = 1600
        preds = score
        label = label_mapping[text_queries[index]]
        targets = batch_labels == label

        preds = torch.from_numpy(preds)
        targets = torch.from_numpy(targets)

        p5 = torchmetrics.functional.retrieval_precision(preds, targets, k=5)
        p10 = torchmetrics.functional.retrieval_precision(preds, targets, k=10)
        p50 = torchmetrics.functional.retrieval_precision(preds, targets, k=50)

        total_p5.append(p5)
        total_p10.append(p10)
        total_p50.append(p50)

    prec_5 = np.array(total_p5).mean()
    prec_10 = np.array(total_p10).mean()
    prec_50 = np.array(total_p50).mean()

    eval_result = {'prec@5': prec_5,
                   'prec@10': prec_10,
                   'prec@50': prec_50}
    return eval_result


def main(args, config):
    utils.init_distributed_mode(args)

    device = torch.device(args.device)

    cudnn.benchmark = True

    #### Dataset ####
    print("Creating retrieval dataset")
    test_dataset = create_dataset('retrieval', config)

    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
    samplers = [None]

    test_loader = create_loader([test_dataset], samplers,
                                  batch_size=[config['batch_size_test']],
                                  num_workers=[4],
                                  is_trains=[False],
                                  collate_fns=[None])[0]

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

    print("Image-to-text and Text-to-image retrieval")

    test_txt_p5, test_txt_p10, test_txt_p50 = evaluation(model_without_ddp, test_loader, test_dataset, tokenizer, device, config)

    dist.barrier()

    print('test t2i_P5: %.3f, test t2i_P10: %.3f, test t2i_P50: %.3f' % (test_txt_p5, test_txt_p10, test_txt_p50))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/Retrieval.yaml')
    parser.add_argument('--output_dir', default='output/Classification')
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
