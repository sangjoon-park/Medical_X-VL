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

from models.model_retrieval import XVLModel
from models.vit import interpolate_pos_embed
from models.tokenization_bert import BertTokenizer
from transformers import AutoTokenizer

import utils
from dataset import create_dataset, create_sampler, create_loader, re_collate_fn
from scheduler import create_scheduler
from optim import create_optimizer
from fuzzywuzzy import fuzz


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

    txt_r1s = []
    img_r1s = []
    txt_r5s = []
    img_r5s = []
    txt_r10s = []
    img_r10s = []

    # eval every batch
    for image, img_id, label, caption, batch_labels in data_loader:
        image = image.to(device)
        image_feat = model.visual_encoder(image)
        image_embed = model.vision_proj(image_feat[:, 0, :])
        image_embed = F.normalize(image_embed, dim=-1)

        image_feats = image_feat
        image_embeds = image_embed

        text = caption
        text_input = tokenizer(text, padding='max_length', truncation=True, max_length=120, return_tensors="pt").to(
            device)
        with torch.no_grad():
            text_output = model.text_encoder(text_input.input_ids, attention_mask=text_input.attention_mask,
                                             mode='text')
        text_feat = text_output.last_hidden_state
        text_embed = F.normalize(model.text_proj(text_feat[:, 0, :]))

        text_embeds = text_embed
        text_feats = text_feat
        text_atts = text_input.attention_mask

        sims_matrix = image_embeds @ text_embeds.t()
        score_matrix_i2t = torch.full((len(image), len(text)), -100.0).to(device)

        num_tasks = utils.get_world_size()
        rank = utils.get_rank()
        step = sims_matrix.size(0) // num_tasks + 1
        start = rank * step
        end = min(sims_matrix.size(0), start + step)

        for i, sims in enumerate(metric_logger.log_every(sims_matrix[start:end], 50, header)):
            topk_sim, topk_idx = sims.topk(k=config['k_test'], dim=0)

            encoder_output = image_feats[start + i].repeat(config['k_test'], 1, 1)
            encoder_att = torch.ones(encoder_output.size()[:-1], dtype=torch.long).to(device)
            with torch.no_grad():
                output_t = model.text_encoder(encoder_embeds=text_feats[topk_idx],
                                              attention_mask=text_atts[topk_idx],
                                              encoder_hidden_states=encoder_output,
                                              encoder_attention_mask=encoder_att,
                                              return_dict=True,
                                              mode='fusion'
                                              )
                output_v = model.text_encoder(encoder_embeds=encoder_output,
                                              attention_mask=encoder_att,
                                              encoder_hidden_states=text_feats[topk_idx],
                                              encoder_attention_mask=text_atts[topk_idx],
                                              return_dict=True,
                                              mode='fusion')

            score_t = model.itm_head_t(output_t.last_hidden_state[:, 0, :])[:, 1]
            score_v = model.itm_head_v(output_v.last_hidden_state[:, 0, :])[:, 1]
            score = (score_t + score_v) / 2
            score_matrix_i2t[start + i, topk_idx] = score

        sims_matrix = sims_matrix.t()
        score_matrix_t2i = torch.full((len(text), len(image)), -100.0).to(device)

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
                                              encoder_hidden_states=text_feats[start + i].repeat(config['k_test'],
                                                                                                 1, 1),
                                              encoder_attention_mask=text_atts[start + i].repeat(config['k_test'],
                                                                                                 1),
                                              return_dict=True,
                                              mode='fusion')

            score_t = model.itm_head_t(output_t.last_hidden_state[:, 0, :])[:, 1]
            score_v = model.itm_head_v(output_v.last_hidden_state[:, 0, :])[:, 1]
            score = (score_t + score_v) / 2
            score_matrix_t2i[start + i, topk_idx] = score

        if args.distributed:
            dist.barrier()
            torch.distributed.all_reduce(score_matrix_i2t, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(score_matrix_t2i, op=torch.distributed.ReduceOp.SUM)

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Evaluation time {}'.format(total_time_str))

        score_val_i2t, score_val_t2i = score_matrix_i2t.cpu().numpy(), score_matrix_t2i.cpu().numpy()

        eval_result = itm_eval(score_val_i2t, score_val_t2i, data_loader.dataset.txt2img, data_loader.dataset.img2txt,
                               data_loader.dataset.label, label, caption, batch_labels)

        txt_r1 = eval_result['txt_r1']  # text retrieval (i2t)
        img_r1 = eval_result['img_r1']  # image retrieval (t2i)
        txt_r5 = eval_result['txt_r5']  # text retrieval (i2t)
        img_r5 = eval_result['img_r5']  # image retrieval (t2i)
        txt_r10 = eval_result['txt_r10']  # text retrieval (i2t)
        img_r10 = eval_result['img_r10']  # image retrieval (t2i)

        txt_r1s.append(txt_r1)
        img_r1s.append(img_r1)
        txt_r5s.append(txt_r5)
        img_r5s.append(img_r5)
        txt_r10s.append(txt_r10)
        img_r10s.append(img_r10)

    txt_r1s = np.array(txt_r1s).mean()  # i2t
    img_r1s = np.array(img_r1s).mean()  # t2i
    txt_r5s = np.array(txt_r5s).mean()  # i2t
    img_r5s = np.array(img_r5s).mean()  # t2i
    txt_r10s = np.array(txt_r10s).mean()  # i2t
    img_r10s = np.array(img_r10s).mean()  # t2i

    return txt_r1s, img_r1s, txt_r5s, img_r5s, txt_r10s, img_r10s


@torch.no_grad()
def itm_eval(scores_i2t, scores_t2i, txt2img, img2txt, labels, imgs, txts, batch_labels):
    # Images->Text
    ranks = np.zeros(scores_i2t.shape[0])
    i2t_results = {}

    for index, score in enumerate(scores_i2t):
        inds = np.argsort(score)[::-1]

        img_id = imgs[index]
        correct = txts[index]
        best_1 = txts[inds[0]]
        best_2 = txts[inds[1]]
        best_3 = txts[inds[2]]
        worst_1 = txts[inds[99]]

        i2t_results[img_id] = {'correct': correct, 'b1': best_1, 'b2': best_2, 'b3': best_3, 'w': worst_1}

        # Overalapping labels
        positives = np.zeros_like(inds)
        for x, ind in enumerate(inds):

            this = ''.join(batch_labels[img2txt[index][0]])
            cand = ''.join(batch_labels[ind])

            if fuzz.token_sort_ratio(this, cand) == 100:
                positives[x] = 1

        # Score
        rank = 1e20
        tmp = np.where(positives == 1)[0]
        tmp = min(tmp)
        if tmp < rank:
            rank = tmp
        ranks[index] = rank

    with open('./retireval_i2t.json', 'w') as f:
        json.dump(i2t_results, f)

    # Compute metrics
    tr1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    tr5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    tr10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)

    # Text->Images
    ranks = np.zeros(scores_t2i.shape[0])
    t2i_results = {}

    for index, score in enumerate(scores_t2i):
        inds = np.argsort(score)[::-1]

        txt_id = txts[index]
        correct = imgs[index]
        best_1 = imgs[inds[0]]
        best_2 = imgs[inds[1]]
        best_3 = imgs[inds[2]]
        worst_1 = imgs[inds[99]]

        t2i_results[txt_id] = {'correct': correct, 'b1': best_1, 'b2': best_2, 'b3': best_3, 'w': worst_1}

        # Overalapping labels
        positives = np.zeros_like(inds)
        for x, ind in enumerate(inds):

            this = ''.join(batch_labels[txt2img[index]])
            cand = ''.join(batch_labels[ind])

            if fuzz.token_sort_ratio(this, cand) == 100:
                positives[x] = 1

        # Score
        rank = 1e20
        tmp = np.where(positives == 1)[0]
        tmp = min(tmp)
        if tmp < rank:
            rank = tmp
        ranks[index] = rank

    with open('./retireval_t2i.json', 'w') as f:
        json.dump(t2i_results, f)

    # Compute metrics
    ir1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    ir5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    ir10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)

    tr_mean = (tr1 + tr5 + tr10) / 3
    ir_mean = (ir1 + ir5 + ir10) / 3
    r_mean = (tr_mean + ir_mean) / 2

    eval_result = {'txt_r1': tr1,
                   'txt_r5': tr5,
                   'txt_r10': tr10,
                   'txt_r_mean': tr_mean,
                   'img_r1': ir1,
                   'img_r5': ir5,
                   'img_r10': ir10,
                   'img_r_mean': ir_mean,
                   'r_mean': r_mean}
    return eval_result


def main(args, config):
    utils.init_distributed_mode(args)

    device = torch.device(args.device)

    val_txt_r1t = []
    val_img_r1t = []
    val_txt_r5t = []
    val_img_r5t = []
    val_txt_r10t = []
    val_img_r10t = []

    test_txt_r1t = []
    test_img_r1t = []
    test_txt_r5t = []
    test_img_r5t = []
    test_txt_r10t = []
    test_img_r10t = []

    for j in range(1):

        cudnn.benchmark = True

        #### Dataset ####
        print("Creating retrieval dataset")
        train_dataset, val_dataset, test_dataset = create_dataset('re', config)

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
                                                              collate_fns=[None, re_collate_fn, re_collate_fn])

        # tokenizer = BertTokenizer.from_pretrained(args.text_encoder)
        # tokenizer = AutoTokenizer.from_pretrained("./my_tokenizer/")
        tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

        #### Model ####
        print("Creating model")
        model = XVLModel(train_loader, config=config, tokenizer=tokenizer)

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

        print("Image-to-text and Text-to-image retrieval")
        start_time = time.time()
        for epoch in range(0, max_epoch):
            if not args.evaluate:
                if args.distributed:
                    train_loader.sampler.set_epoch(epoch)
                train_stats = train(model, train_loader, optimizer, tokenizer, epoch, warmup_steps, device,
                                    lr_scheduler,
                                    config)

            val_txt_r1, val_img_r1, val_txt_r5, val_img_r5, val_txt_r10, val_img_r10 = evaluation(model_without_ddp,
                                                                                                  val_loader, tokenizer,
                                                                                                  device, config)
            test_txt_r1, test_img_r1, test_txt_r5, test_img_r5, test_txt_r10, test_img_r10 = evaluation(
                model_without_ddp,
                test_loader,
                tokenizer, device,
                config)

            val_txt_r1t.append(val_txt_r1)
            val_img_r1t.append(val_img_r1)
            val_txt_r5t.append(val_txt_r5)
            val_img_r5t.append(val_img_r5)
            val_txt_r10t.append(val_txt_r10)
            val_img_r10t.append(val_img_r10)

            test_txt_r1t.append(test_txt_r1)
            test_img_r1t.append(test_img_r1)
            test_txt_r5t.append(test_txt_r5)
            test_img_r5t.append(test_img_r5)
            test_txt_r10t.append(test_txt_r10)
            test_img_r10t.append(test_img_r10)

            val_r5 = (val_txt_r5 + val_img_r5) / 2.

            if utils.is_main_process():
                if val_r5 > best:
                    save_obj = {
                        'model': model_without_ddp.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'config': config,
                        'epoch': epoch,
                    }
                    torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint_best.pth'))
                    best = val_r5
                    best_epoch = epoch

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

    val_txt_r1t = np.array(val_txt_r1t).mean()  # i2t
    val_img_r1t = np.array(val_img_r1t).mean()  # t2i
    val_txt_r5t = np.array(val_txt_r5t).mean()  # i2t
    val_img_r5t = np.array(val_img_r5t).mean()  # t2i
    val_txt_r10t = np.array(val_txt_r10t).mean()  # i2t
    val_img_r10t = np.array(val_img_r10t).mean()  # t2i

    test_txt_r1t = np.array(test_txt_r1t).mean()  # i2t
    test_img_r1t = np.array(test_img_r1t).mean()  # t2i
    test_txt_r5t = np.array(test_txt_r5t).mean()  # i2t
    test_img_r5t = np.array(test_img_r5t).mean()  # t2i
    test_txt_r10t = np.array(test_txt_r10t).mean()  # i2t
    test_img_r10t = np.array(test_img_r10t).mean()  # t2i

    print('val i2t_R1: %.3f, val t2i_R1: %.3f' % (val_txt_r1t, val_img_r1t))
    print('test i2t_R1: %.3f, test t2i_R1: %.3f' % (test_txt_r1t, test_img_r1t))
    print('-------------------------------------------')
    print('val i2t_R5: %.3f, val t2i_R5: %.3f' % (val_txt_r5t, val_img_r5t))
    print('test i2t_R5: %.3f, test t2i_R5: %.3f' % (test_txt_r5t, test_img_r5t))
    print('-------------------------------------------')
    print('val i2t_R10: %.3f, val t2i_R10: %.3f' % (val_txt_r10t, val_img_r10t))
    print('test i2t_R10: %.3f, test t2i_R10: %.3f' % (test_txt_r10t, test_img_r10t))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/Retrieval.yaml')
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
