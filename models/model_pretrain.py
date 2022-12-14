'''
 * Copyright (c) 2021, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see LICENSE.txt file in the repo root or https://opensource.org/licenses/BSD-3-Clause
'''

from functools import partial

import ibot_utils
from models.ibot_vit import VisionTransformer, interpolate_pos_embed, vit_base, vit_small
from models.xbert import BertConfig, BertForMaskedLM, BertModel

from models.vit import interpolate_pos_embed
from models.vit import VisionTransformer as VisionTransformer_deit

import torch
import torch.nn.functional as F
from torch import nn

import numpy as np
import random
from ibot_heads import iBOTHead
import math
import sys
from fuzzywuzzy import fuzz
from dataset.utils import pre_caption


class XVLModel(nn.Module):
    def __init__(self,
                 data_loader,
                 text_encoder=None,
                 tokenizer=None,
                 config=None,
                 temp=0.07,
                 init_deit=True
                 ):
        super().__init__()

        self.config = config
        self.data_loader = data_loader
        self.tokenizer = tokenizer
        self.mlm_probability = config['mlm_probability']
        embed_dim = config['embed_dim']

        # MeSH keyword weighting
        with open('./my_tokenizer/mesh.txt') as f:
            lines = f.readlines()
        processed_lines = []
        for line in lines:
            line = line.replace('\n', '')
            processed_lines.append(int(line))
        self.mesh = processed_lines

        visual_encoder = vit_small(
            img_size=(config['image_res'], config['image_res']),
            patch_size=config['patch_size'],
            drop_path_rate=config['drop_path'],
            return_all_tokens=True,
            masked_im_modeling=config['use_masked_im_modeling'],
        )
        self.visual_encoder = ibot_utils.MultiCropWrapper(visual_encoder, None)
        self.head = iBOTHead(visual_encoder.embed_dim,
                             config['out_dim'],
                             patch_out_dim=config['patch_out_dim'],
                             norm=config['norm_in_head'],
                             act=config['act_in_head'],
                             norm_last_layer=config['norm_last_layer'],
                             shared_head=config['shared_head'],
                             )

        state_dict = torch.load('/COVID_8TB/sangjoon/vision_language/checkpoint/small_checkpoint.pth')['student']
        pos_embed_reshaped = interpolate_pos_embed(state_dict['module.backbone.pos_embed'],
                                                   self.visual_encoder.backbone)
        state_dict['module.backbone.pos_embed'] = pos_embed_reshaped
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        state_dict = {k.replace("last_layer", ""): v for k, v in state_dict.items()}
        msg = self.visual_encoder.load_state_dict(state_dict, strict=False)
        print(msg)

        vision_width = config['vision_width']
        bert_config = BertConfig.from_json_file(config['bert_config'])

        self.text_encoder = BertForMaskedLM(config=bert_config)

        text_width = self.text_encoder.config.hidden_size
        self.vision_proj = nn.Linear(vision_width, embed_dim)
        self.text_proj = nn.Linear(text_width, embed_dim)

        self.temp = nn.Parameter(torch.ones([]) * config['temp'])
        self.queue_size = config['queue_size']
        self.momentum = config['momentum']

        self.itm_head_v = nn.Linear(text_width, 2)
        self.itm_head_t = nn.Linear(text_width, 2)

        # create momentum models
        visual_encoder_m = vit_small(
            img_size=(config['image_res'], config['image_res']),
            patch_size=config['patch_size'],
            return_all_tokens=True,
        )
        self.visual_encoder_m = ibot_utils.MultiCropWrapper(visual_encoder_m, None)
        self.head_m = iBOTHead(visual_encoder.embed_dim,
                               config['out_dim'],
                               patch_out_dim=config['patch_out_dim'],
                               norm=config['norm_in_head'],
                               act=config['act_in_head'],
                               shared_head=config['shared_head'],
                               )
        msg = self.visual_encoder_m.load_state_dict(state_dict, strict=False)
        print('Pre-trained weights loaded with {}'.format(msg))

        self.vision_proj_m = nn.Linear(vision_width, embed_dim)

        self.text_encoder_m = BertForMaskedLM(config=bert_config)
        self.text_proj_m = nn.Linear(text_width, embed_dim)

        self.model_pairs = [
            [self.vision_proj, self.vision_proj_m],
            [self.text_encoder, self.text_encoder_m],
            [self.text_proj, self.text_proj_m],
            [self.head, self.head_m],
        ]

        self.copy_params()

        # create the queue
        self.register_buffer("image_queue", torch.randn(embed_dim, self.queue_size))
        self.register_buffer("text_queue", torch.randn(embed_dim, self.queue_size))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        self.image_queue = nn.functional.normalize(self.image_queue, dim=0)
        self.text_queue = nn.functional.normalize(self.text_queue, dim=0)
        self.config = config
        self.momentum_schedule = ibot_utils.cosine_scheduler(config['momentum'], 1,
                                                             config['schedular']['epochs'], len(data_loader))

    def forward(self, images, text, masks, ibot_loss, epoch, fp16_scaler, alpha=0):
        bs = images[0].size(0)

        # common params
        names_q, params_q, names_k, params_k = [], [], [], []
        for name_q, param_q in self.visual_encoder.named_parameters():
            names_q.append(name_q)
            params_q.append(param_q)
        for name_k, param_k in self.visual_encoder_m.named_parameters():
            names_k.append(name_k)
            params_k.append(param_k)
        names_common = list(set(names_q) & set(names_k))
        params_q = [param_q for name_q, param_q in zip(names_q, params_q) if name_q in names_common]
        params_k = [param_k for name_k, param_k in zip(names_k, params_k) if name_k in names_common]

        with torch.no_grad():
            self.temp.clamp_(0.001, 0.5)

        with torch.cuda.amp.autocast(fp16_scaler is not None):

            text_output = self.text_encoder.bert(text.input_ids, attention_mask=text.attention_mask,
                                                 return_dict=True, mode='text')
            text_embeds = text_output.last_hidden_state
            text_feat = F.normalize(self.text_proj(text_embeds[:, 0, :]), dim=-1)

            with torch.no_grad():
                text_output_m = self.text_encoder_m.bert(text.input_ids, attention_mask=text.attention_mask,
                                                         return_dict=True, mode='text')
                text_embeds_m = text_output_m.last_hidden_state
                text_feat_m = F.normalize(self.text_proj_m(text_embeds_m[:, 0, :]), dim=-1)

            # get global views -> used for ita, itm, mlm
            with torch.no_grad():
                teacher_raw = self.visual_encoder_m(images[:self.config['global_crops_number']])
            student_raw = self.visual_encoder(images[:self.config['global_crops_number']],
                                              mask=masks[:self.config['global_crops_number']])

            with torch.no_grad():
                teacher_output_ = self.text_encoder_m.bert(encoder_embeds=teacher_raw,
                                                        attention_mask=torch.ones(
                                                            (teacher_raw.size(0), teacher_raw.size(1))).to(
                                                            teacher_raw.device),
                                                        encoder_hidden_states=torch.cat((text_embeds_m, text_embeds_m),
                                                                                        dim=0),
                                                        encoder_attention_mask=torch.cat(
                                                            (text.attention_mask, text.attention_mask), dim=0),
                                                        return_dict=True,
                                                        mode='fusion').last_hidden_state

            student_output_ = self.text_encoder.bert(encoder_embeds=student_raw,
                                                  attention_mask=torch.ones(
                                                      (student_raw.size(0), student_raw.size(1))).to(
                                                      student_raw.device),
                                                  encoder_hidden_states=torch.cat((text_embeds, text_embeds), dim=0),
                                                  encoder_attention_mask=torch.cat(
                                                      (text.attention_mask, text.attention_mask), dim=0),
                                                  return_dict=True,
                                                  mode='fusion').last_hidden_state

            # get local views
            self.visual_encoder.backbone.masked_im_modeling = False
            student_local_cls = self.visual_encoder(images[self.config['global_crops_number']:]) if len(images) > \
                                                                                                    self.config[
                                                                                                        'global_crops_number'] else None
            self.visual_encoder.backbone.masked_im_modeling = self.config['use_masked_im_modeling']

            # visual fusion ibot_head
            with torch.no_grad():
                teacher_output_cls = self.head_m(teacher_raw)[0]
                teacher_output_patch = self.head_m(teacher_output_)[1]
            student_output_cls = self.head(student_raw)[0]
            student_output_patch = self.head(student_output_)[1]
            student_local_cls_ = self.head(student_local_cls)[0]

            all_loss = ibot_loss((student_output_cls, student_output_patch), (teacher_output_cls, teacher_output_patch),
                                 student_local_cls_, masks, epoch)
            loss_ibot = all_loss.pop('loss')

        if not math.isfinite(loss_ibot.item()):
            print("Loss is {}, stopping training".format(loss_ibot.item()), force=True)
            sys.exit(1)

        image_embeds_m = teacher_raw[:bs]
        image_embeds = student_raw[:bs]

        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(images[0].device)

        image_feat = F.normalize(self.vision_proj(image_embeds[:, 0, :]), dim=-1)

        # get momentum features
        with torch.no_grad():
            # other models momentum update
            self._momentum_update()

            # student and teacher momentum update
            m = self.config['momentum']
            for param_q, param_k in zip(params_q, params_k):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

            image_feat_m = F.normalize(self.vision_proj_m(image_embeds_m[:, 0, :]), dim=-1)
            image_feat_all = torch.cat([image_feat_m.t(), self.image_queue.clone().detach()], dim=1)
            text_feat_all = torch.cat([text_feat_m.t(), self.text_queue.clone().detach()], dim=1)

            # image-text alignment (diagonal)
            sim_i2t_m = image_feat_m @ text_feat_all / self.temp
            sim_t2i_m = text_feat_m @ image_feat_all / self.temp

            sim_targets = torch.zeros(sim_i2t_m.size()).to(images[0].device)
            sim_targets.fill_diagonal_(1)

            sim_i2t_targets = alpha * F.softmax(sim_i2t_m, dim=1) + (1 - alpha) * sim_targets
            sim_t2i_targets = alpha * F.softmax(sim_t2i_m, dim=1) + (1 - alpha) * sim_targets

            # intramodal alignment (diagonal)
            sim_i2i_m = image_feat_m @ image_feat_all / self.temp
            sim_t2t_m = text_feat_m @ text_feat_all / self.temp

            sim_i2i_targets = alpha * F.softmax(sim_i2i_m, dim=1) + (1 - alpha) * sim_targets
            sim_t2t_targets = alpha * F.softmax(sim_t2t_m, dim=1) + (1 - alpha) * sim_targets

        sim_i2t = image_feat @ text_feat_all / self.temp
        sim_t2i = text_feat @ image_feat_all / self.temp

        loss_i2t = -torch.sum(F.log_softmax(sim_i2t, dim=1) * sim_i2t_targets, dim=1).mean()
        loss_t2i = -torch.sum(F.log_softmax(sim_t2i, dim=1) * sim_t2i_targets, dim=1).mean()

        # jinyu: add in-modality g2g loss (G2G)
        sim_i2i = image_feat @ image_feat_all / self.temp
        sim_t2t = text_feat @ text_feat_all / self.temp

        loss_i2i = -torch.sum(F.log_softmax(sim_i2i, dim=1) * sim_i2i_targets, dim=1).mean()
        loss_t2t = -torch.sum(F.log_softmax(sim_t2t, dim=1) * sim_t2t_targets, dim=1).mean()

        loss_ita = (loss_i2t + loss_t2i + loss_i2i + loss_t2t) / 4

        self._dequeue_and_enqueue(image_feat_m, text_feat_m)

        ###=================================###
        # forward the positve image-text pair
        output_pos = self.text_encoder.bert(encoder_embeds=text_embeds,
                                            attention_mask=text.attention_mask,
                                            encoder_hidden_states=image_embeds,
                                            encoder_attention_mask=image_atts,
                                            output_attentions=True,
                                            return_dict=True,
                                            mode='fusion',
                                            )

        with torch.no_grad():
            weights_i2t = F.softmax(sim_i2t[:, :bs], dim=1)
            weights_t2i = F.softmax(sim_t2i[:, :bs], dim=1)

            weights_i2t.fill_diagonal_(0)
            weights_t2i.fill_diagonal_(0)

        # select a negative image for each text
        image_embeds_neg = []
        for b in range(bs):
            try:
                neg_idx = torch.multinomial(weights_t2i[b], 1).item()
            except:
                neg_idx = torch.randint(0, bs, (1,)).item()
                while neg_idx == b:
                    neg_idx = torch.randint(0, bs, (1,)).item()
            image_embeds_neg.append(image_embeds[neg_idx])
        image_embeds_neg = torch.stack(image_embeds_neg, dim=0)

        # select a negative text for each image
        text_embeds_neg = []
        text_atts_neg = []
        for b in range(bs):
            try:
                neg_idx = torch.multinomial(weights_i2t[b], 1).item()
            except:
                neg_idx = torch.randint(0, bs, (1,)).item()
                while neg_idx == b:
                    neg_idx = torch.randint(0, bs, (1,)).item()
            text_embeds_neg.append(text_embeds[neg_idx])
            text_atts_neg.append(text.attention_mask[neg_idx])
        text_embeds_neg = torch.stack(text_embeds_neg, dim=0)
        text_atts_neg = torch.stack(text_atts_neg, dim=0)

        text_embeds_all = torch.cat([text_embeds, text_embeds_neg], dim=0)
        text_atts_all = torch.cat([text.attention_mask, text_atts_neg], dim=0)

        image_embeds_all = torch.cat([image_embeds_neg, image_embeds], dim=0)
        image_atts_all = torch.cat([image_atts, image_atts], dim=0)

        output_neg = self.text_encoder.bert(encoder_embeds=text_embeds_all,
                                            attention_mask=text_atts_all,
                                            encoder_hidden_states=image_embeds_all,
                                            encoder_attention_mask=image_atts_all,
                                            return_dict=True,
                                            mode='fusion',
                                            )

        # image negative output
        student_raw_all = torch.cat([image_embeds_neg, image_embeds], dim=0)
        student_output_neg = self.text_encoder.bert(encoder_embeds=student_raw_all,
                                                 attention_mask=torch.ones(
                                                     (student_raw_all.size(0), student_raw_all.size(1))).to(
                                                     student_raw_all.device),
                                                 encoder_hidden_states=text_embeds_all,
                                                 encoder_attention_mask=text_atts_all,
                                                 return_dict=True,
                                                 mode='fusion').last_hidden_state

        vl_embeddings_t = torch.cat([output_pos.last_hidden_state[:, 0, :], output_neg.last_hidden_state[:, 0, :]],
                                    dim=0)
        vl_embeddings_v = torch.cat([student_output_[:bs, 0, :], student_output_neg[:, 0, :]], dim=0)
        vl_output_t = self.itm_head_t(vl_embeddings_t)
        vl_output_v = self.itm_head_v(vl_embeddings_v)

        itm_labels = torch.cat([torch.ones(bs, dtype=torch.long), torch.zeros(2 * bs, dtype=torch.long)],
                               dim=0).to(images[0].device)
        loss_itm_t = F.cross_entropy(vl_output_t, itm_labels)
        loss_itm_v = F.cross_entropy(vl_output_v, itm_labels)
        loss_itm = loss_itm_t + loss_itm_v

        ##================= MLM ========================##
        input_ids = text.input_ids.clone()
        labels = input_ids.clone()

        # MeSH weighted probability matrix
        probability_matrix = torch.full(labels.shape, self.mlm_probability)

        for i in range(len(input_ids)):
            for j in range(len(input_ids[i])):
                if self.mesh[input_ids[i][j]] == 1:
                    probability_matrix[i][j] = probability_matrix[i][j] * 6.

        input_ids, labels = self.mask(input_ids, self.text_encoder.config.vocab_size, images[0].device, targets=labels,
                                      probability_matrix=probability_matrix)

        with torch.no_grad():
            logits_m = self.text_encoder_m(input_ids,
                                           attention_mask=text.attention_mask,
                                           encoder_hidden_states=image_embeds_m,
                                           encoder_attention_mask=image_atts,
                                           return_dict=True,
                                           return_logits=True,
                                           )
        mlm_output = self.text_encoder(input_ids,
                                       attention_mask=text.attention_mask,
                                       encoder_hidden_states=image_embeds,
                                       encoder_attention_mask=image_atts,
                                       return_dict=True,
                                       labels=labels,
                                       soft_labels=F.softmax(logits_m, dim=-1),
                                       alpha=alpha,
                                       )
        loss_mlm = mlm_output.loss

        return loss_mlm, loss_ita, loss_itm, loss_ibot

    @torch.no_grad()
    def copy_params(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data.copy_(param.data)  # initialize
                param_m.requires_grad = False  # not update by gradient

    @torch.no_grad()
    def _momentum_update(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data = param_m.data * self.momentum + param.data * (1. - self.momentum)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, image_feat, text_feat):
        # gather keys before updating queue
        image_feats = concat_all_gather(image_feat)
        text_feats = concat_all_gather(text_feat)

        batch_size = image_feats.shape[0]

        ptr = int(self.queue_ptr)
        assert self.queue_size % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.image_queue[:, ptr:ptr + batch_size] = image_feats.T
        self.text_queue[:, ptr:ptr + batch_size] = text_feats.T
        ptr = (ptr + batch_size) % self.queue_size  # move pointer

        self.queue_ptr[0] = ptr

    def mask(self, input_ids, vocab_size, device, targets=None, masked_indices=None, probability_matrix=None):
        if masked_indices is None:
            masked_indices = torch.bernoulli(probability_matrix).bool()

        masked_indices[input_ids == self.tokenizer.pad_token_id] = False
        masked_indices[input_ids == self.tokenizer.cls_token_id] = False

        if targets is not None:
            targets[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(input_ids.shape, 0.8)).bool() & masked_indices
        input_ids[indices_replaced] = self.tokenizer.mask_token_id

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(input_ids.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(vocab_size, input_ids.shape, dtype=torch.long).to(device)
        input_ids[indices_random] = random_words[indices_random]
        # The rest of the time (10% of the time) we keep the masked input tokens unchanged

        if targets is not None:
            return input_ids, targets
        else:
            return input_ids

    # jinyu: patch pooling of image patches to reduce computation and enlarge receptive field
    def patch_pooling(self, x):
        batch_size, seq_length, dim = x.size()
        b1 = int(np.sqrt(seq_length))
        x = x.reshape(batch_size, b1, b1, dim)
        x = x.permute(0, 3, 1, 2)
        c1 = int(np.sqrt(b1))
        x = F.avg_pool2d(x, c1, stride=c1)
        x = x.permute(0, 2, 3, 1).reshape(batch_size, c1 * c1, dim)
        return x

    # jinyu: in-batch g2l loss
    def in_batch_g2l_loss(self, l, m, temp, attention_mask=None):
        m = m.unsqueeze(1)
        N, n_locals, dim = l.size()
        l_n = l.reshape(-1, dim)  # (N * n_locals) * d
        m_n = m.reshape(-1, dim)  # N * d

        # Inner product for positive samples. Outer product for negative. We need to do it this way
        # for the multiclass loss. For the outer product, we want a N x N x n_locals x 1 tensor.
        u_p = torch.matmul(l, m.permute(0, 2, 1)).unsqueeze(2) / temp  # N * n_locals * 1 * 1

        # if l comes from text, then attention_mask is not None
        if attention_mask is not None:
            temp_mask = attention_mask.unsqueeze(2).unsqueeze(3)
            u_p = (temp_mask * u_p) + (10000. * (1 - temp_mask))

        u_n = torch.mm(m_n, l_n.t()) / temp
        u_n = u_n.reshape(N, 1, N, n_locals).permute(0, 2, 3, 1)  # N x N x n_locals x 1

        # We need to mask the diagonal part of the negative tensor.
        mask = torch.eye(N)[:, :, None, None].to(l.device)  # N*N*1*1
        n_mask = 1 - mask

        # Masking is done by shifting the diagonal before exp.
        u_n = (n_mask * u_n) - (10000. * (1 - n_mask))  # mask out "self" examples
        # if l comes from test, we mask out the padding tokens
        if attention_mask is not None:
            temp_mask = attention_mask.unsqueeze(0).unsqueeze(3).expand(N, -1, -1, -1)
            u_n = (temp_mask * u_n) - (10000. * (1 - temp_mask))

        u_n = u_n.reshape(N, N * n_locals, 1).unsqueeze(dim=1).expand(-1, n_locals, -1, -1)

        # Since this is multiclass, we concat the positive along the class dimension before performing log softmax.
        pred_lgt = torch.cat([u_p, u_n], dim=2)
        pred_log = F.log_softmax(pred_lgt, dim=2)

        # The positive score is the first element of the log softmax.
        if attention_mask is not None:
            loss = (torch.sum(-pred_log[:, :, 0].squeeze(), dim=1) / torch.sum(attention_mask, dim=1)).mean()
        else:
            loss = -pred_log[:, :, 0].mean()

        return loss

@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
                      for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output