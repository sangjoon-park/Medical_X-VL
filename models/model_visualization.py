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


class ALBEF(nn.Module):
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
        embed_dim = config['embed_dim']

        # MeSH terms
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
        )
        self.visual_encoder = visual_encoder
        self.head = iBOTHead(visual_encoder.embed_dim,
                             config['out_dim'],
                             patch_out_dim=config['patch_out_dim'],
                             norm=config['norm_in_head'],
                             act=config['act_in_head'],
                             norm_last_layer=config['norm_last_layer'],
                             shared_head=config['shared_head'],
                             )

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
        self.visual_encoder_m = visual_encoder_m
        self.head_m = iBOTHead(visual_encoder.embed_dim,
                               config['out_dim'],
                               patch_out_dim=config['patch_out_dim'],
                               norm=config['norm_in_head'],
                               act=config['act_in_head'],
                               shared_head=config['shared_head'],
                               )

        # for p in self.visual_encoder_m.parameters():
        #     p.requires_grad = False

        self.vision_proj_m = nn.Linear(vision_width, embed_dim)

        self.text_encoder_m = BertForMaskedLM(config=bert_config)
        self.text_proj_m = nn.Linear(text_width, embed_dim)

        # config_decoder = BertConfig.from_json_file(config['bert_config'])
        # # self.vision_decoder = BertModel(config=config_decoder)
        # # self.vision_decoder_m = BertModel(config=config_decoder)

        self.model_pairs = [
            [self.vision_proj, self.vision_proj_m],
            [self.text_encoder, self.text_encoder_m],
            # [self.vision_decoder, self.vision_decoder_m],
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

    def forward(self, image, text):
        image_embeds = self.visual_encoder(image)
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image[0].device)

        ##================= MLM ========================##
        input_ids = text.input_ids

        # iter over word tokens
        word_length = torch.count_nonzero(input_ids)
        pred_tokens = [self.tokenizer.cls_token_id]

        for i in range(1, word_length):
            masked_input_ids = input_ids.clone()
            masked_input_ids[0][i] = self.tokenizer.mask_token_id

            with torch.no_grad():
                logits = self.text_encoder(masked_input_ids,
                                                   attention_mask=text.attention_mask,
                                                   encoder_hidden_states=image_embeds,
                                                   encoder_attention_mask=image_atts,
                                                   return_dict=True,
                                                   return_logits=True,
                                                   )
                probability = F.softmax(logits[0][i], dim=-1)
                pseudo_label = torch.argmax(probability, dim=-1)
                pred_tokens.append(int(pseudo_label))

        return pred_tokens

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