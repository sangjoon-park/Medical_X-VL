'''
 * Copyright (c) 2021, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see LICENSE.txt file in the repo root or https://opensource.org/licenses/BSD-3-Clause
'''

from functools import partial

import ibot_utils
from models.ibot_vit import VisionTransformer, interpolate_pos_embed, vit_base, vit_small
from models.xbert import BertConfig, BertLMHeadModel, BertModel

from models.vit import interpolate_pos_embed
from models.vit import VisionTransformer as VisionTransformer_deit

import torch
import torch.nn.functional as F
from torch import nn

from models.predictor import TextGenerator

import numpy as np
import random
from ibot_heads import iBOTHead
import math
import sys
from fuzzywuzzy import fuzz
from dataset.utils import pre_caption


class XVLModel(nn.Module):
    def __init__(self,
                 tokenizer=None,
                 config=None,
                 temp=0.07
                 ):
        super().__init__()

        self.tokenizer = tokenizer
        self.distill = config['distill']

        self.visual_encoder = vit_base(
            img_size=(config['image_res'], config['image_res']),
            patch_size=config['patch_size'],
            drop_path_rate=config['drop_path'],
            return_all_tokens=True,
        )

        bert_config = BertConfig.from_json_file(config['bert_config'])
        self.text_encoder = BertLMHeadModel(config=bert_config)

        self.beam_generator = TextGenerator(config, self.text_encoder)

        if self.distill:
            self.visual_encoder_m = vit_base(
            img_size=(config['image_res'], config['image_res']),
            patch_size=config['patch_size'],
            drop_path_rate=config['drop_path'],
            return_all_tokens=True,
        )
            self.text_encoder_m = BertLMHeadModel(config=bert_config)
            self.model_pairs = [[self.visual_encoder, self.visual_encoder_m],
                                [self.text_encoder, self.text_encoder_m],
                                [self.text_decoder, self.text_decoder_m],
                                ]
            self.copy_params()
            self.momentum = 0.995

    def forward(self, image, answer=None, train=True, alpha=0):

        bs = image.size(0)

        image_embeds = self.visual_encoder(image)

        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)

        if train:
            answer_targets = answer.input_ids.masked_fill(answer.input_ids == self.tokenizer.pad_token_id, -100)

            if self.distill:
                with torch.no_grad():
                    self._momentum_update()
                    image_embeds_m = self.visual_encoder_m(image)
                    answer_output_m = self.text_encoder_m(answer.input_ids,
                                                   attention_mask=answer.attention_mask,
                                                   encoder_hidden_states=image_embeds_m,
                                                   encoder_attention_mask=image_atts,
                                                   return_dict=True,
                                                   is_decoder=True,
                                                   output_hidden_states=True
                                                   )
                    logits_m = answer_output_m.logits[:, :-1, :].contiguous()

                answer_output = self.text_encoder(answer.input_ids,
                                                  attention_mask=answer.attention_mask,
                                                  encoder_hidden_states=image_embeds,
                                                  encoder_attention_mask=image_atts,
                                                  labels=answer_targets,
                                                  return_dict=True,
                                                  is_decoder=True,
                                                  soft_labels=F.softmax(logits_m, dim=-1),
                                                  alpha=alpha,
                                                  reduction='none',
                                                  output_hidden_states=True
                                                  )
            else:
                answer_output = self.text_encoder(answer.input_ids,
                                                  attention_mask=answer.attention_mask,
                                                  encoder_hidden_states=image_embeds,
                                                  encoder_attention_mask=image_atts,
                                                  labels=answer_targets,
                                                  return_dict=True,
                                                  is_decoder=True,
                                                  reduction='none',
                                                  output_hidden_states=True
                                                  )
            loss = answer_output.loss
            loss = loss.sum() / image.size(0)

            return loss

        else:
            topk_ids, topk_probs = self.generation(image_embeds, image_atts)

            return topk_ids, topk_probs

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

    def generation(self, question_states, question_atts, out_size=1):
        encoder_inputs = [question_states, question_atts]
        # topk_ids, topk_probs = self.beam_generator.translate_batch_scst(encoder_inputs, out_size=out_size)
        topk_ids, topk_probs = self.beam_generator.translate_batch(encoder_inputs, out_size=out_size)
        return topk_ids, topk_probs

    def rank_answer(self, question_states, question_atts, answer_ids, answer_atts, k):

        num_ques = question_states.size(0)
        start_ids = answer_ids[0, 0].repeat(num_ques, 1)  # bos token

        start_output = self.text_decoder(start_ids,
                                         encoder_hidden_states=question_states,
                                         encoder_attention_mask=question_atts,
                                         return_dict=True,
                                         reduction='none')
        logits = start_output.logits[:, 0, :]  # first token's logit

        # topk_probs: top-k probability
        # topk_ids: [num_question, k]
        answer_first_token = answer_ids[:, 1]
        prob_first_token = F.softmax(logits, dim=1).index_select(dim=1, index=answer_first_token)
        topk_probs, topk_ids = prob_first_token.topk(k, dim=1)

        # answer input: [num_question*k, answer_len]
        input_ids = []
        input_atts = []
        for b, topk_id in enumerate(topk_ids):
            input_ids.append(answer_ids.index_select(dim=0, index=topk_id))
            input_atts.append(answer_atts.index_select(dim=0, index=topk_id))
        input_ids = torch.cat(input_ids, dim=0)
        input_atts = torch.cat(input_atts, dim=0)

        targets_ids = input_ids.masked_fill(input_ids == self.tokenizer.pad_token_id, -100)

        # repeat encoder's output for top-k answers
        question_states = tile(question_states, 0, k)
        question_atts = tile(question_atts, 0, k)

        output = self.text_decoder(input_ids,
                                   attention_mask=input_atts,
                                   encoder_hidden_states=question_states,
                                   encoder_attention_mask=question_atts,
                                   labels=targets_ids,
                                   return_dict=True,
                                   reduction='none')

        answer_loss = output.loss
        answer_loss = answer_loss.view(input_ids.size(0), -1)

        # topk_prob: first token probability
        topk_probs = topk_probs.view(-1, 1)
        log_probs = torch.cat([topk_probs.log(), -answer_loss], dim=1)

        # re-calculate log probabilities for the answer sequences using chain rule
        log_probs_sum = log_probs.sum(1)
        log_probs_sum = log_probs_sum.view(num_ques, k)

        topk_probs = F.softmax(log_probs_sum, dim=-1)
        # get top-k after re-ranking
        topk_probs, rerank_id = topk_probs.topk(k, dim=1)
        topk_ids = torch.gather(topk_ids, 1, rerank_id)

        return topk_ids, topk_probs

def tile(x, dim, n_tile):
    init_dim = x.size(dim)
    repeat_idx = [1] * x.dim()
    repeat_idx[dim] = n_tile
    x = x.repeat(*(repeat_idx))
    order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)]))
    return torch.index_select(x, dim, order_index.to(x.device))

