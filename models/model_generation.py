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
        fusion_config = BertConfig.from_json_file(config['fusion_config'])

        self.text_encoder = BertLMHeadModel(config=bert_config)
        self.fusion_encoder = BertLMHeadModel(config=fusion_config)

        self.beam_generator = TextGenerator(config, encoder=self.text_encoder, model=self.fusion_encoder)

        if self.distill:
            self.visual_encoder_m = vit_base(
                img_size=(config['image_res'], config['image_res']),
                patch_size=config['patch_size'],
                drop_path_rate=config['drop_path'],
                return_all_tokens=True,
            )
            self.text_encoder_m = BertLMHeadModel(config=bert_config)
            self.fusion_encoder_m = BertLMHeadModel(config=fusion_config)

            self.model_pairs = [
                # [self.visual_encoder, self.visual_encoder_m],
                                # [self.text_encoder, self.text_encoder_m],
                                [self.fusion_encoder, self.fusion_encoder_m],
                                ]
            self.copy_params()
            self.momentum = 0.995

        for param in self.text_encoder.parameters():
            param.requires_grad = False
        # for param in self.visual_encoder.parameters():
        #     param.requires_grad = False

        if self.distill:
            for param in self.text_encoder_m.parameters():
                param.requires_grad = False
            # for param in self.visual_encoder_m.parameters():
            #     param.requires_grad = False

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
                    fnd_output_m = self.text_encoder_m.bert(answer.input_ids,
                                                            attention_mask=answer.attention_mask,
                                                            return_dict=True,
                                                            is_decoder=True,
                                                            mode='text'
                                                            )
                    answer_output_m = self.fusion_encoder_m(encoder_embeds=fnd_output_m.last_hidden_state,
                                                            attention_mask=answer.attention_mask,
                                                            encoder_hidden_states=image_embeds_m,
                                                            encoder_attention_mask=image_atts,
                                                            return_dict=True,
                                                            is_decoder=True,
                                                            output_hidden_states=True,
                                                            mode='fusion'
                                                            )

                    logits_m = answer_output_m.logits[:, :-1, :].contiguous()

                fnd_output = self.text_encoder.bert(answer.input_ids,
                                                    attention_mask=answer.attention_mask,
                                                    return_dict=True,
                                                    is_decoder=True,
                                                    mode='text'
                                                    )
                answer_output = self.fusion_encoder(encoder_embeds=fnd_output.last_hidden_state,
                                                  attention_mask=answer.attention_mask,
                                                  encoder_hidden_states=image_embeds,
                                                  encoder_attention_mask=image_atts,
                                                  labels=answer_targets,
                                                  return_dict=True,
                                                  is_decoder=True,
                                                  soft_labels=F.softmax(logits_m, dim=-1),
                                                  alpha=alpha,
                                                  reduction='none',
                                                  output_hidden_states=True,
                                                  mode='fusion'
                                                  )

            else:
                fnd_output = self.text_encoder.bert(answer.input_ids,
                                                    attention_mask=answer.attention_mask,
                                                    is_decoder=True,
                                                    return_dict=True,
                                                    mode='text'
                                                    )
                answer_output = self.fusion_encoder(encoder_embeds=fnd_output.last_hidden_state,
                                                  attention_mask=answer.attention_mask,
                                                  encoder_hidden_states=image_embeds,
                                                  encoder_attention_mask=image_atts,
                                                  labels=answer_targets,
                                                  return_dict=True,
                                                  is_decoder=True,
                                                  reduction='none',
                                                  output_hidden_states=True,
                                                  mode='fusion'
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


def tile(x, dim, n_tile):
    init_dim = x.size(dim)
    repeat_idx = [1] * x.dim()
    repeat_idx[dim] = n_tile
    x = x.repeat(*(repeat_idx))
    order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)]))
    return torch.index_select(x, dim, order_index.to(x.device))

