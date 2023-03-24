from functools import partial
from models.vit import VisionTransformer, interpolate_pos_embed, vit_base
from models.xbert import BertConfig, BertModel, BertLMHeadModel
from models.vit import VisionTransformer as VisionTransformer_deit
from health_multimodal.text.utils import get_cxr_bert
# from health_multimodal.image.model.model import get_biovil_resnet
from health_multimodal.image.data.transforms import create_chest_xray_transform_for_inference, create_chest_xray_transform_for_train
from health_multimodal.text.inference_engine import TextInferenceEngine

import torch
from torch import nn
import torch.nn.functional as F
from models.predictor import TextGenerator

import numpy as np


class XVLModel(nn.Module):
    def __init__(self,
                 config=None,
                 ):
        super().__init__()

        self.config = config
        self.distill = config['distill']

        self.visual_encoder = vit_base(
            img_size=(config['image_res'], config['image_res']),
            patch_size=config['patch_size'],
            drop_path_rate=config['drop_path'],
        )

        self.tokenizer, _ = get_cxr_bert()
        config_encoder = BertConfig.from_json_file(config['bert_config'])
        self.text_encoder = BertModel(config=config_encoder, add_pooling_layer=False)

        config_decoder = BertConfig.from_json_file(config['fusion_config'])
        config_decoder.fusion_layer = 0
        config_decoder.num_hidden_layers = 4
        self.fusion_decoder = BertLMHeadModel(config=config_decoder)

        self.beam_generator = TextGenerator(config, self.fusion_decoder,
                                            bos_token=config['bos_token'],
                                            eos_token=config['eos_token'])

        if self.distill:
            self.visual_encoder_m = vit_base(
            img_size=(config['image_res'], config['image_res']),
            patch_size=config['patch_size'],
            drop_path_rate=config['drop_path'],
        )

            self.text_encoder_m = BertModel(config=config_encoder, add_pooling_layer=False)
            self.fusion_decoder_m = BertLMHeadModel(config=config_decoder)
            self.model_pairs = [[self.visual_encoder, self.visual_encoder_m],
                                [self.text_encoder, self.text_encoder_m],
                                [self.fusion_decoder, self.fusion_decoder_m],
                                ]
            self.copy_params()
            self.momentum = config['momentum']

    def forward(self, image, stt_text, answer=None, alpha=0, k=None, weights=None, train=True):

        image_embeds_raw = self.visual_encoder(image)
        image_embeds = image_embeds_raw
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)

        if train:
            '''
            k: number of answers for each question
            weights: weight for each answer
            '''

            # Embed into tokens
            stt_text = self.tokenizer.batch_encode_plus(batch_text_or_text_pairs=stt_text,
                                                    add_special_tokens=True,
                                                    padding='longest',
                                                    return_tensors='pt').to(image.device)

            answer = self.tokenizer.batch_encode_plus(batch_text_or_text_pairs=answer,
                                                    add_special_tokens=True,
                                                    padding='longest',
                                                    return_tensors='pt').to(image.device)

            answer_targets = answer.input_ids.masked_fill(answer.input_ids == self.tokenizer.pad_token_id, -100)
            stt_text_output = self.text_encoder(stt_text.input_ids,
                                                attention_mask=stt_text.attention_mask,
                                                encoder_hidden_states=image_embeds,
                                                encoder_attention_mask=image_atts,
                                                return_dict=True)

            # stt_text_states = []
            # stt_text_atts   = []
            # for b, n in enumerate(k):
            #     stt_text_states += [stt_text_output.last_hidden_state[b]]
            #     stt_text_atts   += [stt_text.attention_mask[b]]
            # stt_text_states = torch.stack(stt_text_states, 0)
            # stt_text_atts   = torch.stack(stt_text_atts, 0)

            stt_text_states = stt_text_output.last_hidden_state
            stt_text_atts = stt_text.attention_mask

            if self.distill:
                with torch.no_grad():
                    self._momentum_update()
                    image_embeds_m_raw = self.visual_encoder_m(image)
                    image_embeds_m = image_embeds_m_raw
                    stt_text_output_m = self.text_encoder_m(stt_text.input_ids,
                                                            attention_mask=stt_text.attention_mask,
                                                            encoder_hidden_states=image_embeds_m,
                                                            encoder_attention_mask=image_atts,
                                                            return_dict=True)

                    # stt_text_states_m   = []
                    # for b, n in enumerate(k):
                    #     stt_text_states_m += [stt_text_output_m.last_hidden_state[b]]
                    # stt_text_states_m   = torch.stack(stt_text_states_m, 0)
                    stt_text_states_m = stt_text_output_m.last_hidden_state

                    logits_m = self.fusion_decoder_m(answer.input_ids,
                                                   attention_mask=answer.attention_mask,
                                                   encoder_hidden_states=stt_text_states_m,
                                                   encoder_attention_mask=stt_text_atts,
                                                   return_logits=True,
                                                   )
                corrected_output = self.fusion_decoder(answer.input_ids,
                                                     attention_mask=answer.attention_mask,
                                                     encoder_hidden_states=stt_text_states,
                                                     encoder_attention_mask=stt_text_atts,
                                                     labels=answer_targets,
                                                     return_dict=True,
                                                     soft_labels=F.softmax(logits_m, dim=-1),
                                                     reduction='none',
                                                     )
            else:
                corrected_output = self.fusion_decoder(answer.input_ids,
                                                     attention_mask=answer.attention_mask,
                                                     encoder_hidden_states=stt_text_states,
                                                     encoder_attention_mask=stt_text_atts,
                                                     labels=answer_targets,
                                                     return_dict=True,
                                                     reduction='none',
                                                     )
            loss = corrected_output.loss
            loss = loss.sum() / image.size(0)
            return loss

        else:
            fused_output = self.text_encoder(stt_text.input_ids,
                                             attention_mask=stt_text.attention_mask,
                                             encoder_hidden_states=image_embeds,
                                             encoder_attention_mask=image_atts,
                                             return_dict=True)

            fused_states = fused_output.last_hidden_state
            fused_atts = stt_text.attention_mask

            topk_ids, topk_probs = self.generation(fused_states, fused_atts)

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

    def interpolate_pos_encoding(self, ds_embed, x):
        npatch = x.shape[1]
        N = ds_embed.shape[1]
        patch_pos_embed = ds_embed[:, :]

        bs, ds, p, dim = patch_pos_embed.size()
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(-1, dim, ds)
        patch_pos_embed = nn.functional.interpolate(patch_pos_embed, scale_factor=npatch/N, mode='linear')
        patch_pos_embed = patch_pos_embed.view(bs, p, dim, -1).permute(0, 3, 1, 2)

        return patch_pos_embed


def tile(x, dim, n_tile):
    init_dim = x.size(dim)
    repeat_idx = [1] * x.dim()
    repeat_idx[dim] = n_tile
    x = x.repeat(*(repeat_idx))
    order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)]))
    return torch.index_select(x, dim, order_index.to(x.device))



