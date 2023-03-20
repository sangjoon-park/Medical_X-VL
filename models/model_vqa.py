from functools import partial
from models.vit import VisionTransformer, interpolate_pos_embed, vit_base
from models.xbert import BertConfig, BertModel, BertLMHeadModel
from models.vit import VisionTransformer as VisionTransformer_deit
from health_multimodal.text.utils import get_cxr_bert
# from health_multimodal.image.model.model import get_biovil_resnet
from health_multimodal.image.data.transforms import create_chest_xray_transform_for_inference, create_chest_xray_transform_for_train
from health_multimodal.text.inference_engine import TextInferenceEngine
import clip

import torch
from torch import nn
import torch.nn.functional as F

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

            bos_token = answer
            seq = torch.cuda.LongTensor([bos_token]).repeat(fused_states.size(0), 1).to(image.device)  # bos token
            for i in range(self.config['max_words_length']):
                next_token = self.pred_next(fused_states, fused_atts, seq)
                seq = torch.cat([seq, next_token.unsqueeze(0)], dim=1)
                if next_token[0] == int(self.config['eos_token']):
                    break
            return seq

    def pred_next(self, fused_states, fused_atts, seq_ids):
        output          = self.text_decoder(seq_ids,
                                  encoder_hidden_states=fused_states,
                                  encoder_attention_mask=fused_atts,
                                  return_dict=True,
                                  reduction='none',
                                  )
        logits          = output.logits[:, -1, :]  # first token's logit
        prob_next_token = F.softmax(logits, dim=1)
        pred_next_token = torch.argmax(prob_next_token, dim=1)
        return pred_next_token

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