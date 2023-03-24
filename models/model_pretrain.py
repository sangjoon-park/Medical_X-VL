from functools import partial
from models.vit import VisionTransformer, interpolate_pos_embed, vit_base
from models.xbert import BertConfig, BertForMaskedLM

import torch
import torch.nn.functional as F
from torch import nn

import numpy as np
import random
from dataset.utils import split
from health_multimodal.text.utils import get_cxr_bert
# from health_multimodal.image.model.model import get_biovil_resnet
from health_multimodal.image.data.transforms import create_chest_xray_transform_for_inference, create_chest_xray_transform_for_train
from health_multimodal.text.inference_engine import TextInferenceEngine

from fuzzywuzzy import fuzz


def load_clip(model_path=None, context_length=77):
    '''
    FUNCTION: load_clip
    -------------------------------
    This function loads in a model with the CLIP model
    architecture.

    args:
        * model_path (optional) - path to model weights that the model
        will be initialized with
        * pretrained (optional) - if True, will load the pretrained
        CLIP model
        * context_length (optional) - length of the maximum number of
        tokens that can be inputted into the CLIP model
    '''

    params = {
        'embed_dim': 768,
        'image_resolution': 320,
        'vision_layers': 12,
        'vision_width': 768,
        'vision_patch_size': 16,
        'context_length': context_length,
        'vocab_size': 49408,
        'transformer_width': 512,
        'transformer_heads': 8,
        'transformer_layers': 12
    }

    # set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
    print("Loaded in pretrained model.")
    msg = model.load_state_dict(torch.load(model_path, map_location=device))
    print('load checkpoint: {}'.format(model_path))
    print(msg)
    return model


class XVLModel(nn.Module):
    def __init__(self,
                 config=None,
                 ):
        super().__init__()

        self.visual_encoder = vit_base(
            img_size=(config['image_res'], config['image_res']),
            patch_size=config['patch_size'],
            drop_path_rate=config['drop_path'],
        )

        # Load MIMIC pre-trained weights
        state_dict = torch.load('./pretrained.pth')['teacher']
        pos_embed_reshaped = interpolate_pos_embed(state_dict['backbone.pos_embed'],
                                                   self.visual_encoder)
        state_dict['backbone.pos_embed'] = pos_embed_reshaped
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        state_dict = {k.replace("last_layer", ""): v for k, v in state_dict.items()}
        msg = self.visual_encoder.load_state_dict(state_dict, strict=False)
        print(msg)

        self.tokenizer, self.text_encoder = get_cxr_bert()
        # self.text_engine = TextInferenceEngine(text_model=self.text_encoder, tokenizer=self.tokenizer)

        self.mlm_probability = config['mlm_probability']
        embed_dim = config['embed_dim']

        vision_width = config['vision_width']
        fusion_config = BertConfig.from_json_file(config['fusion_config'])

        self.fusion_encoder = BertForMaskedLM(config=fusion_config)

        text_width = self.text_encoder.config.hidden_size
        self.vision_proj = nn.Linear(vision_width, embed_dim)
        self.text_proj = nn.Linear(text_width, embed_dim)

        self.temp = nn.Parameter(torch.ones([]) * config['temp'])
        self.queue_size = config['queue_size']
        self.momentum = config['momentum']
        self.itm_head = nn.Linear(text_width, 2)

        # create momentum models
        self.visual_encoder_m = vit_base(
            img_size=(config['image_res'], config['image_res']),
            patch_size=config['patch_size'],
            drop_path_rate=config['drop_path'],
        )
        msg = self.visual_encoder.load_state_dict(state_dict, strict=False)
        print(msg)

        self.vision_proj_m = nn.Linear(vision_width, embed_dim)

        _, self.text_encoder_m = get_cxr_bert()
        self.fusion_encoder_m = BertForMaskedLM(config=fusion_config)

        self.text_proj_m = nn.Linear(text_width, embed_dim)

        self.model_pairs = [[self.visual_encoder, self.visual_encoder_m],
                            [self.vision_proj, self.vision_proj_m],
                            [self.text_encoder, self.text_encoder_m],
                            [self.fusion_encoder, self.fusion_encoder_m],
                            [self.text_proj, self.text_proj_m],
                            ]

        self.copy_params()

        # create the queue
        self.register_buffer("image_queue", torch.randn(embed_dim, self.queue_size))
        self.register_buffer("text_queue", torch.randn(embed_dim, self.queue_size))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        self.image_queue = nn.functional.normalize(self.image_queue, dim=0)
        self.text_queue = nn.functional.normalize(self.text_queue, dim=0)

        self.config = config

    def forward(self, image, image_aug, text, epoch, alpha=0):
        with torch.no_grad():
            self.temp.clamp_(0.001, 0.5)

        caption = text.copy()

        bs = image.size(0)
        # text = self.text_engine.tokenize_input_prompts(prompts=text, verbose=True).to(image.device)

        text = self.tokenizer.batch_encode_plus(batch_text_or_text_pairs=text,
                                                            add_special_tokens=True,
                                                            padding='longest',
                                                            return_tensors='pt').to(image.device)

        image_embeds = self.visual_encoder(image_aug)
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)

        image_feat = F.normalize(self.vision_proj(image_embeds[:, 0, :]), dim=-1)

        text_output = self.text_encoder(text.input_ids, attention_mask=text.attention_mask,
                                             return_dict=True)
        text_embeds = text_output.last_hidden_state
        text_feat = F.normalize(self.text_proj(text_embeds[:, 0, :]), dim=-1)

        # get momentum features
        with torch.no_grad():
            self._momentum_update()
            image_embeds_m = self.visual_encoder_m(image)
            image_feat_m = F.normalize(self.vision_proj_m(image_embeds_m[:, 0, :]), dim=-1)
            image_feat_all = torch.cat([image_feat_m.t(), self.image_queue.clone().detach()], dim=1)

            text_output_m = self.text_encoder_m(text.input_ids, attention_mask=text.attention_mask,
                                                     return_dict=True)
            text_feat_m = F.normalize(self.text_proj_m(text_output_m.last_hidden_state[:, 0, :]), dim=-1)

            # # jinyu: local features of text part
            # # text_feat_m_l = F.normalize(self.text_proj_m(text_output_m.last_hidden_state[:, 1:, :]), dim=-1)
            # text_feat_m_l = []
            # text_attention_mask_l_all = []
            #
            # lengths = []
            # for k in range(bs):
            #     lengths.append(len(split(caption[k])))
            # max_len = np.array(lengths).max()
            #
            # for j in range(bs):
            #     text_feat_m_l_ = []
            #     sentences = split(caption[j])
            #     sent_len = len(sentences)
            #     for i in range(sent_len):
            #         sentence = [sentences[i]]
            #         sentence = self.tokenizer.batch_encode_plus(batch_text_or_text_pairs=sentence,
            #                                                 add_special_tokens=True,
            #                                                 padding='longest',
            #                                                 return_tensors='pt').to(image.device)
            #         text_input_l, text_attention_mask_l = sentence.input_ids, sentence.attention_mask
            #         text_output_l = self.text_encoder_m(text_input_l, attention_mask=text_attention_mask_l,
            #                                                  return_dict=True)
            #         text_feat_l = F.normalize(self.text_proj_m(text_output_l.last_hidden_state[:, 0, :]), dim=-1)
            #         text_feat_l = text_feat_l.view(1, 1, text_feat_l.size(1))
            #         text_feat_m_l_.append(text_feat_l)
            #     text_feat_m_l_ = torch.cat(text_feat_m_l_, dim=1)
            #     if sent_len < max_len:
            #         text_feat_m_l_pad = torch.zeros(text_feat_m_l_.size(0), max_len - sent_len, text_feat_m_l_.size(2))
            #         text_feat_m_l_all = torch.cat([text_feat_m_l_, text_feat_m_l_pad.to(image.device)], dim=1)
            #     else:
            #         text_feat_m_l_all = text_feat_m_l_
            #     text_attention_mask = torch.zeros(max_len).to(image.device)
            #     text_attention_mask[:sent_len] = 1
            #     text_feat_m_l.append(text_feat_m_l_all)
            #     text_attention_mask_l_all.append(text_attention_mask)
            # text_feat_m_l = torch.cat(text_feat_m_l, dim=0)
            # text_attention_mask_l_all = torch.stack(text_attention_mask_l_all, dim=0)

            text_feat_all = torch.cat([text_feat_m.t(), self.text_queue.clone().detach()], dim=1)

            sim_i2t_m = image_feat_m @ text_feat_all / self.temp
            sim_t2i_m = text_feat_m @ image_feat_all / self.temp

            sim_targets = torch.zeros(sim_i2t_m.size()).to(image.device)
            sim_targets.fill_diagonal_(1)

            sim_i2t_targets = alpha * F.softmax(sim_i2t_m, dim=1) + (1 - alpha) * sim_targets
            sim_t2i_targets = alpha * F.softmax(sim_t2i_m, dim=1) + (1 - alpha) * sim_targets

        # loss_t2i_inMod_l = self.in_batch_g2l_loss(text_feat_m_l, image_feat, self.temp, text_attention_mask_l_all)

        sim_i2t = image_feat @ text_feat_all / self.temp
        sim_t2i = text_feat @ image_feat_all / self.temp

        loss_i2t = -torch.sum(F.log_softmax(sim_i2t, dim=1) * sim_i2t_targets, dim=1).mean()
        loss_t2i = -torch.sum(F.log_softmax(sim_t2i, dim=1) * sim_t2i_targets, dim=1).mean()

        # jinyu: add in-modality g2g loss
        sim_i2i = image_feat @ image_feat_all / self.temp
        sim_t2t = text_feat @ text_feat_all / self.temp

        loss_i2i = -torch.sum(F.log_softmax(sim_i2i, dim=1) * sim_targets, dim=1).mean()
        loss_t2t = -torch.sum(F.log_softmax(sim_t2t, dim=1) * sim_targets, dim=1).mean()

        loss_ita = (loss_i2t + loss_t2i + loss_i2i + loss_t2t) / 4.

        self._dequeue_and_enqueue(image_feat_m, text_feat_m)

        ###=================================###
        # forward the positve image-text pair
        output_pos = self.fusion_encoder.bert(encoder_embeds=text_embeds,
                                            attention_mask=text.attention_mask,
                                            encoder_hidden_states=image_embeds,
                                            encoder_attention_mask=image_atts,
                                            return_dict=True,
                                            mode='fusion',
                                            )
        with torch.no_grad():
            bs = image.size(0)
            weights_i2t = F.softmax(sim_i2t[:, :bs], dim=1)
            weights_t2i = F.softmax(sim_t2i[:, :bs], dim=1)

            weights_i2t.fill_diagonal_(0)
            weights_t2i.fill_diagonal_(0)

        # select a negative image for each text
        image_embeds_neg = []
        n = 0
        for b in range(bs):
            while True:
                try:
                    neg_idx = torch.multinomial(weights_t2i[b], 1).item()
                except:
                    neg_idx = torch.randint(0, bs, (1,)).item()
                    while neg_idx == b:
                        neg_idx = torch.randint(0, bs, (1,)).item()

                n += 1
                this = caption[b]
                cand = caption[neg_idx]

                if fuzz.token_sort_ratio(this, cand) < 80:
                    break
                if n > 100:
                    break

            image_embeds_neg.append(image_embeds[neg_idx])
        image_embeds_neg = torch.stack(image_embeds_neg, dim=0)

        # select a negative text for each image
        text_embeds_neg = []
        text_atts_neg = []

        n = 0
        for b in range(bs):
            while True:
                try:
                    neg_idx = torch.multinomial(weights_i2t[b], 1).item()
                except:
                    neg_idx = torch.randint(0, bs, (1,)).item()
                    while neg_idx == b:
                        neg_idx = torch.randint(0, bs, (1,)).item()

                n += 1
                this = caption[b]
                cand = caption[neg_idx]

                if fuzz.token_sort_ratio(this, cand) < 80:
                    break
                if n > 100:
                    break

            text_embeds_neg.append(text_embeds[neg_idx])
            text_atts_neg.append(text.attention_mask[neg_idx])
        text_embeds_neg = torch.stack(text_embeds_neg, dim=0)
        text_atts_neg = torch.stack(text_atts_neg, dim=0)

        text_embeds_all = torch.cat([text_embeds, text_embeds_neg], dim=0)
        text_atts_all = torch.cat([text.attention_mask, text_atts_neg], dim=0)

        image_embeds_all = torch.cat([image_embeds_neg, image_embeds], dim=0)
        image_atts_all = torch.cat([image_atts, image_atts], dim=0)

        output_neg = self.fusion_encoder.bert(encoder_embeds=text_embeds_all,
                                            attention_mask=text_atts_all,
                                            encoder_hidden_states=image_embeds_all,
                                            encoder_attention_mask=image_atts_all,
                                            return_dict=True,
                                            mode='fusion',
                                            )

        vl_embeddings = torch.cat([output_pos.last_hidden_state[:, 0, :], output_neg.last_hidden_state[:, 0, :]], dim=0)

        vl_output = self.itm_head(vl_embeddings)

        itm_labels = torch.cat([torch.ones(bs, dtype=torch.long), torch.zeros(2 * bs, dtype=torch.long)],
                               dim=0).to(image.device)
        loss_itm = F.cross_entropy(vl_output, itm_labels)

        ##================= MLM ========================##
        input_ids = text.input_ids.clone()
        labels = input_ids.clone()

        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        input_ids, labels = self.mask(input_ids, self.text_encoder.config.vocab_size, image.device, targets=labels,
                                      probability_matrix=probability_matrix)

        with torch.no_grad():
            text_output_m = self.text_encoder_m(input_ids, attention_mask=text.attention_mask,
                                                 return_dict=True)
            logits_m = self.fusion_encoder_m(encoder_embeds=text_output_m.last_hidden_state,
                                           attention_mask=text.attention_mask,
                                           encoder_hidden_states=image_embeds_m,
                                           encoder_attention_mask=image_atts,
                                           return_dict=True,
                                           return_logits=True,
                                           mode='fusion'
                                           )
        text_output = self.text_encoder.bert(input_ids, attention_mask=text.attention_mask,
                                               return_dict=True)
        mlm_output = self.fusion_encoder(encoder_embeds=text_output.last_hidden_state,
                                       attention_mask=text.attention_mask,
                                       encoder_hidden_states=image_embeds,
                                       encoder_attention_mask=image_atts,
                                       return_dict=True,
                                       labels=labels,
                                       soft_labels=F.softmax(logits_m, dim=-1),
                                       alpha=alpha,
                                       mode='fusion'
                                       )
        loss_mlm = mlm_output.loss

        return loss_mlm, loss_ita, loss_itm

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
        x = x.permute(0,3,1,2)
        # c1 = int(np.sqrt(b1))
        c = int(np.round(np.sqrt(b1)))
        c1 = int(np.sqrt(b1))
        x = F.avg_pool2d(x, c1, stride=c1)
        x = x.permute(0,2,3,1).reshape(batch_size, c*c, dim)
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
            loss = (torch.sum(-pred_log[:, :, 0].squeeze(-1), dim=1) / torch.sum(attention_mask, dim=1)).mean()
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