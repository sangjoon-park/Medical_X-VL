from functools import partial
from models.ibot_vit import VisionTransformer, interpolate_pos_embed, vit_base, vit_small
from models.xbert import BertConfig, BertModel

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from fuzzywuzzy import fuzz


class ALBEF(nn.Module):
    def __init__(self,
                 train_loader,
                 text_encoder=None,
                 tokenizer=None,
                 config=None,
                 ):
        super().__init__()

        self.train_loader = train_loader
        self.tokenizer = tokenizer
        self.distill = config['distill']
        embed_dim = config['embed_dim']
        vision_width = config['vision_width']

        visual_encoder = vit_small(
            img_size=(config['image_res'], config['image_res']),
            patch_size=config['patch_size'],
            drop_path_rate=config['drop_path'],
            return_all_tokens=True,
        )
        self.visual_encoder = visual_encoder

        bert_config = BertConfig.from_json_file(config['bert_config'])
        if not config['fusion']:
            bert_config.fusion_layer = 12
        self.text_encoder = BertModel(config=bert_config, add_pooling_layer=False)

        text_width = self.text_encoder.config.hidden_size
        self.vision_proj = nn.Linear(vision_width, embed_dim)
        self.text_proj = nn.Linear(text_width, embed_dim)

        # config_decoder = BertConfig.from_json_file(config['bert_config'])
        # if not config['fusion']:
        #     config_decoder.fusion_layer = 12
        # self.vision_decoder = BertModel(config=config_decoder)

        self.temp = nn.Parameter(torch.ones([]) * config['temp'])
        self.queue_size = config['queue_size']
        self.momentum = config['momentum']

        # create momentum models
        visual_encoder_m = vit_small(
            img_size=(config['image_res'], config['image_res']),
            patch_size=config['patch_size'],
            return_all_tokens=True,
        )
        self.visual_encoder_m = visual_encoder_m

        # checkpoint = torch.hub.load_state_dict_from_url(
        #     url="https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth",
        #     map_location="cpu", check_hash=True)
        # state_dict = checkpoint["model"]
        # pos_embed_reshaped = interpolate_pos_embed(state_dict['pos_embed'], self.visual_encoder)
        # state_dict['pos_embed'] = pos_embed_reshaped
        # msg = self.visual_encoder.load_state_dict(state_dict, strict=False)
        # print(msg)

        # state_dict = torch.load('/COVID_8TB/sangjoon/vision_language/checkpoint/small_checkpoint.pth')['student']
        # pos_embed_reshaped = interpolate_pos_embed(state_dict['module.backbone.pos_embed'],
        #                                            self.visual_encoder)
        # state_dict['module.backbone.pos_embed'] = pos_embed_reshaped
        # state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        # state_dict = {k.replace("last_layer", ""): v for k, v in state_dict.items()}
        # state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
        # msg = self.visual_encoder.load_state_dict(state_dict, strict=False)
        # msg2 = self.visual_encoder_m.load_state_dict(state_dict, strict=False)
        # print(msg)
        # print(msg2)

        self.vision_proj_m = nn.Linear(vision_width, embed_dim)
        self.text_encoder_m = BertModel(config=bert_config, add_pooling_layer=False)
        self.text_proj_m = nn.Linear(text_width, embed_dim)

        # self.vision_decoder_m = BertModel(config=config_decoder)

        # classification specific
        self.loss_fct = nn.BCEWithLogitsLoss()
        self.cls_0 = nn.Linear(text_width, 1)
        self.classifiers = [self.cls_0]

        self.model_pairs = [[self.visual_encoder, self.visual_encoder_m],
                            [self.vision_proj, self.vision_proj_m],
                            [self.text_encoder, self.text_encoder_m],
                            # [self.vision_decoder, self.vision_decoder_m],
                            [self.text_proj, self.text_proj_m]]

        self.copy_params()

        # create the queue
        self.register_buffer("image_queue", torch.randn(embed_dim, self.queue_size))
        self.register_buffer("text_queue", torch.randn(embed_dim, self.queue_size))
        self.register_buffer("idx_queue", torch.full((1, self.queue_size), -100))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        self.image_queue = nn.functional.normalize(self.image_queue, dim=0)
        self.text_queue = nn.functional.normalize(self.text_queue, dim=0)

        self.config = config

        # # freeze vision weights
        # for param in self.visual_encoder.parameters():
        #     param.requires_grad = False
        # for param in self.text_encoder.parameters():
        #     param.requires_grad = False
        #
        # # freeze language weights
        # for name, param in self.text_encoder.named_parameters():
        #     if 'layer' in name:
        #         encoder_keys = name.split('.')
        #         layer_num = int(encoder_keys[2])
        #         if layer_num < 6:
        #             param.requires_grad = False

    def forward(self, image, label, mode='train'):
        bs = image.size(0)

        # with torch.no_grad():
        #     image_embeds_m_raw = self.visual_encoder_m.get_intermediate_layers(image, 1)[0]
        image_embeds_raw = self.visual_encoder.get_intermediate_layers(image, 1)[0]

        image_embeds = image_embeds_raw
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)

        # text_output = self.text_encoder(text.input_ids, attention_mask=text.attention_mask,
        #                                 return_dict=True, mode='text')
        # text_embeds = text_output.last_hidden_state

        ###=================================###
        # if self.config['fusion']:
        #     # fusion image encoder
        #     vl_embeddings = self.vision_decoder(encoder_embeds=image_embeds,
        #                                         attention_mask=image_atts,
        #                                         encoder_hidden_states=text_embeds,
        #                                         encoder_attention_mask=text.attention_mask,
        #                                         return_dict=True,
        #                                         mode='fusion').last_hidden_state[:, 0, :]
        # else:
        # vl_embeddings = self.text_encoder(encoder_embeds=image_embeds,
        #                                         attention_mask=image_atts,
        #                                         return_dict=True,
        #                                         mode='fusion').last_hidden_state[:, 0, :]
        vl_embeddings = image_embeds[:,0,:]

        losses = 0.
        if mode == 'train':
            for i in range(self.config['n_classes']):
                output = self.classifiers[i](vl_embeddings)
                loss = self.loss_fct(output.view(-1), label[:, i].view(-1))
                losses += loss
            return losses
        else:
            outputs = []
            for i in range(self.config['n_classes']):
                outputs.append(self.classifiers[i](vl_embeddings))
            return outputs

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
    def _dequeue_and_enqueue(self, image_feat, text_feat, idx):
        # gather keys before updating queue
        image_feats = concat_all_gather(image_feat)
        text_feats = concat_all_gather(text_feat)
        idxs = concat_all_gather(idx)

        batch_size = image_feats.shape[0]

        ptr = int(self.queue_ptr)
        assert self.queue_size % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.image_queue[:, ptr:ptr + batch_size] = image_feats.T
        self.text_queue[:, ptr:ptr + batch_size] = text_feats.T
        self.idx_queue[:, ptr:ptr + batch_size] = idxs.T
        ptr = (ptr + batch_size) % self.queue_size  # move pointer

        self.queue_ptr[0] = ptr

        # jinyu: patch pooling of image patches to reduce computation and enlarge receptive field

    def patch_pooling(self, x):
        pooled_patch_length = 16
        batch_size, seq_length, dim = x.size()
        b1 = int(np.sqrt(seq_length))
        x = x.reshape(batch_size, b1, b1, dim)
        x = x.permute(0, 3, 1, 2)
        c1 = b1 // int(np.sqrt(pooled_patch_length))
        x = F.avg_pool2d(x, c1, stride=c1)
        x = x.permute(0, 2, 3, 1).reshape(batch_size, pooled_patch_length, dim)
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

