from functools import partial
from models.ibot_vit import VisionTransformer, interpolate_pos_embed, vit_base, vit_small
from models.xbert import BertConfig, BertModel

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from fuzzywuzzy import fuzz
from models.vit import VisionTransformer as VisionTransformer_deit

class XVLModel(nn.Module):
    def __init__(self,
                 tokenizer = None,
                 config = None,
                 ):
        super().__init__()

        self.tokenizer = tokenizer
        self.distill = config['distill']
        embed_dim = config['embed_dim']
        vision_width = config['vision_width']

        visual_encoder = vit_base(
            img_size=(config['image_res'], config['image_res']),
            patch_size=config['patch_size'],
            drop_path_rate=config['drop_path'],
            return_all_tokens=True,
        )
        self.visual_encoder = visual_encoder

        bert_config = BertConfig.from_json_file(config['bert_config'])
        fusion_config = BertConfig.from_json_file(config['fusion_config'])
        self.text_encoder = BertModel(config=bert_config, add_pooling_layer=False)
        self.fusion_encoder = BertModel(config=fusion_config, add_pooling_layer=False)

        text_width = self.text_encoder.config.hidden_size
        self.vision_proj = nn.Linear(vision_width, embed_dim)
        self.text_proj = nn.Linear(text_width, embed_dim)

        self.temp = nn.Parameter(torch.ones([]) * config['temp'])
        self.queue_size = config['queue_size']
        self.momentum = config['momentum']
        self.itm_head_t = nn.Linear(text_width, 2)
        self.itm_head_v = nn.Linear(text_width, 2)

        # create momentum models
        visual_encoder_m = vit_base(
            img_size=(config['image_res'], config['image_res']),
            patch_size=config['patch_size'],
            return_all_tokens=True,
        )
        self.visual_encoder_m = visual_encoder_m

        self.vision_proj_m = nn.Linear(vision_width, embed_dim)
        self.text_encoder_m = BertModel(config=bert_config, add_pooling_layer=False)
        self.fusion_encoder_m = BertModel(config=fusion_config, add_pooling_layer=False)
        self.text_proj_m = nn.Linear(text_width, embed_dim)

        self.model_pairs = [[self.visual_encoder,self.visual_encoder_m],
                            [self.vision_proj,self.vision_proj_m],
                            [self.fusion_encoder,self.fusion_encoder_m],
                            [self.text_proj,self.text_proj_m],
                           ]
        self.copy_params()

        # create the queue
        self.register_buffer("image_queue", torch.randn(embed_dim, self.queue_size))
        self.register_buffer("fnd_queue", torch.randn(embed_dim, self.queue_size))
        self.register_buffer("imp_queue", torch.randn(embed_dim, self.queue_size))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        self.image_queue = nn.functional.normalize(self.image_queue, dim=0)
        self.fnd_queue = nn.functional.normalize(self.fnd_queue, dim=0)
        self.imp_queue = nn.functional.normalize(self.imp_queue, dim=0)


    def forward(self, image, text, label, alpha, fp16_scaler, n, idx):
        bs = image.size(0)

        with torch.cuda.amp.autocast(fp16_scaler is not None):
            with torch.no_grad():
                image_embeds_m_raw = self.visual_encoder_m.get_intermediate_layers(image, n)[0]
            image_embeds_raw = self.visual_encoder.get_intermediate_layers(image, n)[0]

        image_embeds = image_embeds_raw
        image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(image.device)

        image_feat = F.normalize(self.vision_proj(image_embeds[:,0,:]),dim=-1)
        text_output = self.text_encoder(text.input_ids, attention_mask = text.attention_mask,
                                        return_dict = True, mode = 'text')
        text_embeds = text_output.last_hidden_state
        text_feat = F.normalize(self.text_proj(text_embeds[:,0,:]),dim=-1)

        idx = idx.view(-1,1)
        idx_all = torch.cat([idx.t(), self.idx_queue.clone().detach()],dim=1)
        pos_idx = torch.eq(idx, idx_all).float()
        sim_targets = pos_idx / pos_idx.sum(1,keepdim=True)

        with torch.no_grad():
            self._momentum_update()

            image_embeds_m = image_embeds_m_raw
            image_feat_m = F.normalize(self.vision_proj_m(image_embeds_m[:,0,:]),dim=-1)
            image_feat_all = torch.cat([image_feat_m.t(),self.image_queue.clone().detach()],dim=1)
            text_output_m = self.text_encoder_m(text.input_ids, attention_mask = text.attention_mask,
                                                return_dict = True, mode = 'text')
            text_feat_m = F.normalize(self.text_proj_m(text_output_m.last_hidden_state[:,0,:]),dim=-1)
            text_feat_all = torch.cat([text_feat_m.t(),self.text_queue.clone().detach()],dim=1)

            if self.distill:
                sim_i2t_m = image_feat_m @ text_feat_all / self.temp
                sim_t2i_m = text_feat_m @ image_feat_all / self.temp

                sim_i2t_targets = alpha * F.softmax(sim_i2t_m, dim=1) + (1 - alpha) * sim_targets
                sim_t2i_targets = alpha * F.softmax(sim_t2i_m, dim=1) + (1 - alpha) * sim_targets

        sim_i2t = image_feat @ text_feat_all / self.temp
        sim_t2i = text_feat @ image_feat_all / self.temp

        if self.distill:
            loss_i2t = -torch.sum(F.log_softmax(sim_i2t, dim=1)*sim_i2t_targets,dim=1).mean()
            loss_t2i = -torch.sum(F.log_softmax(sim_t2i, dim=1)*sim_t2i_targets,dim=1).mean()
        else:
            loss_i2t = -torch.sum(F.log_softmax(sim_i2t, dim=1)*sim_targets,dim=1).mean()
            loss_t2i = -torch.sum(F.log_softmax(sim_t2i, dim=1)*sim_targets,dim=1).mean()

        loss_ita = (loss_i2t + loss_t2i)/2

        self._dequeue_and_enqueue(image_feat_m, text_feat_m, idx)

        ###=================================###
        # forward the positve image-text pair
        output_pos = self.text_encoder(encoder_embeds = text_embeds,
                                        attention_mask = text.attention_mask,
                                        encoder_hidden_states = image_embeds,
                                        encoder_attention_mask = image_atts,
                                        return_dict = True,
                                        mode = 'fusion',
                                       )
        with torch.no_grad():
            dataframe = self.train_loader.dataset.train_df

            weights_i2t = F.softmax(sim_i2t[:,:bs],dim=1)
            weights_t2i = F.softmax(sim_t2i[:,:bs],dim=1)

            mask = torch.eq(idx, idx.T)
            weights_i2t.masked_fill_(mask, 0)
            weights_t2i.masked_fill_(mask, 0)

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

                this = ''.join(dataframe.loc[dataframe.image == label[b]].label.iloc[0])
                cand = ''.join(dataframe.loc[dataframe.image == label[neg_idx]].label.iloc[0])

                if fuzz.token_sort_ratio(this, cand) != 100:
                    break
                if n > 100:
                    break
            image_embeds_neg.append(image_embeds[neg_idx])
        image_embeds_neg = torch.stack(image_embeds_neg, dim=0)

        # select a negative text for each image
        text_embeds_neg = []
        text_atts_neg = []
        for b in range(bs):
            while True:
                try:
                    neg_idx = torch.multinomial(weights_i2t[b], 1).item()
                except:
                    neg_idx = torch.randint(0, bs, (1,)).item()
                    while neg_idx == b:
                        neg_idx = torch.randint(0, bs, (1,)).item()
                n += 1

                this = ''.join(dataframe.loc[dataframe.image == label[b]].label.iloc[0])
                cand = ''.join(dataframe.loc[dataframe.image == label[neg_idx]].label.iloc[0])

                if fuzz.token_sort_ratio(this, cand) != 100:
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

        output_neg = self.text_encoder(encoder_embeds=text_embeds_all,
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

        return loss_ita, loss_itm



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
        x = x.permute(0,3,1,2)
        c1 = b1 // int(np.sqrt(pooled_patch_length))
        x = F.avg_pool2d(x, c1, stride=c1)
        x = x.permute(0,2,3,1).reshape(batch_size, pooled_patch_length, dim)
        return x


    # jinyu: in-batch g2l loss
    def in_batch_g2l_loss(self, l, m, temp, attention_mask=None):
        m = m.unsqueeze(1)
        N, n_locals, dim = l.size()
        l_n = l.reshape(-1, dim) # (N * n_locals) * d
        m_n = m.reshape(-1, dim) # N * d

        # Inner product for positive samples. Outer product for negative. We need to do it this way
        # for the multiclass loss. For the outer product, we want a N x N x n_locals x 1 tensor.
        u_p = torch.matmul(l, m.permute(0,2,1)).unsqueeze(2) / temp # N * n_locals * 1 * 1

        # if l comes from text, then attention_mask is not None
        if attention_mask is not None:
            temp_mask = attention_mask.unsqueeze(2).unsqueeze(3)
            u_p = (temp_mask * u_p) + (10000. * (1-temp_mask))

        u_n = torch.mm(m_n, l_n.t()) / temp
        u_n = u_n.reshape(N, 1, N, n_locals).permute(0, 2, 3, 1) # N x N x n_locals x 1

        # We need to mask the diagonal part of the negative tensor.
        mask = torch.eye(N)[:, :, None, None].to(l.device) # N*N*1*1
        n_mask = 1 - mask

        # Masking is done by shifting the diagonal before exp.
        u_n = (n_mask * u_n) - (10000. * (1 - n_mask))  # mask out "self" examples
        # if l comes from test, we mask out the padding tokens
        if attention_mask is not None:
            temp_mask = attention_mask.unsqueeze(0).unsqueeze(3).expand(N, -1, -1, -1)
            u_n = (temp_mask * u_n) - (10000. * (1-temp_mask))

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

