model_path = '/COVID_8TB/sangjoon/vision_language/checkpoint/FINAL_seed42/checkpoint_14.pth'
bert_config_path = 'configs/config_bert.json'
use_cuda = True

from functools import partial
from models.vit import VisionTransformer
from models.xbert import BertConfig, BertModel
from models.tokenization_bert import BertTokenizer
from models.ibot_vit import VisionTransformer, interpolate_pos_embed, vit_small

import torch
from torch import nn
from torchvision import transforms
import ruamel.yaml as yaml
from transformers import AutoTokenizer
from ibot_utils import GaussianBlur
from dataset import create_dataset, create_loader

import json

config = yaml.load(open('./configs/Pretrain.yaml', 'r'), Loader=yaml.Loader)

class VL_Transformer_ITM(nn.Module):
    def __init__(self,
                 text_encoder=None,
                 config_bert=''
                 ):
        super().__init__()

        bert_config = BertConfig.from_json_file(config_bert)

        self.visual_encoder = vit_small(
            img_size=(config['image_res'], config['image_res']),
            patch_size=config['patch_size'],
        return_all_tokens=True,
        )

        self.text_encoder = BertModel(config=bert_config, add_pooling_layer=False)
        self.itm_head_t = nn.Linear(384, 2)

    def forward(self, image, text):
        image_embeds = self.vision_encoder(image)
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)

        output = self.text_encoder(text.input_ids,
                                   attention_mask=text.attention_mask,
                                   encoder_hidden_states=image_embeds,
                                   encoder_attention_mask=image_atts,
                                   return_dict=True,
                                   )

        vl_embeddings = output.last_hidden_state[:, 0, :]
        vl_output = self.itm_head_t(vl_embeddings)
        return vl_output
# %% md

import re


def pre_caption(caption, max_words=120):
    caption = re.sub(
        r"([,'!?\"()*#:;~])",
        '',
        caption.lower(),
    ).replace('/', ' ').replace('<person>', 'person')
    caption = caption.replace('[sep]', '[SEP]')

    caption = re.sub(
        r"\s{2,}",
        ' ',
        caption,
    )
    caption = caption.rstrip('\n')
    caption = caption.strip(' ')

    # truncate caption
    caption_words = caption.split(' ')
    if len(caption_words) > max_words:
        caption = ' '.join(caption_words[:max_words])

    return caption

from PIL import Image

import cv2
import numpy as np

from skimage import transform as skimage_transform
from scipy.ndimage import filters
from matplotlib import pyplot as plt
import gc


def getAttMap(img, attMap, blur = True, overlap = True):
    attMap -= attMap.min()
    if attMap.max() > 0:
        attMap /= attMap.max()
    attMap = skimage_transform.resize(attMap, (img.shape[:2]), order = 3, mode = 'constant')
    if blur:
        attMap = filters.gaussian_filter(attMap, 0.02*max(img.shape[:2]))
        attMap -= attMap.min()
        attMap /= attMap.max()
    cmap = plt.get_cmap('jet')
    attMapV = cmap(attMap)
    attMapV = np.delete(attMapV, 3, 2)
    if overlap:
        attMap = 1*(1-attMap**0.7).reshape(attMap.shape + (1,))*img + (attMap**0.7).reshape(attMap.shape+(1,)) * attMapV
    return attMap

transform = transforms.Compose([
    transforms.Resize((224,224),interpolation=Image.BICUBIC),
    GaussianBlur(1.0, radius_min=0.5, radius_max=0.5),
    transforms.ToTensor(),
    # normalize,
])

tokenizer = AutoTokenizer.from_pretrained("./my_tokenizer/")

model = VL_Transformer_ITM(text_encoder='bert-base-uncased', config_bert=bert_config_path)

checkpoint = torch.load(model_path, map_location='cpu')['model']
checkpoint = {k.replace("bert.", ""): v for k, v in checkpoint.items()}
checkpoint = {k.replace("backbone.", ""): v for k, v in checkpoint.items()}
msg = model.load_state_dict(checkpoint,strict=False)
print(msg)
model.eval()

block_num = 8

loss_fn = nn.BCEWithLogitsLoss()

model.text_encoder.base_model.base_model.encoder.layer[block_num].crossattention.self.save_attention = True

if use_cuda:
    model.cuda()
    loss_fn = loss_fn.cuda()

print("Creating retrieval dataset")
train_dataset, val_dataset, test_dataset = create_dataset('re', config)
samplers = [None, None, None]

_, _, test_loader = create_loader([train_dataset, val_dataset, test_dataset], samplers,
                                                      batch_size=[config['batch_size']] + [
                                                          80] * 2,
                                                      num_workers=[4, 4, 4],
                                                      is_trains=[True, False, False],
                                                      collate_fns=[None, None, None])

# Set index 0 - 19
index = 16

for i, (images, img_ids, labels, captions) in enumerate(test_loader):
    if i == index:
        for image, img_id, label, caption in zip(images, img_ids, labels, captions):
            image_path = label
            image_pil = Image.open(image_path).convert('RGB')
            image = transform(image_pil).unsqueeze(0)

            text = pre_caption(caption)
            text_input = tokenizer(text, return_tensors="pt")

            if use_cuda:
                image = image.cuda()
                text_input = text_input.to(image.device)

            image_embeds = model.visual_encoder(image)
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)
            hidden = model.text_encoder(text_input.input_ids,
                                            attention_mask=text_input.attention_mask,
                                            encoder_hidden_states=image_embeds,
                                            encoder_attention_mask=image_atts,
                                            output_hidden_states=True,
                                            return_dict=True,
                                             )
            last_hidden = hidden.hidden_states[-1]
            vl_embeddings = last_hidden[:, 0, :]
            output = model.itm_head_t(vl_embeddings)
            # loss = output[:,1].sum()
            loss = loss_fn(output[:,1], torch.cuda.FloatTensor([1]))

            hidden.hidden_states[-5].retain_grad()

            model.zero_grad()
            loss.backward()

            # calculate gradients
            gradients = hidden.hidden_states[-5].grad
            abs_gradients = torch.norm(gradients, dim=-1)

            with torch.no_grad():
                mask = text_input.attention_mask.view(text_input.attention_mask.size(0),1,-1,1,1)

                grads=model.text_encoder.base_model.base_model.encoder.layer[block_num].crossattention.self.get_attn_gradients()
                cams=model.text_encoder.base_model.base_model.encoder.layer[block_num].crossattention.self.get_attention_map()

                cams = cams[:, :, :, 1:].reshape(image.size(0), 12, -1, 14, 14) * mask
                grads = grads[:, :, :, 1:].clamp(0).reshape(image.size(0), 12, -1, 14, 14) * mask

                gradcam = cams * grads
                gradcam = gradcam[0].mean(0).cpu().detach()
            #%% md

            num_image = len(text_input.input_ids[0])
            fig, ax = plt.subplots(num_image, 1, figsize=(15, 5 * num_image))

            rgb_image = cv2.imread(image_path)[:, :, ::-1]
            rgb_image = np.float32(rgb_image) / 255

            ax[0].imshow(rgb_image)
            ax[0].set_yticks([])
            ax[0].set_xticks([])
            ax[0].set_xlabel("Image")

            for i, token_id in enumerate(text_input.input_ids[0][1:]):
                word = tokenizer.decode([token_id])
                gradcam_image = getAttMap(rgb_image, gradcam[i + 1])
                ax[i + 1].imshow(gradcam_image)
                ax[i + 1].set_yticks([])
                ax[i + 1].set_xticks([])
                ax[i + 1].set_xlabel(word)

            # plt.show()
            try:
                fig.savefig('./cross_attention/' + image_path.split('/')[-1])
            except:
                pass

    else:
        pass