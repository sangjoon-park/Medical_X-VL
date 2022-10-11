import json
import numpy as np
from collections import Counter
from nltk import ngrams
from dataset.utils import pre_caption
import nltk.translate.bleu_score as bleu
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.bleu_score import SmoothingFunction
smooth = SmoothingFunction()

gen_file = './generated_20220713_ALBEF_seed42_mimic.json'

gen_dic = [json.loads(l) for l in open(gen_file)][0]

num_ids = len(gen_dic)

bleu4 = 0.

for i, id in enumerate(gen_dic.keys()):
    references = gen_dic[str(id)]['caption'][0]
    hypotheses = gen_dic[str(id)]['predicted']
    hypotheses = hypotheses.replace('[CLS] ', '').replace(' [SEP]', '')
    references = references.split()
    hypotheses = hypotheses.split()

    # references = references.split('.')
    # hypotheses = hypotheses.split('.')

    # corp_ref = []
    # for ref in references:
    #     if len(ref.split()) != 0:
    #         corp_ref.append(ref.split())
    #
    # corp_hypo = []
    # for hypo in hypotheses:
    #     if len(hypo.split()) != 0:
    #         corp_hypo.append(hypo.split())

    score = bleu.sentence_bleu([references], hypotheses, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth.method1)
    # score = corpus_bleu([corp_ref], corp_hypo, smoothing_function=smooth.method1)
    bleu4 += score

bleu4 = bleu4 / (i + 1)
print(bleu4)