__author__ = 'tylin'
from bleu.bleu import Bleu
from meteor.meteor import Meteor
from rouge.rouge import Rouge
from cider.cider import Cider


class COCOEvalCapDirect:
    def __init__(self, json_file):
        self.evalImgs = []
        self.eval = {}
        self.imgToEval = {}
        # self.coco = coco
        # self.cocoRes = cocoRes
        # self.params = {'image_id': coco.getImgIds()}

        self.json_dic = json_file
        self.keys = self.json_dic.keys()

    def evaluate(self):

        gts = {}
        res = {}

        print('tokenization... [my tokenizer]')

        for key in self.keys:
            gts[key] = [self.json_dic[key]['label']]
            res[key] = [self.json_dic[key]['pred'].replace('[CLS] ', '').replace(' [SEP]','')]
            a = self.json_dic[key]['label']
            b = self.json_dic[key]['pred'].replace('[CLS] ', '').replace(' [SEP]','')
            a = a

        # # =================================================
        # # Set up scorers
        # # =================================================
        # print('tokenization...')
        # tokenizer = PTBTokenizer()
        # gts  = tokenizer.tokenize(gts)
        # res = tokenizer.tokenize(res)

        # =================================================
        # Set up scorers
        # =================================================
        print('setting up scorers...')
        scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Meteor(),"METEOR"),
            (Rouge(), "ROUGE_L"),
            (Cider(), "CIDEr")
        ]

        # =================================================
        # Compute scores
        # =================================================
        eval = {}
        for scorer, method in scorers:
            print('computing %s score...'%(scorer.method()))
            score, scores = scorer.compute_score(gts, res)
            if type(method) == list:
                for sc, scs, m in zip(score, scores, method):
                    self.setEval(sc, m)
                    self.setImgToEvalImgs(scs, gts.keys(), m)
                    print("%s: %0.3f"%(m, sc))
            else:
                self.setEval(score, method)
                self.setImgToEvalImgs(scores, gts.keys(), method)
                print("%s: %0.3f"%(method, score))
        self.setEvalImgs()

    def setEval(self, score, method):
        self.eval[method] = score

    def setImgToEvalImgs(self, scores, imgIds, method):
        for imgId, score in zip(imgIds, scores):
            if not imgId in self.imgToEval:
                self.imgToEval[imgId] = {}
                self.imgToEval[imgId]["image_id"] = imgId
            self.imgToEval[imgId][method] = score

    def setEvalImgs(self):
        self.evalImgs = [eval for imgId, eval in self.imgToEval.items()]