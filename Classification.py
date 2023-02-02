import argparse
import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"
import ruamel.yaml as yaml
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.utils.data import DataLoader

from models.model_classification import XVLModel
from models.vit import interpolate_pos_embed
from models.tokenization_bert import BertTokenizer

import utils
from dataset import create_dataset, create_sampler, create_loader
from scheduler import create_scheduler
from optim import create_optimizer
from transformers import AutoTokenizer
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

import pandas as pd

def train(model, data_loader, optimizer, tokenizer, epoch, warmup_steps, device, scheduler, config):
    # train
    model.train()  
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 50
    step_size = 100
    warmup_iterations = warmup_steps*step_size
    
    for i,(image, label, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        image = image.to(device)
        label = label.to(device)

        loss = model(image, label, mode='train')
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()    
        
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        if epoch==0 and i%step_size==0 and i<=warmup_iterations: 
            scheduler.step(i//step_size)         
        
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())     
    return {k: "{:.3f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}  



@torch.no_grad()
def evaluation(model, data_loader, tokenizer, device, config, n):
    # test
    model.eval()
    
    print('Computing features for evaluation...')
    start_time = time.time()

    y_score_0 = []
    y_true_0 = []
    y_score_1 = []
    y_true_1 = []
    y_score_2 = []
    y_true_2 = []
    y_score_3 = []
    y_true_3 = []
    y_score_4 = []
    y_true_4 = []

    y_scores = [y_score_0, y_score_1, y_score_2, y_score_3, y_score_4]
    y_trues = [y_true_0, y_true_1, y_true_2, y_true_3, y_true_4]

    result_list = []

    for image, label, path in data_loader:
        image = image.to(device)
        label = label.to(device)

        outputs = model(image, label, mode='eval')

        for cls in range(config['n_classes']):
            y_trues[cls].extend(label[:, cls].cpu().numpy())
            y_scores[cls].extend(F.sigmoid(outputs[cls] + 0.5).cpu().numpy())

        y_label = label[:, cls].cpu().numpy()
        y_pred = np.round(F.sigmoid(outputs[cls] + 0.5).cpu().numpy())

        result_list.append([path, y_label, y_pred])
    rf = pd.DataFrame(result_list, columns=['img_path', 'label', 'pred'])
    rf.to_csv("CLS_DINO.csv", mode='w')

    aucs = []
    for c in range(config['n_classes']):
        auc = roc_auc_score(y_trues[c], y_scores[c])
        aucs.append(auc)
    print(aucs)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Evaluation time {}'.format(total_time_str))

    report = classification_report(y_true=y_trues[0], y_pred=np.round(y_scores[0]), digits=4)
    print(report)

    cf_matrix = confusion_matrix(y_true=y_trues[0], y_pred=np.round(y_scores[0]))
    print(cf_matrix)

    return aucs


def main(args, config):
    utils.init_distributed_mode(args)    
    
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    #### Dataset #### 
    print("Creating retrieval dataset")
    train_dataset, val_dataset, test_dataset = create_dataset('cls', config)

    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()            
        samplers = create_sampler([train_dataset], [True], num_tasks, global_rank) + [None, None]
    else:
        samplers = [None, None, None]

    fp16_scaler = None
    if config['use_fp16']:
        fp16_scaler = torch.cuda.amp.GradScaler()

    n = config['n_last_layer']
    
    train_loader, val_loader, test_loader = create_loader([train_dataset, val_dataset, test_dataset],samplers,
                                                          batch_size=[config['batch_size_train']]+[config['batch_size_test']]*2,
                                                          num_workers=[4,4,4],
                                                          is_trains=[True, False, False], 
                                                          collate_fns=[None,None,None])   
       
    tokenizer = AutoTokenizer.from_pretrained("./my_tokenizer/")

    #### Model #### 
    print("Creating model")
    model = XVLModel(train_loader, config=config, text_encoder=args.text_encoder, tokenizer=tokenizer)
    
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        state_dict = checkpoint['student']
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        state_dict = {k.replace("backbone.", "visual_encoder."): v for k, v in state_dict.items()}
        msg = model.load_state_dict(state_dict, strict=False)
        print(msg)

    model = model.to(device)

    model.copy_params()
    print('model parameters are copied.')
    
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module   
    
    arg_opt = utils.AttrDict(config['optimizer'])
    optimizer = create_optimizer(arg_opt, model)
    arg_sche = utils.AttrDict(config['schedular'])
    lr_scheduler, _ = create_scheduler(arg_sche, optimizer)  
    
    max_epoch = config['schedular']['epochs']
    warmup_steps = config['schedular']['warmup_epochs']
    best = 0
    best_epoch = 0

    print("Start training")
    start_time = time.time()    
    for epoch in range(0, max_epoch):
        if not args.evaluate:
            if args.distributed:
                train_loader.sampler.set_epoch(epoch)
            train_stats = train(model, train_loader, optimizer, tokenizer, epoch, warmup_steps, device, lr_scheduler, config)

        val_aucs = evaluation(model, val_loader, tokenizer, device, config, n)
        mean_aucs = np.array(val_aucs).mean()

        save_obj = {
            'model': model_without_ddp.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'config': config,
            'epoch': epoch,
        }
        torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint_best.pth'))
        best = mean_aucs
        best_epoch = epoch
                    
        if args.evaluate: 
            break
           
        lr_scheduler.step(epoch+warmup_steps+1)  
        dist.barrier()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str)) 

    if utils.is_main_process():
        with open(os.path.join(args.output_dir, "log.txt"),"a") as f:
            f.write("best epoch: %d"%best_epoch)               

            
if __name__ == '__main__':
    parser = argparse.ArgumentParser()     
    parser.add_argument('--config', default='./configs/Classification.yaml')
    parser.add_argument('--output_dir', default='output/Classification')
    parser.add_argument('--checkpoint', default='')   
    parser.add_argument('--text_encoder', default='bert-base-uncased')
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')    
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=True, type=bool)
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        
    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))    
    
    main(args, config)
