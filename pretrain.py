import os
import numpy as np
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import pickle
import json
import time
import copy
from tqdm import tqdm
from transformers import BartTokenizer, BartForConditionalGeneration
# from data.data import ZuCo_dataset
from data.data_for_rawEEG import ZuCo_dataset
from dataclasses import asdict
import sys
# import wandb

from utils import *

from eegpt import EEGPT
# from module.eegpt.cnn_eegpt import EEGPT
from utils.configs import *

# import os, tempfile
# TMPDIR = "/home/pymp_tmp"
# os.environ["TMPDIR"] = TMPDIR
# os.environ["TEMP"] = TMPDIR
# os.environ["TMP"] = TMPDIR
# tempfile.tempdir = TMPDIR

def train(dataloaders, device, model, optimizer, scheduler, num_epochs=25, checkpoint_path_best = './checkpoints/decoding/best/temp_decoding.pt', checkpoint_path_last = './checkpoints/decoding/last/temp_decoding.pt'):
    # modified from: https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 100000000000
    early_stopper = EarlyStopping(patience=30) 

    for epoch in range(1, num_epochs+1):
        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('-' * 10)
        # Each epoch has a training and validation phase
        for phase in ['train', 'dev']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()

            running_loss = 0.0

            
            # Iterate over data.
            # for input_embeddings, seq_len, input_masks, input_mask_invert, target_ids, target_mask, sent_level_EEG, rawEEG in tqdm(dataloaders[phase]):
            for sent_level_EEG, rawEEG, input_masks, input_mask_invert, target_ids, target_mask in tqdm(dataloaders[phase], file=sys.stderr):
                
                # load in batch
                input_embeddings_batch = rawEEG.to(device).float()
                # input_embeddings_batch = sent_level_EEG.unsqueeze(1).to(device).float()
                input_masks_batch = input_masks.to(device)
                input_mask_invert_batch = input_mask_invert.to(device)
                target_ids_batch = target_ids.to(device)
                """replace padding ids in target_ids with -100"""
                context = target_ids_batch.clone()
                target_ids_batch[target_ids_batch == tokenizer.pad_token_id] = -100

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
    	        # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # warm_up_rate = get_linear_annealing_weight(epoch, 10, 30)  # Update the annealing weight
                    loss = model(input_embeddings_batch, context)
                    """calculate loss"""

                    if phase == 'train':
                        # with torch.autograd.detect_anomaly():
                        loss.sum().backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                        optimizer.step()
                # statistics
                
                running_loss += loss.sum().item() * input_embeddings_batch.size()[0] # batch loss
                
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]

            print('{} Loss: {:.4f}'.format(phase, epoch_loss))
            
            # deep copy the model
            if phase == 'dev':
                if  epoch_loss < best_loss:
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
                    '''save checkpoint'''
                    # torch.save(model.state_dict(), checkpoint_path_best)

                    target = {k[7:]: v.detach().cpu()  for k, v in model.state_dict().items() if k.startswith("target.")}
                    torch.save(target, checkpoint_path_best)

                    print(f'update best on dev checkpoint: {checkpoint_path_best}')
                # ④ Early Stopping 조건 확인
                if early_stopper(epoch_loss):
                    print(f'\nEarly stopping triggered (patience={20}) at epoch {epoch}')
                    torch.save(model.state_dict(), checkpoint_path_last)
                    # target = {k[7:]: v.detach().cpu()  for k, v in model.state_dict().items() if k.startswith("target.")}
                    # torch.save(target, checkpoint_path_last)
                    model.load_state_dict(best_model_wts)   # 최적 가중치 복원
                    torch.save(model.state_dict(), checkpoint_path_best)
                    # target = {k[7:]: v.detach().cpu()  for k, v in model.state_dict().items() if k.startswith("target.")}
                    # torch.save(target, checkpoint_path_best)

                    time_elapsed = time.time() - since
                    print(f'Training stopped in {time_elapsed//60:.0f}m {time_elapsed%60:.0f}s')
                    print(f'update last checkpoint: {checkpoint_path_last}')
                    return model
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val loss: {:4f}'.format(best_loss))

    # load best model weights
    torch.save(model.state_dict(), checkpoint_path_last)
    # target = {k[7:]: v.detach().cpu()  for k, v in model.state_dict().items() if k.startswith("target.")}
    # torch.save(target, checkpoint_path_last)
    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), checkpoint_path_best)
    # target = {k[7:]: v.detach().cpu()  for k, v in model.state_dict().items() if k.startswith("target.")}
    # torch.save(target, checkpoint_path_best)
    print(f'update last checkpoint: {checkpoint_path_last}')
    return model


if __name__ == '__main__':
    args = get_config('pretrain_autoencoder')

    ''' config param'''

    is_con = args['is_con']
    is_geo = args['is_geo']
    is_kl = args['is_kl']

    
    num_epoch = args['num_epoch']
    lr = args['learning_rate']
    
    batch_size = args['batch_size']
    # task_name = 'task1'
    # task_name = 'task1_task2'
    # task_name = 'task1_task2_task3'
    # task_name = 'task1_task2_taskNRv2'
    task_name = args['task_name']
    save_path = args['save_path']
    # train_input = args['train_input'] # word-level or rawEEG
    # print("train_input is:", train_input)   
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    experiment_name = f"pretrained" if args['save_name'] is None else args['save_name']

    save_name = f"{experiment_name}-module"
 
    if args['is_con']:
        save_name = 'con_' + save_name
    if args['is_geo']:
        save_name = 'geo_' + save_name
    if args['is_kl']:
        save_name = 'kl_' + save_name

    save_path_best = os.path.join(save_path, 'best')
    if not os.path.exists(save_path_best):
        os.makedirs(save_path_best)

    output_checkpoint_name_best = os.path.join(save_path_best, f'{save_name}.pt')

    save_path_last = os.path.join(save_path, 'last')
    if not os.path.exists(save_path_last):
        os.makedirs(save_path_last)

    output_checkpoint_name_last = os.path.join(save_path_last, f'{save_name}.pt')

    # subject_choice = 'ALL
    # subject_choice = args['subjects']
    # print(f'![Debug]using {subject_choice}')
    # eeg_type_choice = 'GD
    eeg_type_choice = args['eeg_type']
    print(f'[INFO]eeg type {eeg_type_choice}')
    # bands_choice = ['_t1'] 
    # bands_choice = ['_t1','_t2','_a1','_a2','_b1','_b2','_g1','_g2'] 
    bands_choice = args['eeg_bands']
    print(f'[INFO]using bands {bands_choice}')


    
    ''' set random seeds '''
    # https://pytorch.org/docs/stable/notes/randomness.html
    seed_val = args['seed']
    set_seed(seed_val)

    ''' set up device '''
    # use cuda
    if torch.cuda.is_available():  
        # dev = "cuda:3" 
        dev = args['cuda'] 
    else:  
        dev = "cpu"
    # CUDA_VISIBLE_DEVICES=0,1,2,3  
    device = torch.device(dev)
    print(f'[INFO]using device {dev}')
    print()

    ''' set up dataloader '''
    whole_dataset_dicts = []
    if 'task1' in task_name:
        dataset_path_task1 = './dataset/downsampledZuCo/task1-SR/task1-SR-dataset.pickle'
        with open(dataset_path_task1, 'rb') as handle:
            whole_dataset_dicts.append(pickle.load(handle))
    if 'task2' in task_name:
        dataset_path_task2 = './dataset/downsampledZuCo/task2-NR/task2-NR-dataset.pickle'
        with open(dataset_path_task2, 'rb') as handle:
            whole_dataset_dicts.append(pickle.load(handle))
    if 'task3' in task_name:
        dataset_path_task3 = './dataset/downsampledZuCo/task3-TSR/task3-TSR-dataset.pickle'
        with open(dataset_path_task3, 'rb') as handle:
            whole_dataset_dicts.append(pickle.load(handle))
    if 'taskNRv2' in task_name:
        dataset_path_taskNRv2 = './dataset/downsampledZuCo/task2-NR-2.0/task2-NR-2.0-dataset.pickle'
        with open(dataset_path_taskNRv2, 'rb') as handle:
            whole_dataset_dicts.append(pickle.load(handle))

    print()


    
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')


    dataset_setting = args['dataset_setting']

    # train_set = ZuCo_dataset(whole_dataset_dicts, 'train', tokenizer, subject = subject_choice, eeg_type = eeg_type_choice, bands = bands_choice, setting = dataset_setting, test_input=train_input)
    # # dev dataset
    # dev_set = ZuCo_dataset(whole_dataset_dicts, 'dev', tokenizer, subject = subject_choice, eeg_type = eeg_type_choice, bands = bands_choice, setting = dataset_setting, test_input=train_input)
    # # test dataset
    # test_set = ZuCo_dataset(whole_dataset_dicts, 'test', tokenizer, subject = subject_choice, eeg_type = eeg_type_choice, bands = bands_choice, setting = dataset_setting, test_input=train_input)
    subject_choice = 'ALL'
    train_set = ZuCo_dataset(whole_dataset_dicts, 'train', tokenizer, subject = subject_choice, bands = bands_choice, setting = dataset_setting)
    # dev dataset
    dev_set = ZuCo_dataset(whole_dataset_dicts, 'dev', tokenizer, subject = subject_choice, bands = bands_choice, setting = dataset_setting)
    # test dataset
    test_set = ZuCo_dataset(whole_dataset_dicts, 'test', tokenizer, subject = subject_choice, bands = bands_choice, setting = dataset_setting)

    dataset_sizes = {'train': len(train_set), 'dev': len(dev_set)}
    print('[INFO]train_set size: ', len(train_set))
    print('[INFO]dev_set size: ', len(dev_set))
    print('[INFO]test_set size: ', len(test_set))
    
    # train dataloader
    train_dataloader = DataLoader(train_set, batch_size = batch_size, shuffle=True, num_workers=4)
    # dev dataloader
    val_dataloader = DataLoader(dev_set, batch_size = 1, shuffle=False, num_workers=4)
    test_dataloader = DataLoader(test_set, batch_size = 1, shuffle=False, num_workers=4)
    # dataloaders
    dataloaders = {'train':train_dataloader, 'dev':val_dataloader, 'test':test_dataloader}

    # # cpu ram 덜 잡아먹는지 확인
    # del whole_dataset_dicts
    # import gc; gc.collect()

    pretrained_LM = BartForConditionalGeneration.from_pretrained('facebook/bart-large').to(device)

    vocab_size = tokenizer.vocab_size
    
    model = EEGPT(get_configs(**(MODELS_CONFIGS[tag])), 
                    USE_LOSS_A =(variant != "A"),
                    USE_LN     =(variant != "B"),
                    USE_SKIP   =(variant != "C"),
                    pretrained_LM = pretrained_LM).to(device)
    
    
    param_groups = [
        {
            'params': (p for n, p in model.encoder.named_parameters()
                    if ('bias' not in n) and (len(p.shape) != 1))
        }, {
            'params': (p for n, p in model.predictor.named_parameters()
                    if ('bias' not in n) and (len(p.shape) != 1))
        }, {
            'params': (p for n, p in model.reconstructor.named_parameters()
                    if ('bias' not in n) and (len(p.shape) != 1))
        }, {
            'params': (p for n, p in model.encoder.named_parameters()
                    if ('bias' in n) or (len(p.shape) == 1)),
            'WD_exclude': True,
            'weight_decay': 0
        }, {
            'params': (p for n, p in model.predictor.named_parameters()
                    if ('bias' in n) or (len(p.shape) == 1)),
            'WD_exclude': True,
            'weight_decay': 0
        }, {
            'params': (p for n, p in model.reconstructor.named_parameters()
                    if ('bias' in n) or (len(p.shape) == 1)),
            'WD_exclude': True,
            'weight_decay': 0
        }
    ]
    for name, param in model.named_parameters():
        if param.requires_grad and 'pretrained' in name:
            # 이것도 멈추는거 한번 보자
            # pretrian에서 BART embedding시키니까 한번 그쪽도 손봐야 되기도 싶고
            param.requires_grad = False

    # optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-5)
    optimizer = torch.optim.AdamW(param_groups, lr=6e-5) 

    """save config"""
    cfg_dir = './save_model/config'

    if not os.path.exists(cfg_dir):
        os.makedirs(cfg_dir)
    
    state_dict = {
        "args":args,
        "module_state_dict": {"model": get_configs(**(MODELS_CONFIGS[tag]))},
    }
    with open(os.path.join(cfg_dir,f'{save_name}.json'), 'w') as out_config:
        json.dump(state_dict, out_config, indent = 4)

    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    ''' training loop '''
    since = time.time()
    model = train(dataloaders, device, model, optimizer, exp_lr_scheduler, num_epochs=num_epoch, checkpoint_path_best = output_checkpoint_name_best, checkpoint_path_last = output_checkpoint_name_last)
    total_time = time.time() - since
    print(f'Training complete in {total_time//60:.0f}m {total_time%60:.0f}s')
