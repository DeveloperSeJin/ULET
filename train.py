
import wandb
import os
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


# from model import *
from utils import *
from evaluation import eval_model
import sys
# from data.data import ZuCo_dataset
from data.data_for_rawEEG import ZuCo_dataset
from functools import partial

from utils import *
from eegpt import EEGPTCausal, EEGDiffusion, Convolution_Block, EEGPT
# from eegpt import Decoder, Convolution_Block
from utils.configs import *
from module.EEGPT_mcae import EEGTransformer


# import os, tempfile
# TMPDIR = "/home/pymp_tmp"
# os.environ["TMPDIR"] = TMPDIR
# os.environ["TEMP"] = TMPDIR
# os.environ["TMP"] = TMPDIR
# tempfile.tempdir = TMPDIR



def train_model(dataloaders, device, model, optimizer, scheduler, num_epochs=25, checkpoint_path_best = './checkpoints/decoding/best/temp_decoding.pt', checkpoint_path_last = './checkpoints/decoding/last/temp_decoding.pt', stage = None):
    # modified from: https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float('inf')

    # contrastive with reg 할 때 15에서 20으로 바꿨음
    early_stopper = EarlyStopping(patience=30) 

    for epoch in range(1, num_epochs+1):
        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('-' * 10)
        # Each epoch has a training and validation phase
        for phase in ['train', 'dev']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0

            
            # Iterate over data.
            # for input_embeddings, seq_len, input_masks, input_mask_invert, target_ids, target_mask, sent_level_EEG, rawEEG in tqdm(dataloaders[phase]):
            for sent_level_EEG, rawEEG, input_masks, input_mask_invert, target_ids, target_mask in tqdm(dataloaders[phase], file=sys.stderr):
            # for  input_embeddings, seq_len, input_masks, input_mask_invert,target_ids, target_mask, sentiment_labels, sent_level_EEG in tqdm(dataloaders[phase], file=sys.stderr):
                
                # load in batch
                # input_embeddings_batch = input_embeddings.to(device).float()
                input_embeddings_batch = rawEEG.to(device).float()
                input_masks_batch = input_masks.to(device)
                target_ids_batch = target_ids.to(device)
                """replace padding ids in target_ids with -100"""
                context = target_ids_batch.clone()

                target_ids_batch[target_ids_batch == tokenizer.pad_token_id] = -100

                with torch.set_grad_enabled(phase == 'train'):
                    optimizer.zero_grad()
                    out, loss = model(batch= (input_embeddings_batch,target_ids_batch),input_masks_batch= input_masks_batch)

                    """calculate loss"""
                 
                    # backward + optimize only if in training phase
                    loss_sum = loss.sum()
                    if phase == 'train':
                        # with torch.autograd.detect_anomaly():
                        loss_sum.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=4.0)
                        optimizer.step()

                # statistics
                running_loss += loss_sum.detach().item() * input_embeddings_batch.size()[0] # batch loss
                

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]

            wandb.log({
                f'{phase}_loss':epoch_loss,
            }, step=epoch)

            print('{} Loss: {:.4f}'.format(phase, epoch_loss))
            
            # deep copy the model
            if phase == 'dev':
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
                    '''save checkpoint'''
                    torch.save(best_model_wts, checkpoint_path_best)
                    print(f'update best on dev checkpoint: {checkpoint_path_best}')
                # ④ Early Stopping 조건 확인
                if early_stopper(epoch_loss):
                    print(f'\nEarly stopping triggered (patience={20}) at epoch {epoch}')
                    torch.save(model.state_dict(), checkpoint_path_last)
                    model.load_state_dict(best_model_wts)   # 최적 가중치 복원
                    torch.save(model.state_dict(), checkpoint_path_best)
                    
                    time_elapsed = time.time() - since
                    print(f'Training stopped in {time_elapsed//60:.0f}m {time_elapsed%60:.0f}s')
                    print(f'update last checkpoint: {checkpoint_path_best}')
                    return model
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val loss: {:4f}'.format(best_loss))

    # load best model weights
    torch.save(model.state_dict(), checkpoint_path_last)
    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), checkpoint_path_best)
    
    print(f'update last checkpoint: {checkpoint_path_last}')
    return model


if __name__ == '__main__':

    args = get_config('train')

    ''' config param'''    
    num_epochs_step1 = args['num_epoch_step1']
    num_epochs_step2 = args['num_epoch_step2']
    step1_lr = args['learning_rate_step1']
    step2_lr = args['learning_rate_step2']
    
    
    batch_size = args['batch_size']
    
    # task_name = 'task1'
    # task_name = 'task1_task2'
    # task_name = 'task1_task2_task3'
    # task_name = 'task1_task2_taskNRv2'
    task_name = args['task_name']
    # train_input = args['train_input'] # word-level or rawEEG
    # print("train_input is:", train_input)
    save_path = args['save_path']
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    base_name = os.path.basename(args['path'])  # pretrain-module.json
    is_con = args['is_con']
    is_geo = args['is_geo']
    is_kl = args['is_kl']

    if is_con:
        base_name = 'con_' + base_name
    if is_geo:
        base_name = 'geo_' + base_name
    if is_kl:
        base_name = 'kl_' + base_name

    experiment_name = f"fintuning_{base_name}" if args['save_name'] is None else args['save_name']

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

    # subject_choice = args['subjects'] # ALL
    # print(f'![Debug]using {subject_choice}')
    
    eeg_type_choice = args['eeg_type'] # GD
    print(f'[INFO]eeg type {eeg_type_choice}')
    
    bands_choice = args['eeg_bands'] # ['_t1','_t2','_a1','_a2','_b1','_b2','_g1','_g2'] 
    print(f'[INFO]using bands {bands_choice}')
    
    seed_val = args['seed']
    set_seed(seed_val)
    ''' set random seeds '''
    # https://pytorch.org/docs/stable/notes/randomness.html


    ''' set up device '''
    # use cuda
    if torch.cuda.is_available():  
        # dev = "cuda:3" 
        dev = args['cuda'] 
    else:  
        dev = "cpu"
    # CUDA_VISIBLE_DEVICES=0,1
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


    # # train dataset
    # train_set = ZuCo_dataset(whole_dataset_dicts, 'train', tokenizer, subject = subject_choice, eeg_type = eeg_type_choice, bands = bands_choice, setting = dataset_setting, test_input=train_input)
    # # dev dataset
    # dev_set = ZuCo_dataset(whole_dataset_dicts, 'dev', tokenizer, subject = subject_choice, eeg_type = eeg_type_choice, bands = bands_choice, setting = dataset_setting, test_input=train_input)
    # # test dataset
    # test_set = ZuCo_dataset(whole_dataset_dicts, 'test', tokenizer, subject = subject_choice, eeg_type = eeg_type_choice, bands = bands_choice, setting = dataset_setting, test_input=train_input)
    dataset_setting = 'unique_sent'
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
    # del whole_dataset_dicts
    # import gc; gc.collect()

    ''' set up model '''
    pretrained_LM = BartForConditionalGeneration.from_pretrained('facebook/bart-large').to(device)
    vocab_size = tokenizer.vocab_size



    # -- load checkpoint
    encoder_path = args['path']

    state_dict = torch.load(encoder_path)

    # save_encoder_state = \
    #     {'img_size': [62, 2560],
    #     # {'img_size': [32, 1024],
    #     # img_size=[19, 2*256],
    #     # img_size=[56, 840],
    #     'patch_size': 32,
    #     'patch_stride': 32,
    #     'embed_num': 4,
    #     'embed_dim': 256,
    #     'depth': 8,
    #     'num_heads': 4,
    #     'mlp_ratio':4.0,
    #     'drop_rate':0.0,
    #     'attn_drop_rate':0.0,
    #     'drop_path_rate':0.0,
    #     'init_std':0.02,
    #     'qkv_bias':True, 
    #     'norm_layer':partial(nn.LayerNorm, eps=1e-6)}
    state = get_configs(**(MODELS_CONFIGS[tag]))

    encoder = EEGTransformer(
        # img_size=[58, 256*4],
        # img_size= [32, 1024],
        img_size= [56, 1024],
        # patch_size=32*2,
        patch_size=32,
        mlp_ratio=4.0,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        init_std=0.02,
        qkv_bias=True, 
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        embed_dim= state['encoder']['embed_dim'],
        embed_num= state['encoder']['embed_num'],
        depth= state['encoder']['depth'],
        num_heads= state['encoder']['num_heads'],)
    
    conv = Convolution_Block(input_size=1, hidden_dim = 56, chan_size = 62, time_size = 128, pooling_size = 64)
    
    encoder_state = {}
    for key,val in state_dict.items():
        if key.startswith("target."):
            encoder_state[key[7:]]=val

    conv_state = {}
    for key,val in state_dict.items():
        if key.startswith("conv."):
            encoder_state[key]=val

    encoder.load_state_dict(encoder_state, strict=False)
    conv.load_state_dict(conv_state, strict=False)

    # encoder = EEGPT(get_configs(**(MODELS_CONFIGS[tag])), 
    #                 USE_LOSS_A =(variant != "A"),
    #                 USE_LN     =(variant != "B"),
    #                 USE_SKIP   =(variant != "C"),
    #                 pretrained_LM = pretrained_LM).to(device)
    # encoder.load_state_dict(state_dict, strict=False)


    # model = EEGDiffusion(encoder = encoder, conformer_module=conv, device=device).to(device)
    
    model = EEGPTCausal(pretrained_encoder=encoder, pretrained_conv = conv).to(device)
    # model = EEGPTCausal(pretrained_encoder=encoder).to(device)
    del state_dict
    del encoder_state

    import gc
    gc.collect()

    optimizer = torch.optim.AdamW(
        list(model.chan_conv.parameters()) +
        list(model.linear_probe1.parameters()),
        # list(model.linear_probe2.parameters()),
        weight_decay=0.01)#

    max_lr = 4e-4

    ''' training loop '''
    ######################################################
    '''step one trainig: freeze most of BART params'''
    ######################################################

    # closely follow BART paper
    for name, param in model.named_parameters():
        if param.requires_grad and 'pretrained' in name:
            # 이것도 멈추는거 한번 보자
            # pretrian에서 BART embedding시키니까 한번 그쪽도 손봐야 되기도 싶고
            if ('shared' in name) or ('embed_positions' in name) or ('encoder.layers.0' in name):
                continue
            else:
                param.requires_grad = False

    ''' set up optimizer and scheduler'''
    print('=== start Step1 training ... ===')
    optimizer_step1 = optim.SGD(model.parameters(), lr=step1_lr, momentum=0.9)
    exp_lr_scheduler_step1 = lr_scheduler.StepLR(optimizer_step1, step_size=20, gamma=0.1)

    since = time.time()

    state_dict = {
        "args":args,
        "module_state_dict": {"encoder": get_configs(**(MODELS_CONFIGS[tag])), "denoising_module": get_configs(**(MODELS_CONFIGS[tag]))},
    }

    wandb.init(project= 'SRDELTA', name = 'first_stage_'+save_name)
    model = train_model(dataloaders, device, model, optimizer_step1, exp_lr_scheduler_step1, num_epochs=num_epochs_step1, 
                        checkpoint_path_best = output_checkpoint_name_best, checkpoint_path_last = output_checkpoint_name_last, stage = 'first_stage')
    wandb.finish()
    first_step_time = time.time() - since
    print(f'Step 1 Training complete in {first_step_time//60:.0f}m {first_step_time%60:.0f}s')

    """save config"""
    cfg_dir = './save_model/config'

    if not os.path.exists(cfg_dir):
        os.makedirs(cfg_dir)


    with open(os.path.join(cfg_dir,f'{save_name}.json'), 'w') as out_config:
        json.dump(state_dict, out_config, indent = 4)
    ######################################################
    '''step two trainig: update whole model for a few iterations'''
    ######################################################
    for name, param in model.named_parameters():
        param.requires_grad = True

    ''' set up optimizer and scheduler'''
    # optimizer_step2 = optim.Adam(model.parameters(), lr=step2_lr)# (B, 56, vocab_size)
    optimizer_step2 = optim.SGD(model.parameters(), lr=step2_lr, momentum=0.9)
    exp_lr_scheduler_step2 = lr_scheduler.StepLR(optimizer_step2, step_size=30, gamma=0.1)

    print()
    print('=== start Step2 training ... ===')
    # print training layers
    # show_require_grad_layers(model)
    
    since = time.time()
    '''main loop'''
    wandb.init(project='SRDELTA', name = 'second_stage_'+save_name)
    model = train_model(dataloaders, device, model, optimizer_step2, exp_lr_scheduler_step2, num_epochs=num_epochs_step2, 
                                       checkpoint_path_best = output_checkpoint_name_best, checkpoint_path_last = output_checkpoint_name_last, stage = 'second_stage')
    wandb.finish()

    second_step_time = time.time() - since
    print(f'Step 2 Training complete in {second_step_time//60:.0f}m {second_step_time%60:.0f}s')
    
    if not os.path.exists('./results'):
        os.makedirs('./results')
    if not os.path.exists('./score_results'):
        os.makedirs('./score_results')
    eval_model(dataloaders, device, tokenizer, model, output_all_results_path = f'./results/best_{save_name}.txt' , score_results=f'./score_results/best_{save_name}.txt')
