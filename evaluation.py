import os
import numpy as np
import torch
from torch.utils.data import DataLoader
import pickle
import json
import time
from tqdm import tqdm
import torch.nn.functional as F
import time
from transformers import BartTokenizer, BartForConditionalGeneration
# from data.data import ZuCo_dataset
from data.data_for_rawEEG import ZuCo_dataset
# from model import *
from utils import *
from nltk.translate.bleu_score import corpus_bleu
from rouge import Rouge
import sys
# import evaluate
# from evaluate import load
import re

from functools import partial
from utils import *
# from eegpt import EEGDiffusion
from eegpt import EEGPTCausal, EEGDiffusion, Convolution_Block
from utils.configs import *
from module.EEGPT_mcae import EEGTransformer

def valid_text(s: str) -> bool:
    """알파벳·숫자 토큰이 하나 이상 남는 문장인지 확인"""
    if re.search(r'\w', s) is None:
        return ''
    return s

def remove_text_after_token(text, token='</s>'):
    # 특정 토큰 이후의 텍스트를 찾아 제거
    token_index = text.find(token)
    if token_index != -1:  # 토큰이 발견된 경우
        return text[:token_index]  # 토큰 이전까지의 텍스트 반환
    return text  # 토큰이 없으면 원본 텍스트 반환

def eval_model(dataloaders, device, tokenizer, model, output_all_results_path = './results/temp.txt' , score_results='./score_results/task.txt'):
    # modified from: https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
    target_string_list = []
    start_time = time.time()
    model.eval()   # Set model to evaluate mode
    
    target_tokens_list = []
    pred_tokens_list = []
    pred_string_list = []
    pred_tokens_list_previous = []
    pred_string_list_previous = []


    with open(output_all_results_path,'w') as f:
        vocab_size = tokenizer.vocab_size
        for sent_level_EEG, rawEEG, input_masks, input_mask_invert, target_ids, target_mask in tqdm(dataloaders['test'], file=sys.stderr):
            # load in batch
            input_embeddings_batch = rawEEG.to(device).float() # B, 56, 840
            # input_embeddings_batch = sent_level_EEG.unsqueeze(1).to(device).float()
            input_masks_batch = input_masks.to(device) # B, 56
            target_ids_batch = target_ids.to(device) # B, 56
            input_mask_invert_batch = input_mask_invert.to(device) # B, 56
            context = target_ids_batch.clone()

            target_tokens = tokenizer.convert_ids_to_tokens(target_ids_batch[0].tolist(), skip_special_tokens = True)
            target_string = tokenizer.decode(target_ids_batch[0], skip_special_tokens = True)
            # print('target ids tensor:',target_ids_batch[0])
            # print('target ids:',target_ids_batch[0].tolist())
            # print('target tokens:',target_tokens)
            # print('target string:',target_strininvert.to(device) # B, 56
            
            f.write(f'target string: {target_string}\n')

            # add to list for later calculate bleu metric
            target_tokens_list.append([target_tokens])
            target_string_list.append(target_string)
            
            """replace padding ids in target_ids with -100"""
            target_ids_batch[target_ids_batch == tokenizer.pad_token_id] = -100 

            # target_ids_batch_label = target_ids_batch.clone().detach()
            # target_ids_batch_label[target_ids_batch_label == tokenizer.pad_token_id] = -100
            
            # Original code 
            seq2seqLMoutput,_ = model(batch= (input_embeddings_batch,target_ids_batch),input_masks_batch= input_masks_batch) # (batch, time, n_class)
            logits_previous = seq2seqLMoutput.logits
            probs_previous = logits_previous[0].softmax(dim = 1)
            values_previous, predictions_previous = probs_previous.topk(1)
            predictions_previous = torch.squeeze(predictions_previous)
            predicted_string_previous = remove_text_after_token(tokenizer.decode(predictions_previous).split('</s></s>')[0].replace('<s>',''))
            
            predicted_string_previous = valid_text(predicted_string_previous)
            f.write(f'predicted string with tf: {predicted_string_previous}\n')
            predictions_previous = predictions_previous.tolist()
            truncated_prediction_previous = []
            for t in predictions_previous:
                if t != tokenizer.eos_token_id:
                    truncated_prediction_previous.append(t)
                else:
                    break
            pred_tokens_previous = tokenizer.convert_ids_to_tokens(truncated_prediction_previous, skip_special_tokens = True)
            pred_tokens_list_previous.append(pred_tokens_previous)
            pred_string_list_previous.append(predicted_string_previous)
            

            # Modify code
            dummy_decoder_input_ids = torch.tensor([[tokenizer.pad_token_id]]).to(device) # B, 1
            predictions=model.generate(input_embeddings_batch, input_masks_batch, dummy_decoder_input_ids,
                                       max_length=56,
                                       num_beams=5,
                                       do_sample=True,
                                       repetition_penalty= 5.0,
                                       no_repeat_ngram_size = 2,
                                       
                                       # num_beams=5,encoder_no_repeat_ngram_size =1,
                                       # do_sample=True, top_k=15,temperature=0.5,num_return_sequences=5,
                                       # early_stopping=True
                                       )
            
            predicted_string=tokenizer.batch_decode(predictions, skip_special_tokens=True)[0]
            # predicted_string=predicted_string.squeeze()
            
            predictions=tokenizer.encode(predicted_string)
            # print('predicted string:',predicted_string)
            predicted_string = valid_text(predicted_string)
            f.write(f'predicted string: {predicted_string}\n')
            f.write(f'################################################\n\n\n')

            # convert to int list
            # predictions = predictions.tolist() # 이미 list 형식이다. 
            truncated_prediction = []
            for t in predictions:
                if t != tokenizer.eos_token_id:
                    truncated_prediction.append(t)
                else:
                    break
            pred_tokens = tokenizer.convert_ids_to_tokens(truncated_prediction, skip_special_tokens = True)
            # print('predicted tokens:',pred_tokens)
            pred_tokens_list.append(pred_tokens)
            pred_string_list.append(predicted_string)
            # pred_tokens_list.extend(pred_tokens)
            # pred_string_list.extend(predicted_string)
            # print('################################################')
            # print()
    # print(f"pred_string_list : {pred_string_list}")
    
    """ calculate corpus bleu score """
    weights_list = [(1.0,),(0.5,0.5),(1./3.,1./3.,1./3.),(0.25,0.25,0.25,0.25)]
    corpus_bleu_scores = []
    corpus_bleu_scores_previous = []
    for weight in weights_list:
        # print('weight:',weight)
        corpus_bleu_score = corpus_bleu(target_tokens_list, pred_tokens_list, weights = weight)
        corpus_bleu_score_previous = corpus_bleu(target_tokens_list, pred_tokens_list_previous, weights = weight)
        corpus_bleu_scores.append(corpus_bleu_score)
        corpus_bleu_scores_previous.append(corpus_bleu_score_previous)
        print(f'corpus BLEU-{len(list(weight))} score:', corpus_bleu_score)
        print(f'corpus BLEU-{len(list(weight))} score with tf:', corpus_bleu_score_previous)
    
    print()
    """ calculate rouge score """
    def preprocess(text: str) -> str:
        tokens = tokenizer.tokenize(text)
        return " ".join(tokens)          # 공백 단위 토큰열
    
    pred_proc   = [preprocess(t) for t in pred_string_list]
    pred_previous_proc   = [preprocess(t) for t in pred_string_list_previous]
    target_proc = [preprocess(t) for t in target_string_list]
    rouge = Rouge()

    try:
        rouge_scores = rouge.get_scores(pred_proc, target_proc, avg = True, ignore_empty=True)
    except ValueError as e:
        rouge_scores = 'Hypothesis is empty'

    try:
        rouge_scores_previous = rouge.get_scores(pred_previous_proc, target_proc, avg = True, ignore_empty=True)
    except ValueError as e:
        rouge_scores_previous = 'Hypothesis is empty'

    print()
    """ calculate WER score """
    #wer = WordErrorRate()
    # wer_scores = wer_metric.compute(predictions=pred_string_list, references=target_string_list)
    # wer_scores_previous = wer_metric.compute(predictions=pred_string_list_previous, references=target_string_list)
    # print("WER score:", wer_scores)
    # print("WER score with tf:", wer_scores_previous)
    

    """ calculate CER score """
    # cer_scores = cer_metric.compute(predictions=pred_string_list, references=target_string_list)
    # cer_scores_previous = cer_metric.compute(predictions=pred_string_list_previous, references=target_string_list)
    # print("CER score:", cer_scores)
    # print("CER score with tf:", cer_scores_previous)


    end_time = time.time()
    print(f"Evaluation took {(end_time-start_time)/60} minutes to execute.")

     # score_results (only fix teacher-forcing)
    file_content = [
    f'corpus_bleu_score = {corpus_bleu_scores}',
    f'rouge_scores = {rouge_scores}',
    # f'wer_scores = {wer_scores}',
    # f'cer_scores = {cer_scores}',
    f'corpus_bleu_score_with_tf = {corpus_bleu_scores_previous}',
    f'rouge_scores_with_tf = {rouge_scores_previous}',
    # f'wer_scores_with_tf = {wer_scores_previous}',
    # f'cer_scores_with_tf = {cer_scores_previous}',
    ]
    if not os.path.exists(score_results):
        with open(score_results, 'w') as f:
            f.write("")
    with open(score_results, "w") as file_results:
        for line in file_content:
            if isinstance(line, list):
                for item in line:
                    file_results.write(str(item) + "\n")
            else:
                file_results.write(str(line) + "\n")
    print(f'[INFO]score results saved in {score_results}', file=sys.stderr)
    print(f'[INFO]score results saved in {score_results}')


if __name__ == '__main__': 
    batch_size = 1
    ''' get args'''
    args = get_config('eval_decoding')

    
    # test_input = args['test_input'] # word-level or rawEEG
    # print("test_input is:", test_input)

    if not os.path.exists('./results'):
        os.makedirs('./results')
    if not os.path.exists('./score_results'):
        os.makedirs('./score_results')
    save_name = 'base_model'  

    output_all_results_path = f'./results/{save_name}.txt'
    score_results = f'./score_results/{save_name}.txt'

    eeg_type_choice = args['eeg_type'] # GD
    print(f'[INFO]eeg type {eeg_type_choice}')
    
    bands_choice = args['eeg_bands'] # ['_t1','_t2','_a1','_a2','_b1','_b2','_g1','_g2'] 
    print(f'[INFO]using bands {bands_choice}')
    task_name = args['task_name']
    ''' set random seeds '''
    seed_val = args['seed']
    set_seed(seed_val)

    ''' set up device '''
    # use cuda
    if torch.cuda.is_available():  
        dev = args['cuda'] 
    else:  
        dev = "cpu"
    # CUDA_VISIBLE_DEVICES=0,1,2,3  
    device = torch.device(dev)
    print(f'[INFO]using device {dev}')

    # task_name = 'task1_task2_task3'

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

    # test dataset
    # test_set = ZuCo_dataset(whole_dataset_dicts, 'test', tokenizer, subject = subject_choice, eeg_type = eeg_type_choice, bands = bands_choice, setting = dataset_setting, test_input=test_input)
    subject_choice = 'ALL'
    test_set = ZuCo_dataset(whole_dataset_dicts, 'test', tokenizer, subject = subject_choice, bands = bands_choice, setting = dataset_setting)

    dataset_sizes = {"test_set":len(test_set)}
    print('[INFO]test_set size: ', len(test_set))
    
    # dataloaders
    test_dataloader = DataLoader(test_set, batch_size = batch_size, shuffle=False, num_workers=4)

    dataloaders = {'test':test_dataloader}

    ''' set up model '''

    

    pretrained_LM = BartForConditionalGeneration.from_pretrained('facebook/bart-large')
    vocab_size = tokenizer.vocab_size
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

    # model = EEGPTCausal(pretrained_encoder=encoder, is_zuco=True).to(device)
    # denoising_module = EEGTransformer(**save_encoder_state).to(device)

    model = EEGDiffusion(encoder = encoder, conformer_module=conv, device=device).to(device)

    # state_dict = torch.load(checkpoint_path)
    state_dict = torch.load(args['path'])

    # model_states = {}
    # for key,val in state_dict.items():
    #     if key.startwith()
        

    # new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)

    # model.load_state_dict(torch.load(checkpoint_path))

    ''' eval '''
    eval_model(dataloaders, device, tokenizer, model, output_all_results_path = output_all_results_path, score_results=score_results)
