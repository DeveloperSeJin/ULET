# Training in 256Hz data and 4s
import torch

from eegpt import EEGPT
from code.ULET.utils.configs import *
from utils.config import get_config
torch.set_float32_matmul_precision("medium")
import os
import numpy as np
from utils import CosineWDSchedule
import time
import copy
from tqdm import tqdm
from utils import CosineWDSchedule
import os, tempfile
import math
import torchvision

# TMPDIR = "/home/pymp_tmp"
# os.environ["TMPDIR"] = TMPDIR
# os.environ["TEMP"] = TMPDIR
# os.environ["TMP"] = TMPDIR
# tempfile.tempdir = TMPDIR

#-- modeling
def seed_torch(seed=1029):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True


def train(dataloaders, device, model, optimizer, lr_scheduler, wd_scheduler, momentum_scheduler, epochs, checkpoint_path='./chekpoints/temp.pt'):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float('inf')

    since = time.time()

    for epoch in range(1, epochs+1):
        print('Epoch {}/{}'.format(epoch, epochs))
        print('-' * 10)

        for phase in ['train', 'dev']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            for step, (x, y) in tqdm(enumerate(dataloaders[phase])):
                
                x = x.to(device)
                y = y.to(device)
                if phase == 'train':
                    # 원본 코드에서 batch단위를 자꾸 받아오는게 pylightning에서는 .fit해서 batch를 따로 알 수가 없구나
                    wd_scheduler.step()
                with torch.set_grad_enabled(phase == 'train'):
                    optimizer.zero_grad()
                    
                    loss = model(x, None)

                    if phase == 'train':   
                        loss.backward()
                        optimizer.step()
                        lr_scheduler.step()
                        with torch.no_grad():
                            m = next(momentum_scheduler)
                            for param_q, param_k in zip(model.encoder.parameters(), model.target_encoder.parameters()):
                                param_k.data.mul_(m).add_((1.-m) * param_q.detach().data) 
                        
                running_loss += loss.detach().item() * x.size()[0] # batch loss

            
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            print('{} Loss: {:.4f}'.format(phase, epoch_loss))
                
            # deep copy the model
            if phase == 'dev':
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
                    '''save checkpoint'''
                    torch.save(best_model_wts, checkpoint_path)
                    print(f'update best on dev checkpoint: {checkpoint_path}')

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val loss: {:4f}'.format(best_loss))

    base_dir = os.path.dirname(checkpoint_path)
    base_name = os.path.basename(checkpoint_path) 
    last_path = os.path.join(base_dir, base_name.split('.')[0]+'_last.pt')
    torch.save(best_model_wts, last_path)
    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

# init model
if __name__ == "__main__":
    args = get_config('pretrain')
    seed_torch(7)
    
    save_path = args['save_path']
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    experiment_name = 'EEGPT' if args['save_name'] is None else args['save_name']
    
    save_name = os.path.join(save_path, f'{experiment_name}_{tag}_{variant}.pt')
    
    max_epochs = 200
    max_lr = 5e-4
    batch_size=64
    devices=[args['cuda']]

    ''' set up device '''
    # use cuda
    if torch.cuda.is_available():  
        # dev = "cuda:3" 
        dev = args['cuda'] 
    else:  
        dev = "cpu"
    # CUDA_VISIBLE_DEVICES=0,1
    device = torch.device(dev)

    model = EEGPT(get_configs(**(MODELS_CONFIGS[tag])), 
                    USE_LOSS_A =(variant != "A"),
                    USE_LN     =(variant != "B"),
                    USE_SKIP   =(variant != "C")).to(device)
    
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
    train_dataset = torchvision.datasets.DatasetFolder(root="./dataset/merged/TrainFolder/", loader=load_fn,  extensions=['.edf'])
    valid_dataset = torchvision.datasets.DatasetFolder(root="./dataset/merged/ValidFolder/", loader=load_fn, extensions=['.edf'])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=16, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, num_workers=16, shuffle=False)


    steps_per_epoch = math.ceil(len(train_loader)/len(devices))
    dataloaders = {'train': train_loader, 'dev': valid_loader}

    optimizer = torch.optim.AdamW(param_groups, lr=6e-5)        
    
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=max_lr, steps_per_epoch=steps_per_epoch, 
                                                        epochs=max_epochs,
                                                        div_factor = 2,
                                                        final_div_factor=8,
                                                        pct_start = 0.2 ,
                                                        )

    wd_scheduler = CosineWDSchedule(
                        optimizer,
                        ref_wd=1e-6,
                        final_wd=1e-6,
                        T_max=int(max_epochs*steps_per_epoch))
    ema = [0.996,1.0]
    momentum_scheduler = (ema[0] + i*(ema[1]-ema[0])/(steps_per_epoch*max_epochs)
                        for i in range(int(steps_per_epoch*max_epochs)+1))


    model = train(dataloaders, device, model, optimizer, lr_scheduler, wd_scheduler, momentum_scheduler, max_epochs, checkpoint_path=save_name)