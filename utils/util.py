import math
import torch
import random
import os
import numpy as np

import torch
import os
import sys
import numpy as np
import random
import scipy.io as sio
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Function
from einops import rearrange
import copy
import gc

import scipy
from data.Data_process.utils import EA

from torch.utils.data import Dataset,DataLoader

from scipy import stats

# from Modules.spdnet.spd import SPDTransform, SPDTangentSpace, SPDRectified,SPDVectorize
# from module.eegpt.Data_process.process_function import Load_BCIC_2a_raw_data
# from collections import Counter
# current_path = os.path.abspath('./')
# root_path = current_path # os.path.split(current_path)[0]
# sys.path.append(root_path)

# data_path = os.path.join(root_path,'Data','BCIC_2a_0_38HZ')
# if not os.path.exists(data_path):
#     print('BCIC_2a_0_38HZ数据不存在，开始初始化！')
#     Load_BCIC_2a_raw_data(0,4,[0,38])
from collections import Counter
        
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed) # CPU
    torch.cuda.manual_seed(seed) # GPU
    torch.cuda.manual_seed_all(seed) # All GPU
    os.environ['PYTHONHASHSEED'] = str(seed) # 禁止hash随机化
    torch.backends.cudnn.deterministic = True # 确保每次返回的卷积算法是确定的
    torch.backends.cudnn.benchmark = False # True的话会自动寻找最适合当前配置的高效算法，来达到优化运行效率的问题。False保证实验结果可复现

def select_devices(num_device,gpus=None):
    if gpus is None:
        gpus = torch.cuda.device_count()  
        gpus = [i for i in range(gpus)]
        
    res = []
    last_id = 0
            
    min_memory = 25447170048 // 2  
    for i in range(num_device):
        device_id = gpus[last_id%len(gpus)]
        last_id+=1
        while torch.cuda.get_device_properties(device_id).total_memory < min_memory:
            device_id = gpus[last_id%len(gpus)]
            last_id+=1
        res.append(torch.device(f'cuda:{device_id}') )
    return res

def select_free_gpu():  
    """选择空闲显存最大的GPU"""  
    gpus = torch.cuda.device_count()  
    if gpus == 0:  
        return None  
    else:  
        device_id = 0  
        min_memory = 25447170048 // 2  
        while True:
            i = random.randint(0, gpus-1)
        # for i in range(gpus):  
            mem_info = torch.cuda.get_device_properties(i)  
            # print(mem_info.total_memory)
            if mem_info.total_memory > min_memory:  
                device_id = i  
                break

        return torch.device(f'cuda:{device_id}') 

def rand_mask(feature):

    for _ in range(np.random.randint(0,4)):
        c = np.random.randint(0,22)

        a = np.random.normal(1,0.4,1)[0]

        feature[:,c] *=a
    return feature

def rand_cov(x):
    # print('xt shape:',xt.shape)
    E = torch.matmul(x, x.transpose(1,2))
    # print(E.shape)
    R = E.mean(0)
    
    U, S, V = torch.svd(R)
    R_mat = U@torch.diag(torch.rand(S.shape[0])*2)@V
    new_x = torch.einsum('n c s,r c -> n r s',x,R_mat)
    return new_x


def shuffle_data(dataset):
    x = rearrange(dataset.x,'(n i) c s->n i c s',n=16)
    y = rearrange(dataset.y,'(n i)->n i',n=16)
    new_x = []
    new_y = []

    for i in np.random.permutation(x.shape[0]):
        index = np.random.permutation(x.shape[1])
        new_x.append(x[i][index])
        new_y.append(y[i][index])

    new_x = torch.stack(new_x)
    new_y = torch.stack(new_y)
    new_x = rearrange(new_x,'a b c d->(a b) c d')
    new_y = rearrange(new_y,'a b->(a b)')

    return eeg_dataset(new_x,new_y)


def print_log(s,path="log.txt"):
    with open(path,"a+") as f:
        f.write((str(s) if type(s) is not str else s) +"\n")
def callback(res):
        print('<进程%s> subject %s accu %s' %(os.getpid(),res['sub'], str(res["accu"])))
        
        
def geban(batch_size=10, n_class=4):
    res = [random.randint(0, batch_size) for i in range(n_class-1) ]
    res.sort()
    # print(res)
    ret=[]
    last=0
    for r in res:
        ret.append(r-last)
        last=r
    ret.append(batch_size-last)
    return ret

def geban_entropy(batch_size=10, n_class=4, entropy_scope=[0,1]):
    while True:
        num_class = geban(batch_size, n_class)
        total = sum(num_class)
        ent = stats.entropy([x/total for x in num_class], base=n_class)
        if entropy_scope[0]<=ent and ent<=entropy_scope[1]: break
    return num_class

def sample(batch_size=10, n_class=4):
    res = [random.randint(0, n_class-1) for i in range(batch_size) ]
    res = Counter(res)
    ret = []
    for i in range(n_class):
        ret.append(res[i])
    return ret

def temporal_interpolation(x, desired_sequence_length, mode='nearest', use_avg=True):
    # print(x.shape)
    # squeeze and unsqueeze because these are done before batching
    if use_avg:
        x = x - torch.mean(x, dim=-2, keepdim=True)
    if len(x.shape) == 2:
        return torch.nn.functional.interpolate(x.unsqueeze(0), desired_sequence_length, mode=mode).squeeze(0)
    # Supports batch dimension
    elif len(x.shape) == 3:
        return torch.nn.functional.interpolate(x, desired_sequence_length, mode=mode)
    else:
        raise ValueError("TemporalInterpolation only support sequence of single dim channels with optional batch")

# 构建用于读取验证集和测试集数据的Dataset类
class eeg_dataset(Dataset):
    '''
    A class need to input the Dataloader in the pytorch.
    '''
    def __init__(self,feature,label,subject_id=None):
        super(eeg_dataset,self).__init__()

        self.x = feature
        self.y = label
        self.s = subject_id

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.x[index], self.y[index]
    
    def get_num_class(self, num_class=[1,1,1,1]):
        res = [[] for i in num_class]
        idxs = [i for i in range(len(self.y))]
        while sum(num_class)>0:
            i = random.choice(idxs)
            label = self.y[i]
            label = int(label)
            if num_class[label]>0:
                num_class[label]-=1
                res[label].append((self.x[i],self.y[i]))
            
        re2= []
        for r in res:
            re2.extend(r)
        x = torch.stack([x[0] for x in re2], dim=0)
        
        y = torch.stack([x[1] for x in re2], dim=0)
        
        return x, y     
    
    def get_num_subject(self, num_class=[1,1,1,1,1,1,1,1]):
        res = [[] for i in num_class]
        idxs = [i for i in range(len(self.y))]
        while sum(num_class)>0:
            i = random.choice(idxs)
            s = self.s[i]
            s = int(s)
            if num_class[s]>0:
                num_class[s]-=1
                res[s].append((self.x[i],self.y[i]))
            
        re2= []
        for r in res:
            re2.extend(r)
        x = torch.stack([x[0] for x in re2], dim=0)
        y = torch.stack([x[1] for x in re2], dim=0)
        
        return x, y        

    
def get_subj_data(sub, data_path, few_shot_number = 1, is_few_EA = False, target_sample=-1, sess=None, use_average=False):
    
    # target_y_data = []
    
    i=sub
    R=None
    source_train_x = []
    source_train_y = []
    source_valid_x = []
    source_valid_y = []
    
    if sess is not None:
        
        train_path = os.path.join(data_path,r'sub{}_sess{}_train/Data.mat'.format(i, sess))
        test_path = os.path.join(data_path,r'sub{}_sess{}_test/Data.mat'.format(i, sess))
    else:
        train_path = os.path.join(data_path,r'sub{}_train/Data.mat'.format(i))
        test_path = os.path.join(data_path,r'sub{}_test/Data.mat'.format(i))
        
    train_data = sio.loadmat(train_path)
    test_data = sio.loadmat(test_path)
    if use_average:
        train_data['x_data'] = train_data['x_data'] - train_data['x_data'].mean(-2, keepdims=True)
    if is_few_EA is True:
        session_1_x = EA(train_data['x_data'],R)
    else:
        session_1_x = train_data['x_data']

    session_1_y = train_data['y_data'].reshape(-1)

    train_x,valid_x,train_y,valid_y = train_test_split(session_1_x,session_1_y,test_size = 0.1,stratify = session_1_y)
    
    source_train_x.extend(train_x)
    source_train_y.extend(train_y)
    
    source_valid_x.extend(valid_x)
    source_valid_y.extend(valid_y)
    if use_average:
        test_data['x_data'] = test_data['x_data'] - test_data['x_data'].mean(-2, keepdims=True)
        
    if is_few_EA is True:
        session_2_x = EA(test_data['x_data'],R)
    else:
        session_2_x = test_data['x_data']

    session_2_y = test_data['y_data'].reshape(-1)

    train_x,valid_x,train_y,valid_y = train_test_split(session_2_x,session_2_y,test_size = 0.1,stratify = session_2_y)
    
    source_train_x.extend(train_x)
    source_train_y.extend(train_y)
    
    source_valid_x.extend(valid_x)
    source_valid_y.extend(valid_y)
    
    source_train_x = torch.FloatTensor(np.array(source_train_x))
    source_train_y = torch.LongTensor(np.array(source_train_y))
    

    source_valid_x = torch.FloatTensor(np.array(source_valid_x))
    source_valid_y = torch.LongTensor(np.array(source_valid_y))
    
    
    if target_sample>0:
        source_train_x = temporal_interpolation(source_train_x, target_sample)
        source_valid_x = temporal_interpolation(source_valid_x, target_sample)
        
    train_dataset = eeg_dataset(source_train_x,source_train_y)
    valid_datset  = eeg_dataset(source_valid_x,source_valid_y)
    
    return train_dataset, valid_datset

def get_data(sub,data_path,few_shot_number = 1, is_few_EA = False, target_sample=-1, use_avg=True, use_channels=None):
    
    target_session_1_path = os.path.join(data_path,r'sub{}_train/Data.mat'.format(sub))
    target_session_2_path = os.path.join(data_path,r'sub{}_test/Data.mat'.format(sub))

    session_1_data = sio.loadmat(target_session_1_path)
    session_2_data = sio.loadmat(target_session_2_path)
    R = None
    if is_few_EA is True:
        session_1_x = EA(session_1_data['x_data'],R)
    else:
        session_1_x = session_1_data['x_data']
        
    if is_few_EA is True:
        session_2_x = EA(session_2_data['x_data'],R)
    else:
        session_2_x = session_2_data['x_data']
    
    # -- debug for BCIC 2b
    test_x_1 = torch.FloatTensor(session_1_x)      
    test_y_1 = torch.LongTensor(session_1_data['y_data']).reshape(-1)

    test_x_2 = torch.FloatTensor(session_2_x)      
    test_y_2 = torch.LongTensor(session_2_data['y_data']).reshape(-1)
    
    if target_sample>0:
        test_x_1 = temporal_interpolation(test_x_1, target_sample, use_avg=use_avg)
        test_x_2 = temporal_interpolation(test_x_2, target_sample, use_avg=use_avg)
    if use_channels is not None:
        test_dataset = eeg_dataset(torch.cat([test_x_1,test_x_2],dim=0)[:,use_channels,:],torch.cat([test_y_1,test_y_2],dim=0))
    else:
        test_dataset = eeg_dataset(torch.cat([test_x_1,test_x_2],dim=0),torch.cat([test_y_1,test_y_2],dim=0))

    source_train_x = []
    source_train_y = []
    source_train_s = []
    
    source_valid_x = []
    source_valid_y = []
    source_valid_s = []
    subject_id = 0
    for i in range(1,10):
        if i == sub:
            continue
        train_path = os.path.join(data_path,r'sub{}_train/Data.mat'.format(i))
        train_data = sio.loadmat(train_path)
    
        test_path = os.path.join(data_path,r'sub{}_test/Data.mat'.format(i))
        test_data = sio.loadmat(test_path)
        if is_few_EA is True:
            session_1_x = EA(train_data['x_data'],R)
        else:
            session_1_x = train_data['x_data']

        session_1_y = train_data['y_data'].reshape(-1)

        train_x,valid_x,train_y,valid_y = train_test_split(session_1_x,session_1_y,test_size = 0.1,stratify = session_1_y)
        
        source_train_x.extend(train_x)
        source_train_y.extend(train_y)
        source_train_s.append(torch.ones((len(train_y),))*subject_id)

        source_valid_x.extend(valid_x)
        source_valid_y.extend(valid_y)
        source_valid_s.append(torch.ones((len(valid_y),))*subject_id)

        if is_few_EA is True:
            session_2_x = EA(test_data['x_data'],R)
        else:
            session_2_x = test_data['x_data']

        session_2_y = test_data['y_data'].reshape(-1)

        train_x,valid_x,train_y,valid_y = train_test_split(session_2_x,session_2_y,test_size = 0.1,stratify = session_2_y)
        
        source_train_x.extend(train_x)
        source_train_y.extend(train_y)
        source_train_s.append(torch.ones((len(train_y),))*subject_id)

        source_valid_x.extend(valid_x)
        source_valid_y.extend(valid_y)
        source_valid_s.append(torch.ones((len(valid_y),))*subject_id)
        subject_id+=1
    
    source_train_x = torch.FloatTensor(np.array(source_train_x))
    source_train_y = torch.LongTensor(np.array(source_train_y))
    source_train_s = torch.cat(source_train_s, dim=0)

    source_valid_x = torch.FloatTensor(np.array(source_valid_x))
    source_valid_y = torch.LongTensor(np.array(source_valid_y))
    source_valid_s = torch.cat(source_valid_s, dim=0)
    
    if target_sample>0:
        source_train_x = temporal_interpolation(source_train_x, target_sample, use_avg=use_avg)
        source_valid_x = temporal_interpolation(source_valid_x, target_sample, use_avg=use_avg)
        
    if use_channels is not None:
        train_dataset = eeg_dataset(source_train_x[:,use_channels,:],source_train_y,source_train_s)
    else:
        train_dataset = eeg_dataset(source_train_x,source_train_y,source_train_s)
    
    if use_channels is not None:
        valid_datset = eeg_dataset(source_valid_x[:,use_channels,:],source_valid_y,source_valid_s)
    else:
        valid_datset = eeg_dataset(source_valid_x,source_valid_y,source_valid_s)
    
    return train_dataset,valid_datset,test_dataset

def get_data_openbmi(data_path,few_shot_number = 1, is_few_EA = False, target_sample=-1):
    test_rate = 0.5
    subs_list = np.int32(np.linspace(1,54, 54))
    np.random.shuffle(subs_list)
    test_size = int(test_rate* len(subs_list))
    test_subs, train_subs = subs_list[:test_size],subs_list[test_size:]
    print(test_subs)
    source_test_x = []
    source_test_y = []
    for sub in test_subs:
        target_session_1_path = os.path.join(data_path,r'sub{}_sess1_train/Data.mat'.format(sub))
        target_session_2_path = os.path.join(data_path,r'sub{}_sess2_train/Data.mat'.format(sub))

        session_1_data = sio.loadmat(target_session_1_path)
        session_2_data = sio.loadmat(target_session_2_path)
        R = None
        if is_few_EA is True:
            session_1_x = EA(session_1_data['x_data'],R)
        else:
            session_1_x = session_1_data['x_data']
            
        if is_few_EA is True:
            session_2_x = EA(session_2_data['x_data'],R)
        else:
            session_2_x = session_2_data['x_data']
        
        # -- debug for BCIC 2b
        test_x_1 = torch.FloatTensor(session_1_x)      
        test_y_1 = torch.LongTensor(session_1_data['y_data']).reshape(-1)

        test_x_2 = torch.FloatTensor(session_2_x)      
        test_y_2 = torch.LongTensor(session_2_data['y_data']).reshape(-1)
        
        if target_sample>0:
            test_x_1 = temporal_interpolation(test_x_1, target_sample)
            test_x_2 = temporal_interpolation(test_x_2, target_sample)
        source_test_x.extend([test_x_1, test_x_2])
        source_test_y.extend([test_y_1, test_y_2])
            
    test_dataset = eeg_dataset(torch.cat(source_test_x,dim=0),torch.cat(source_test_y,dim=0))

    source_train_x = []
    source_train_y = []
    for i in train_subs:
        train_path = os.path.join(data_path,r'sub{}_sess1_train/Data.mat'.format(i))
        train_data = sio.loadmat(train_path)
    
        test_path = os.path.join(data_path,r'sub{}_sess2_train/Data.mat'.format(i))
        test_data = sio.loadmat(test_path)
        
        if is_few_EA is True:
            session_1_x = EA(train_data['x_data'],R)
        else:
            session_1_x = train_data['x_data']

        session_1_y = train_data['y_data'].reshape(-1)
        
        source_train_x.append(session_1_x)
        source_train_y.append(session_1_y)


        if is_few_EA is True:
            session_2_x = EA(test_data['x_data'],R)
        else:
            session_2_x = test_data['x_data']

        session_2_y = test_data['y_data'].reshape(-1)
        
        source_train_x.append(session_2_x)
        source_train_y.append(session_2_y)
    
    source_train_x = torch.FloatTensor(np.array(source_train_x))
    source_train_y = torch.LongTensor(np.array(source_train_y))

    
    if target_sample>0:
        source_train_x = temporal_interpolation(source_train_x, target_sample)
        source_valid_x = temporal_interpolation(source_valid_x, target_sample)
        
    train_dataset = eeg_dataset(source_train_x,source_train_y)
    
    return train_dataset,test_dataset

def get_data_Nakanishi2015(sub,data_path="Data/Nakanishi2015_8_64HZ/",few_shot_number = 1, is_few_EA = False, target_sample=-1):
    
    target_session_1_path = os.path.join(data_path,r'sub{}_train/Data.mat'.format(sub))

    session_1_data = sio.loadmat(target_session_1_path)
    R = None
    if is_few_EA is True:
        session_1_x = EA(session_1_data['x_data'],R)
    else:
        session_1_x = session_1_data['x_data']
    
    # -- debug for BCIC 2b
    test_x_1 = torch.FloatTensor(session_1_x)      
    test_y_1 = torch.LongTensor(session_1_data['y_data']).reshape(-1)
    
    if target_sample>0:
        test_x_1 = temporal_interpolation(test_x_1, target_sample)
        
    test_dataset = eeg_dataset(test_x_1,test_y_1)

    source_train_x = []
    source_train_y = []
    source_train_s = []
    
    source_valid_x = []
    source_valid_y = []
    source_valid_s = []
    subject_id = 0
    for i in tqdm(range(1,11)):
        if i == sub:
            continue
        train_path = os.path.join(data_path,r'sub{}_train/Data.mat'.format(i))
        # print(train_path)
        train_data = sio.loadmat(train_path)
        
        if is_few_EA is True:
            session_1_x = EA(train_data['x_data'],R)
        else:
            session_1_x = train_data['x_data']

        session_1_y = train_data['y_data'].reshape(-1)

        train_x,valid_x,train_y,valid_y = train_test_split(session_1_x,session_1_y,test_size = 40,stratify = session_1_y)
        
        source_train_x.extend(train_x)
        source_train_y.extend(train_y)
        source_train_s.append(torch.ones((len(train_y),))*subject_id)

        source_valid_x.extend(valid_x)
        source_valid_y.extend(valid_y)
        source_valid_s.append(torch.ones((len(valid_y),))*subject_id)
        subject_id+=1
    
    source_train_x = torch.FloatTensor(np.array(source_train_x))
    source_train_y = torch.LongTensor(np.array(source_train_y))
    source_train_s = torch.cat(source_train_s, dim=0)

    source_valid_x = torch.FloatTensor(np.array(source_valid_x))
    source_valid_y = torch.LongTensor(np.array(source_valid_y))
    source_valid_s = torch.cat(source_valid_s, dim=0)
    
    if target_sample>0:
        source_train_x = temporal_interpolation(source_train_x, target_sample)
        source_valid_x = temporal_interpolation(source_valid_x, target_sample)
        
    train_dataset = eeg_dataset(source_train_x,source_train_y,source_train_s)
    
    valid_datset = eeg_dataset(source_valid_x,source_valid_y,source_valid_s)
    
    return train_dataset,valid_datset,test_dataset

def get_data_Wang2016(sub,data_path="Data/Wang2016_4_20HZ/",few_shot_number = 1, is_few_EA = False, target_sample=-1):
    
    target_session_1_path = os.path.join(data_path,r'sub{}_train/Data.mat'.format(sub))

    session_1_data = sio.loadmat(target_session_1_path)
    R = None
    if is_few_EA is True:
        session_1_x = EA(session_1_data['x_data'],R)
    else:
        session_1_x = session_1_data['x_data']
    
    # -- debug for BCIC 2b
    test_x_1 = torch.FloatTensor(session_1_x)      
    test_y_1 = torch.LongTensor(session_1_data['y_data']).reshape(-1)
    
    if target_sample>0:
        test_x_1 = temporal_interpolation(test_x_1, target_sample)
        
    test_dataset = eeg_dataset(test_x_1,test_y_1)

    source_train_x = []
    source_train_y = []
    source_train_s = []
    
    source_valid_x = []
    source_valid_y = []
    source_valid_s = []
    subject_id = 0
    for i in tqdm(range(1,36)):
        if i == sub:
            continue
        train_path = os.path.join(data_path,r'sub{}_train/Data.mat'.format(i))
        # print(train_path)
        train_data = sio.loadmat(train_path)
        
        if is_few_EA is True:
            session_1_x = EA(train_data['x_data'],R)
        else:
            session_1_x = train_data['x_data']

        session_1_y = train_data['y_data'].reshape(-1)

        train_x,valid_x,train_y,valid_y = train_test_split(session_1_x,session_1_y,test_size = 40,stratify = session_1_y)
        
        source_train_x.extend(train_x)
        source_train_y.extend(train_y)
        source_train_s.append(torch.ones((len(train_y),))*subject_id)

        source_valid_x.extend(valid_x)
        source_valid_y.extend(valid_y)
        source_valid_s.append(torch.ones((len(valid_y),))*subject_id)
        subject_id+=1
    
    source_train_x = torch.FloatTensor(np.array(source_train_x))
    source_train_y = torch.LongTensor(np.array(source_train_y))
    source_train_s = torch.cat(source_train_s, dim=0)

    source_valid_x = torch.FloatTensor(np.array(source_valid_x))
    source_valid_y = torch.LongTensor(np.array(source_valid_y))
    source_valid_s = torch.cat(source_valid_s, dim=0)
    
    if target_sample>0:
        source_train_x = temporal_interpolation(source_train_x, target_sample)
        source_valid_x = temporal_interpolation(source_valid_x, target_sample)
        
    train_dataset = eeg_dataset(source_train_x,source_train_y,source_train_s)
    
    valid_datset = eeg_dataset(source_valid_x,source_valid_y,source_valid_s)
    
    return train_dataset,valid_datset,test_dataset
if __name__=="__main__":
    train_dataset,valid_dataset,test_dataset = get_data_Wang2016(1,"Data/Wang2016_4_20HZ/", 1, is_few_EA = False, target_sample=-1)
    # # train_dataset,valid_dataset,test_dataset = get_data(1,data_path,1,True)
    # avg_ent = 0 

    # for i in range(1000):
    #     # print(geban()) 
    #     # print(sample())
    #     num_class = geban_entropy(entropy_scope=[1.2,1e6])#geban()#sample()
    #     total = sum(num_class)
    #     num_class = [x/total for x in num_class]
    #     # print(num_class)
    #     # print(sum([-x*(math.log(x)) for x in num_class if x>0]))
    #     ent = stats.entropy(num_class) 
    #     avg_ent+=ent
    #     print(avg_ent/1000) # sample 1.2110981470145854 geban 0.9734407215366253
    
    
import mne
import torch
from tqdm import tqdm
import pandas as pd 
import csv
import numpy as np
import os
import scipy.io as scio

import random
mne.set_log_level("ERROR")

def min_max_normalize(x: torch.Tensor, data_max=None, data_min=None, low=-1, high=1):
    if data_max is not None:
        max_scale = data_max - data_min
        scale = 2 * (torch.clamp_max((x.max() - x.min()) / max_scale, 1.0) - 0.5)
        
    if len(x.shape) == 2:
        xmin = x.min()
        xmax = x.max()
        if xmax - xmin == 0:
            x = 0
            return x
    elif len(x.shape) == 3:
        xmin = torch.min(torch.min(x, keepdim=True, dim=1)[0], keepdim=True, dim=-1)[0]
        xmax = torch.max(torch.max(x, keepdim=True, dim=1)[0], keepdim=True, dim=-1)[0]
        constant_trials = (xmax - xmin) == 0
        if torch.any(constant_trials):
            # If normalizing multiple trials, stabilize the normalization
            xmax[constant_trials] = xmax[constant_trials] + 1e-6

    x = (x - xmin) / (xmax - xmin)

    # Now all scaled 0 -> 1, remove 0.5 bias
    x -= 0.5
    # Adjust for low/high bias and scale up
    x += (high + low) / 2
    x  = (high - low) * x
    
    if data_max is not None:
        x = torch.cat([x, torch.ones((1, x.shape[-1])).to(x)*scale])
    return x
    
    
use_channels_names=[
               'FP1', 'FP2',
        'F7', 'F3', 'FZ', 'F4', 'F8',
        'T7', 'C3', 'CZ', 'C4', 'T8',
        'P7', 'P3', 'PZ', 'P4', 'P8',
                'O1', 'O2'
    ]

# -- read Kaggle ERN
ch_names_kaggle_ern = list("Fp1,Fp2,AF7,AF3,AF4,AF8,F7,F5,F3,F1,Fz,F2,F4,F6,F8,FT7,FC5,FC3,FC1,FCz,FC2,FC4,FC6,FT8,T7,C5,C3,C1,Cz,C2,C4,C6,T8,TP7,CP5,CP3,CP1,CPz,CP2,CP4,CP6,TP8,P7,P5,P3,P1,Pz,P2,P4,P6,P8,PO7,POz,PO8,O1,O2".split(','))

def read_csv_epochs(filename, tmin, tlen, use_channels_names=use_channels_names, data_max=None, data_min=None):
    sample_rate = 200
    raw = pd.read_csv(filename)
    
    data = torch.tensor(raw.iloc[:,1:-2].values) # exclude time EOG Feedback
    feed = torch.tensor(raw['FeedBackEvent'].values)
    stim_pos = torch.nonzero(feed>0)
    # print(stim_pos)
    datas = []
    
    # -- get channel id by use chan names
    if use_channels_names is not None:
        choice_channels = []
        for ch in use_channels_names:
            choice_channels.append([x.lower().strip('.') for x in ch_names_kaggle_ern].index(ch.lower()))
        use_channels = choice_channels
    if data_max is not None: use_channels+=[-1]
    
    xform = lambda x: min_max_normalize(x, data_max, data_min)
    
    for fb, pos in enumerate(stim_pos, 1):
        start_i = max(pos + int(sample_rate * tmin), 0)
        end___i = min(start_i + int(sample_rate * tlen), len(feed))
        # print(start_i, end___i)
        trial = data[start_i:end___i, :].clone().detach().cpu().numpy().T
        # print(trial.shape)
        info = mne.create_info(
            ch_names=[str(i) for i in range(trial.shape[0])],
            ch_types="eeg",  # channel type
            sfreq=200,  # frequency
            #
        )
        raw = mne.io.RawArray(trial, info)  # create raw
        # raw = raw.filter(5,40)
        # raw = raw.resample(256)
        
        trial = torch.tensor(raw.get_data()).float()

        trial = xform(trial)
        if use_channels_names is not None:
            trial = trial[use_channels]
        datas.append(trial)
    return datas
    
def read_kaggle_ern_test(
                    path     = "datas/",
                    subjects = [1,3,4,5,8,9,10,15,19,25],#
                    sessions = [1,2,3,4,5],#
                    tmin     = -0.7,
                    tlen     = 2,
                    data_max=None,
                    data_min=None,
                    use_channels_names = use_channels_names,
                    ):
    # -- read labels
    labels = pd.read_csv(os.path.join(path, 'KaggleERN', 'true_labels.csv'))['label']
    
    # -- read datas
    label_id = 0
    datas = []
    for i in tqdm(subjects):
        for j in sessions:
            filename = os.path.join(path, "KaggleERN", "test", "Data_S{:02d}_Sess{:02d}.csv".format(i,j))

            # -- read data
            for data in read_csv_epochs(filename, tmin=tmin, tlen=tlen, data_max=data_max, data_min=data_min, use_channels_names = use_channels_names): 
                label = labels[label_id]
                label_id += 1
                datas.append((data, int(label)))
    return datas

def read_kaggle_ern_train(
                    path     = "datas/",
                    subjects = [2,6,7,11,12,13,14,16,17,18,20,21,22,23,24,26, ],#
                    sessions = [1,2,3,4,5],#
                    tmin     = -0.7,
                    tlen     = 2,
                    data_max=None,
                    data_min=None,
                    use_channels_names = use_channels_names,
                    ):
    # -- read labels
    labels = []
    with open(os.path.join(path, 'KaggleERN', 'TrainLabels.csv'), 'r') as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if(i>0): labels.append(row)
    labels = dict(labels) # [['S02_Sess01_FB001', '1'],
    
    # -- read datas
    datas = []
    for i in tqdm(subjects):
        for j in sessions:
            if i>9:
                if(i == 22 and j == 5):
                    print("Skipped error file " + "KagglERN/train/Data_S"+ str(i)+"_Sess0"+str(j)+".csv" )
                else:
                    filename = os.path.join(path, "KaggleERN","train","Data_S"+ str(i)+"_Sess0"+str(j)+".csv")
            else:
                filename = os.path.join(path, "KaggleERN", "train", "Data_S0"+ str(i)+"_Sess0"+str(j)+".csv")
            
            # -- read data
            for fb,trial in enumerate(read_csv_epochs(filename, tmin=tmin, tlen=tlen, data_max=data_max, data_min=data_min, use_channels_names = use_channels_names),1): 
                label = labels["S{:02d}_Sess{:02d}_FB{:03d}".format(i,j,fb)]
                datas.append((trial, int(label)))
    return datas

#  chs: 25 EEG
#  custom_ref_applied: False
#  highpass: 0.5 Hz
#  lowpass: 100.0 Hz
#  meas_date: 2005-01-19 12:00:00 UTC
#  nchan: 25
#  projs: []
#  sfreq: 250.0 Hz
if __name__=="__main__":
    # datas = read_edf_epochs('datas\\PhysioNetMI\\S001\\S001R04.edf')
    # print(datas[0])
    # subject_data = read_physionetmi()
    # print(len(subject_data))
    path = "D:\\Dav\\PythonScripts\\BCI\\TORCHEEGBCI\\backup"
    data = read_kaggle_ern_test(path, subjects=[1], sessions=[1])
    print(data[0][0].shape)
    # read_kaggle_ern_train(path)


def InfoNCELoss(pred, target, t=0.):
    # B, NC, D
    NC, B, D = pred.shape
    similarity = torch.matmul(pred, target.transpose(1,2)) * math.exp(t)#  NC, B, B
    
    label = torch.arange(B).repeat(repeats=(NC,)).to(pred.device)
    
    logit1 = similarity.view((NC*B, B))
    accuracy1 = ((torch.argmax(logit1, dim=-1)==label)*1.0).mean()
    loss1 = torch.nn.functional.cross_entropy(logit1, label)
    
    logit2 = similarity.transpose(1,2).contiguous().view((NC*B, B))
    accuracy2 = ((torch.argmax(logit2, dim=-1)==label)*1.0).mean()
    loss2 = torch.nn.functional.cross_entropy(logit2, label)
    
    return (loss1+loss2)/2, (accuracy1 + accuracy2)/2

def BatchMAE_InfoNCELoss(pred, target, t=0.):
    # B, N, D
    # 对比时空维
    pred1    = pred - pred.mean(dim=0)
    target1  = target - target.mean(dim=0)
    loss1, accuracy1 = InfoNCELoss(pred1, target1)
    
    # 对比Batch维
    pred2    = pred.transpose(0,1) # N,B,D
    target2  = target.transpose(0,1) 
    
    loss2, accuracy2 = InfoNCELoss(pred2, target2)
    
    return loss1, loss2, accuracy1, accuracy2

def CoupleInfoNCELoss(pred, target, t=0.):
    
    # 对比Batch维
    pred2    = pred.flatten(1,2).transpose(0,1) # NC,B,D
    target2  = target.flatten(1,2).transpose(0,1) 
    
    loss2, accuracy2 = InfoNCELoss(pred2, target2)
    
    # 对比时间维
    pred1    = pred.transpose(1,2).flatten(0,1)# BC, N, D
    target1  = target.transpose(1,2).flatten(0,1)
    loss1, accuracy1 = InfoNCELoss(pred1, target1)
    
    # 对比Channel维
    pred3    = pred - pred.mean(dim=0)
    pred3    = pred3.flatten(0,1) # BN, C, D
    # pred3    = pred.flatten(0,1).transpose(0,2) # D, C, BN
    # pred3    = torch.layer_norm(pred3, normalized_shape=pred3.shape[-1:]).transpose(0,2) # BN, C, D
    
    
    # target3  = target.flatten(0,1).transpose(0,2) # D, C, BN
    # target3  = torch.layer_norm(target3, normalized_shape=target3.shape[-1:]).transpose(0,2)
    
    target3  = target - target.mean(dim=0)
    target3  = target3.flatten(0,1)
    
    loss3, accuracy3 = InfoNCELoss(pred3, target3)
    return loss1, loss2, loss3, accuracy1, accuracy2, accuracy3
    
def _generate_negatives(z, num_negatives=20):
    """Generate negative samples to compare each sequence location against"""
    batch_size, feat, full_len = z.shape
    z_k = z.permute([0, 2, 1]).reshape(-1, feat)
    with torch.no_grad():
        # candidates = torch.arange(full_len).unsqueeze(-1).expand(-1, self.num_negatives).flatten()
        negative_inds = torch.randint(0, full_len, size=(batch_size, full_len * num_negatives))
        # From wav2vec 2.0 implementation, I don't understand
        # negative_inds[negative_inds >= candidates] += 1

        for i in range(1, batch_size):
            negative_inds[i] += i * full_len

    z_k = z_k[negative_inds.view(-1)].view(batch_size, full_len, num_negatives, feat)
    return z_k, negative_inds

def _calculate_similarity(z, c, negatives, temp=0.1):
    c = c.permute([0, 2, 1]).unsqueeze(-2) # b, s, 1, t
    z = z.permute([0, 2, 1]).unsqueeze(-2)

    # negatives不含冲突的损失项
    negative_in_target = (c == negatives).all(dim=-1) | (z == negatives).all(dim=-1)

    targets = torch.cat([c, negatives], dim=-2)
    # print(targets.shape)
    logits = torch.nn.functional.cosine_similarity(z, targets, dim=-1) / temp

    if negative_in_target.any():
        # print(negative_in_target.shape, logits.shape)
        logits[:,:,1:][negative_in_target] = float("-inf")
    
    return logits.view(-1, logits.shape[-1])

class SelfSuperviseLoss():
    def __init__(self, device='cuda', beta=1.0, num_negatives=10) -> None:
        self.beta = beta
        self.device = device
        self.num_negatives = num_negatives
        self.loss_fn = torch.nn.CrossEntropyLoss().to(device)

    def __call__(self, dec_data, enc_data):
        negatives, _ = _generate_negatives(enc_data, num_negatives=self.num_negatives)
        # Prediction -> batch_size x predict_length x predict_length
        logits = _calculate_similarity(enc_data, dec_data, negatives)
        labels = torch.zeros(logits.shape[0], device=logits.device, dtype=torch.long)
        return self.loss_fn(logits, labels) + self.beta * enc_data.pow(2).mean(), logits 
    
    

class AverageMeter(object):
    """computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.max = float('-inf')
        self.min = float('inf')
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        try:
            self.max = max(val, self.max)
            self.min = min(val, self.min)
        except Exception:
            pass
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def grad_logger(named_params):
    stats = AverageMeter()
    stats.first_layer = None
    stats.last_layer = None
    for n, p in named_params:
        if (p.grad is not None) and not (n.endswith('.bias') or len(p.shape) == 1):
            grad_norm = float(torch.norm(p.grad.data))
            stats.update(grad_norm)
            if 'qkv' in n:
                stats.last_layer = grad_norm
                if stats.first_layer is None:
                    stats.first_layer = grad_norm
    if stats.first_layer is None or stats.last_layer is None:
        stats.first_layer = stats.last_layer = 0.
    return stats

class WarmupCosineSchedule(object):

    def __init__(
        self,
        optimizer,
        warmup_steps,
        start_lr,
        ref_lr,
        T_max,
        last_epoch=-1,
        final_lr=0.
    ):
        self.optimizer = optimizer
        self.start_lr = start_lr
        self.ref_lr = ref_lr
        self.final_lr = final_lr
        self.warmup_steps = warmup_steps
        self.T_max = T_max - warmup_steps
        self._step = 0.

    def step(self):
        self._step += 1
        if self._step < self.warmup_steps:
            progress = float(self._step) / float(max(1, self.warmup_steps))
            new_lr = self.start_lr + progress * (self.ref_lr - self.start_lr)
        else:
            # -- progress after warmup
            progress = float(self._step - self.warmup_steps) / float(max(1, self.T_max))
            new_lr = max(self.final_lr,
                         self.final_lr + (self.ref_lr - self.final_lr) * 0.5 * (1. + math.cos(math.pi * progress)))

        for group in self.optimizer.param_groups:
            group['lr'] = new_lr

        return new_lr


class CosineWDSchedule(object):

    def __init__(
        self,
        optimizer,
        ref_wd,
        T_max,
        final_wd=0.
    ):
        self.optimizer = optimizer
        self.ref_wd = ref_wd
        self.final_wd = final_wd
        self.T_max = T_max
        self._step = 0.

    def step(self):
        self._step += 1
        progress = self._step / self.T_max
        new_wd = self.final_wd + (self.ref_wd - self.final_wd) * 0.5 * (1. + math.cos(math.pi * progress))

        if self.final_wd <= self.ref_wd:
            new_wd = max(self.final_wd, new_wd)
        else:
            new_wd = min(self.final_wd, new_wd)

        for group in self.optimizer.param_groups:
            if ('WD_exclude' not in group) or not group['WD_exclude']:
                group['weight_decay'] = new_wd
        return new_wd

def seed_torch(seed=1029):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True

import torch.nn as nn
import torch
from torch.autograd import Function

class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None

class Conv2dWithConstraint(nn.Conv2d):
    '''
    Lawhern V J, Solon A J, Waytowich N R, et al. EEGNet: a compact convolutional neural network for EEG-based brain–computer interfaces[J]. Journal of neural engineering, 2018, 15(5): 056013.
    '''
    def __init__(self, *args, doWeightNorm = True, max_norm=1, **kwargs):
        self.max_norm = max_norm
        self.doWeightNorm = doWeightNorm
        super(Conv2dWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        if self.doWeightNorm: 
            self.weight.data = torch.renorm(
                self.weight.data, p=2, dim=0, maxnorm=self.max_norm
            )
        return super(Conv2dWithConstraint, self).forward(x)

class Conv1dWithConstraint(nn.Conv1d):
    '''
    Lawhern V J, Solon A J, Waytowich N R, et al. EEGNet: a compact convolutional neural network for EEG-based brain–computer interfaces[J]. Journal of neural engineering, 2018, 15(5): 056013.
    '''
    def __init__(self, *args, doWeightNorm = True, max_norm=1, **kwargs):
        self.max_norm = max_norm
        self.doWeightNorm = doWeightNorm
        super(Conv1dWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        if self.doWeightNorm: 
            self.weight.data = torch.renorm(
                self.weight.data, p=2, dim=0, maxnorm=self.max_norm
            )
        return super(Conv1dWithConstraint, self).forward(x)

class LinearWithConstraint(nn.Linear):
    def __init__(self, *args, doWeightNorm = True, max_norm=1, **kwargs):
        self.max_norm = max_norm
        self.doWeightNorm = doWeightNorm
        super(LinearWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        if self.doWeightNorm: 
            self.weight.data = torch.renorm(
                self.weight.data, p=2, dim=0, maxnorm=self.max_norm
            )
        return super(LinearWithConstraint, self).forward(x)


def SMMDL_marginal(Cs,Ct):

    '''
    The SMMDL used in the CRGNet.
    Arg:
        Cs:The source input which shape is NxdXd.
        Ct:The target input which shape is Nxdxd.
    '''
    
    Cs = torch.mean(Cs,dim=0)
    Ct = torch.mean(Ct,dim=0)
    
    # loss = torch.mean((Cs-Ct)**2)
    loss = torch.mean(torch.mul((Cs-Ct), (Cs-Ct)))
    
    return loss

def SMMDL_conditional(Cs,s_label,Ct,t_label):
  
    '''
    The Conditional SMMDL of the source and target data.
    Arg:
        Cs:The source input which shape is NxdXd.
        s_label:The label of Cs data.
        Ct:The target input which shape is Nxdxd.
        t_label:The label of Ct data.
    '''
    s_label = s_label.reshape(-1)
    t_label = t_label.reshape(-1)
    
    class_unique = torch.unique(s_label)
    
    class_num = len(class_unique)
    all_loss = 0.0
    
    for c in class_unique:
        s_index = (s_label == c)
        t_index = (t_label == c)
        # print(t_index)
        if torch.sum(t_index)==0:
            class_num-=1
            continue
        c_Cs = Cs[s_index]
        c_Ct = Ct[t_index]
        m_Cs = torch.mean(c_Cs,dim = 0)
        m_Ct = torch.mean(c_Ct,dim = 0)
        loss = torch.mean((m_Cs-m_Ct)**2)
        all_loss +=loss
        
    if class_num == 0:
        return 0
    
    return all_loss/class_num   

import torch
import torch.nn as nn
import os
import numpy as np
from dataclasses import dataclass, field
from typing import Callable, Dict
# from model.decoding_model import DELTA
# from model.pretrain_model import pretrainedmodule
from transformers import PreTrainedModel
# from model.wrappers import Wrapper_pretrain
from typing import List

def get_linear_annealing_weight(step, start_step, end_step):
    """
    선형 스케줄에 따른 KL 가중치를 계산합니다.
    특정 스텝 구간(start_step ~ end_step) 동안 가중치가 0에서 1로 선형적으로 증가합니다.

    Args:
        step (int): 현재 학습 스텝.
        start_step (int): 어닐링이 시작되는 스텝 (가중치 0).
        end_step (int): 어닐링이 종료되는 스텝 (가중치 1).

    Returns:
        float: 계산된 KL 가중치 (0.0에서 1.0 사이).
    """
    if step < start_step:
        return 0.0
    # (step - start_step) / (end_step - start_step)을 계산하여 선형적으로 증가
    # min(..., 1.0)을 통해 가중치가 1을 넘지 않도록 보장
    return min(1.0, (step - start_step) / (end_step - start_step))


def set_seed(seed_val):
# random.seed(seed_val)
    os.environ['PYTHONHASHSEED'] = str(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed(seed_val)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def show_require_grad_layers(model):
    print()
    print(' require_grad layers:')
    # sanity check
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(' ', name)

class EarlyStopping:
    def __init__(self, patience: int = 10, delta: float = 0.0):
        """
        Args:
            patience (int): 개선이 없다고 판단할 연속 epoch 수
            delta (float): '개선'으로 간주할 최소 손실 감소 폭
        """
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_loss: float) -> bool:
        score = -val_loss                       # 손실이 작을수록 좋은 모델
        if self.best_score is None:
            self.best_score = score
            return False
        
        if score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
        
        return self.early_stop

def checkpoint(func, inputs, params, flag):
    """
    Evaluate a function without caching intermediate activations, allowing for
    reduced memory at the expense of extra compute in the backward pass.
    :param func: the function to evaluate.
    :param inputs: the argument sequence to pass to `func`.
    :param params: a sequence of parameters `func` depends on but does not
                   explicitly take as arguments.
    :param flag: if False, disable gradient checkpointing.
    """
    if flag:
        args = tuple(inputs) + tuple(params)
        return CheckpointFunction.apply(func, len(inputs), *args)
    else:
        return func(*inputs)

class CheckpointFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, run_function, length, *args):
        ctx.run_function = run_function
        ctx.input_tensors = list(args[:length])
        ctx.input_params = list(args[length:])

        with torch.no_grad():
            output_tensors = ctx.run_function(*ctx.input_tensors)
        return output_tensors

    @staticmethod
    def backward(ctx, *output_grads):
        ctx.input_tensors = [x.detach().requires_grad_(True) for x in ctx.input_tensors]
        with torch.enable_grad():
            # Fixes a bug where the first op in run_function modifies the
            # Tensor storage in place, which is not allowed for detach()'d
            # Tensors.
            shallow_copies = [x.view_as(x) for x in ctx.input_tensors]
            output_tensors = ctx.run_function(*shallow_copies)
        input_grads = torch.autograd.grad(
            output_tensors,
            ctx.input_tensors + ctx.input_params,
            output_grads,
            allow_unused=True,
        )
        del ctx.input_tensors
        del ctx.input_params
        del output_tensors
        return (None, None) + input_grads