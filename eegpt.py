# Training in 256Hz data and 4s
import torch
from torch import nn
import torch.nn.functional as F
import torch.nn as nn
import torch

from functools import partial
import random
import copy
from module.EEGPT_mcae import EEGTransformer, EEGTransformerPredictor, EEGTransformerReconstructor, apply_mask
from utils import temporal_interpolation, Conv1dWithConstraint, LinearWithConstraint
from transformers import BartForConditionalGeneration
# from module.ldm.attention import CrossAttention

# from configet_metricsgs import *
#-- use channels for model

use_channels_names = [      'FP1', 'FPZ', 'FP2', 
                               'AF3', 'AF4', 
            'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8', 
        'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 
            'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8', 
        'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8',
             'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8', 
                      'PO7', 'PO3', 'POZ',  'PO4', 'PO8', 
                               'O1', 'OZ', 'O2', ]


Kaggle_use_channels_names=[
               'FP1', 'FP2',
        'F7', 'F3', 'FZ', 'F4', 'F8',
        'T7', 'C3', 'CZ', 'C4', 'T8',
        'P7', 'P3', 'PZ', 'P4', 'P8',
                'O1', 'O2'
    ]

Zuco_use_channels_names = [
                            'FP1', 'FPZ', 'FP2', 
                        "AF7", 'AF3', 'AF4', "AF8", 
            'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8', 
        'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 
            'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8', 
        'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8',
             'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8', 
                      'PO7', "PO5", 'PO3', 'POZ', 'PO4', "PO6", 'PO8', 
                               'O1', 'OZ', 'O2', ]


class EEGPT(nn.Module):

    def __init__(self, models_configs, USE_LOSS_A=True, USE_LN=True, USE_SKIP=True, pretrained_LM = None):
        super().__init__()    
        self.USE_LOSS_A = USE_LOSS_A
        self.USE_LN     = USE_LN
        self.USE_SKIP   = USE_SKIP

        # self.pretrained_LM = pretrained_LM
        self.conv = Convolution_Block(input_size=1, hidden_dim = 56, chan_size = 62, time_size = 128, pooling_size = 64)

        
        # 659 줄
        encoder = EEGTransformer(
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
            **models_configs['encoder'])
        
        # 518 줄
        predictor = EEGTransformerPredictor(
            num_patches=encoder.num_patches,
            use_part_pred=True,################
            mlp_ratio=4.0,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.0,
            init_std=0.02,
            qkv_bias=True, 
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            **models_configs['predictor'])
        
        # 365줄
        reconstructor = EEGTransformerReconstructor(
            num_patches=encoder.num_patches,
            # patch_size=32*2,
            patch_size=32,
            mlp_ratio=4.0,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.0,
            init_std=0.02,
            qkv_bias=True, 
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            **models_configs['reconstructor'])
        

        target_encoder = copy.deepcopy(encoder)
        for p in target_encoder.parameters():
            p.requires_grad = False

        
        self.encoder        = encoder
        self.target_encoder = target_encoder
        self.predictor      = predictor
        self.reconstructor  = reconstructor
        self.chans_id       = encoder.prepare_chan_ids(Zuco_use_channels_names)
        
        self.loss_fn        = torch.nn.MSELoss()
        # self.w = nn.Parameter(torch.randn(56))


        

    def make_masks(self, num_patchs, mC_x=12, p_n_y=0.5, p_c_y=0.2):
        
        C, N = num_patchs # (62, 80)
        
        while True:
            mask_x = []# mN, mC
            mask_y = []
            mask_y_bx = []
            for i in range(N):
                c_idx = torch.randperm(C) + i*C
                if random.random()>p_n_y:
                    mask_x.append(c_idx[:mC_x])
                    mask_y_bx.append(c_idx[mC_x:])
                else:
                    mask_y.append(c_idx)
            if len(mask_x)==0: continue
            if len(mask_y_bx)==0: continue
            mask_y_bx = torch.cat(mask_y_bx, dim=0)
            mask_y_bx = mask_y_bx[torch.rand(mask_y_bx.shape)<p_c_y]
            if len(mask_y_bx)==0: continue
            break
        
        return torch.stack(mask_x, dim=0), torch.cat(mask_y+[mask_y_bx], dim=0)
    # (num_mask_x, mC_x),   # 2D tensor
    # (K,)                  # 1D tensor
    def forward_target(self, x, mask_y):
        with torch.no_grad():
            # h = self.target_encoder(x, self.chans_id.to(x))
            h = self.target_encoder(x) # [32, 32, 4, 256]
            h = F.layer_norm(h, (h.size(-1),))  # normalize over feature-dim
            C, N = self.encoder.num_patches # 62, 80

            # x = [32, 62, 2560]
            assert x.shape[-1]%N==0 and x.shape[-2]%C == 0
            block_size_c, block_size_n = x.shape[-2]//C, x.shape[-1]//N # 41,1 
            x = x.view(x.shape[0], C, block_size_c, N, block_size_n) # [32, 62, 1, 80, 41]
            # 将维度重新排列以使分块沿着通道轴和空间轴
            # 블록들이 채널 축과 공간 축에 맞춰 정렬되도록 치수를 재배열하십시오
            x = x.permute(0, 3, 1, 2, 4).contiguous() # B, N, C, bc, bn # [32, 80, 62, 1, 41
            x = x.view(x.shape[0], C, N, block_size_c * block_size_n) # [32, 62, 80, 41]
            y = apply_mask(mask_y.to(x.device), x)
            if self.USE_LN:
                y = F.layer_norm(y, (y.size(-1),))
            return h, y

    def forward_context(self, x, mask_x, mask_y):
        # z = self.encoder(x, self.chans_id.to(x), mask_x=mask_x)
        z = self.encoder(x, mask_x=mask_x)
        z, comb_z = self.predictor(z, mask_x=mask_x)
        if not self.USE_SKIP:
            comb_z = z
        r = self.reconstructor(comb_z, self.chans_id.to(x), mask_y=mask_y)
        return z, r
    

    def forward(self, x, context):
        mask_x, mask_y = self.make_masks(self.encoder.num_patches) # (62, 80) ->
        # (num_mask_x, mC_x),   # 2D tensor
        # (K,)                  # 1D tensor

        x = x.unsqueeze(1)
        x = self.conv(x)
        x = x.squeeze(-2)
        # 이 부분에 learnable parameter 넣기
        # x: (B, S, T)
        # channel-wise scaling
        # x= (x * self.w.view(1, S, 1))  # (B, S, T)

        
        # embeded_context = self.pretrained_LM.model.shared(context)

        h, y = self.forward_target(x, mask_y) # embeddied x, label
        z, r = self.forward_context(x, mask_x, mask_y) # embeded x, reconstruction EEG

        loss2 = self.loss_fn(y, r)
        if self.USE_LOSS_A:
            loss1 = self.loss_fn(h, z)
            loss  = loss1 + loss2
        else:
            loss  = loss2

        # if self.is_zuco:
        #     B, N, C, D = h.shape
        #     bar_h = h.view(B, N * C, D)
        #     det_h = bar_h.view(B, N * C, D).detach() # [32, 320, 128] torch.Size([32, 288, 128])
        #     h_hat = self.decoder(det_h)
        #     embeded_context = self.pretrained_LM.model.shared(context)
        #     de_loss = F.mse_loss(h_hat, embeded_context)
        #     # loss 스케일 맞추기
        #     loss = de_loss + loss
        
        return r, loss

class Convolution_Block(nn.Module):
    def __init__(self, input_size=1, hidden_dim = 56, chan_size = 62, time_size = 64, pooling_size = 2):
        # self.patch_size = patch_size
        super().__init__()

        self.shallownet = nn.Sequential(
            nn.Conv2d(input_size, hidden_dim, (1, time_size), (1, 2)),
            nn.Conv2d(hidden_dim, hidden_dim, (chan_size, 1), (1, 1)),
            nn.BatchNorm2d(hidden_dim),
            nn.ELU(),
            nn.AvgPool2d((1, pooling_size), (1, 1)),  # pooling acts as slicing to obtain 'patch' along the time dimension as in ViT
            nn.Dropout(0.5),
        )

        self.projection = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, (1, 1), stride=(1, 1)),  # transpose, conv could enhance fiting ability slightly
            # Rearrange('b e (h) (w) -> b (h w) e'),
        )

        self.fc = nn.Linear(1154, 1024)

    def forward(self, x):
        # b, _, _, _ = x.shape # 32, 80, 62, 128

        x = self.shallownet(x)
        x = self.projection(x)
        x = self.fc(x)
        return x

from module.diffusion_module import register_schedule, extract_into_tensor, q_sample

class EEGDiffusion(nn.Module):
    
    def __init__(self, encoder=None, conformer_module = None, noise_scheduler='linear',time_step=1000, device='cuda:0'):
        super().__init__()    

        # init model
        self.encoder = encoder        
        self.conv = conformer_module

        self.chans_num = 56
        # self.chans_num = len(56)
        # self.chans_id       = self.denoising_module.prepare_chan_ids(Zuco_use_channels_names) # 주석처리
        
        self.chan_conv       =   Conv1dWithConstraint(self.chans_num, self.chans_num, 1, max_norm=1)  
        self.linear_probe1  =   LinearWithConstraint(32, 56, max_norm=1)
        
        self.pretrained_LM = BartForConditionalGeneration.from_pretrained('facebook/bart-large')
        
        
            
        self.drop           = torch.nn.Dropout(p=0.50)
        
        self.loss_fn        = torch.nn.CrossEntropyLoss()
        self.is_sanity=True

        self._schedule = register_schedule(beta_schedule = noise_scheduler, timesteps = time_step, device= device)
        self.time_step = time_step


    def forward(self, batch, input_masks_batch= None, stage = None):
        # training_step defined the train loop.
        # It is independent of forward
        x, y = batch
        
        x = x.unsqueeze(1)
        x = self.conv(x)
        x = x.squeeze(-2)

        device = x.device
        # --- diffusion process ---
        noise = torch.randn_like(x, device=device) # B, 56, 1024
        timesteps = torch.randint(0, self.time_step, (x.shape[0],), device=device).long() # (B,)
        x_noisy = q_sample(x, timesteps, self._schedule['sqrt_alphas_cumprod'], self._schedule['sqrt_one_minus_alphas_cumprod'], noise=noise) # (B, 1024, 8)

        # predicted_noise = self.denoising_module(x_noisy, timesteps, context)
        # predicted_noise = self.denoising_module(x_noisy, timesteps=timesteps) # B, 80, 128, 4
        predicted_noise = self.encoder(x_noisy, timesteps=timesteps) # [32, 32, 4, 256]

        B, N , _, _ = predicted_noise.shape
        predicted_noise = predicted_noise.view(B,N,-1)
        predicted_noise = self.linear_probe1(predicted_noise.permute(0, 2, 1)).permute(0, 2, 1) # B, S, 56

        # --- recon latent vector z ---
        sqrt_one_minus_alphas_cumprod = extract_into_tensor(self._schedule['sqrt_one_minus_alphas_cumprod'], timesteps, predicted_noise.shape)    
        sqrt_alphas_cumprod = extract_into_tensor(self._schedule['sqrt_alphas_cumprod'], timesteps, predicted_noise.shape)
        z_recon = (x_noisy - (sqrt_one_minus_alphas_cumprod * predicted_noise)) / sqrt_alphas_cumprod # (B, 56, 106)

        # z_recon = self.proj(z_recon)
        # --- loss design ---         
        out = self.pretrained_LM(inputs_embeds = z_recon, attention_mask = input_masks_batch,
                                        return_dict = True, labels = y)# (B, 56, vocab_size)
        
        if stage == 'first_stage':
            noise_loss = F.mse_loss(predicted_noise, noise)
            loss = out.loss + noise_loss
            return out, loss
        else:
            return out, out.loss
    
    @torch.no_grad()
    def generate(
            self,
            input_embeddings_batch, input_masks_batch,dummy_decoder_input_ids,
            **kwargs,
    ):

        x = input_embeddings_batch
        x = x.unsqueeze(1)
        x = self.conv(x)
        x = x.squeeze(-2)

        device = x.device
        # --- diffusion process ---
        noise = torch.randn_like(x, device=device) # B, 56, 1024
        timesteps = torch.randint(0, self.time_step, (x.shape[0],), device=device).long() # (B,)
        x_noisy = q_sample(x, timesteps, self._schedule['sqrt_alphas_cumprod'], self._schedule['sqrt_one_minus_alphas_cumprod'], noise=noise) # (B, 1024, 8)

        # predicted_noise = self.denoising_module(x_noisy, timesteps, context)
        # predicted_noise = self.denoising_module(x_noisy, timesteps=timesteps) # B, 80, 128, 4
        predicted_noise = self.encoder(x_noisy, timesteps=timesteps) # [32, 32, 4, 256]

        B, N , _, _ = predicted_noise.shape
        predicted_noise = predicted_noise.view(B,N,-1)
        predicted_noise = self.linear_probe1(predicted_noise.permute(0, 2, 1)).permute(0, 2, 1) # B, S, 56

        # --- recon latent vector z ---
        sqrt_one_minus_alphas_cumprod = extract_into_tensor(self._schedule['sqrt_one_minus_alphas_cumprod'], timesteps, predicted_noise.shape)    
        sqrt_alphas_cumprod = extract_into_tensor(self._schedule['sqrt_alphas_cumprod'], timesteps, predicted_noise.shape)
        z_recon = (x_noisy - (sqrt_one_minus_alphas_cumprod * predicted_noise)) / sqrt_alphas_cumprod # (B, 56, 106)

        # z_recon = self.proj(z_recon)
        # --- loss design ---         

        output=self.pretrained_LM.generate(
            inputs_embeds = z_recon,
            attention_mask = input_masks_batch[:,:z_recon.shape[1]],
            decoder_input_ids = dummy_decoder_input_ids,  # Exclude the last token for decoder input
            **kwargs,
            )
        

        return output
class EEGPTCausal(nn.Module):

    def __init__(self, pretrained_encoder=None, pretrained_conv = None):
        super().__init__()    

        # init model

        self.target_encoder = pretrained_encoder
        self.conv = pretrained_conv    
        
        self.chans_num = 56
        
        self.pretrained_LM = BartForConditionalGeneration.from_pretrained('facebook/bart-large')
            
        
        self.chan_conv       =   Conv1dWithConstraint(self.chans_num, self.chans_num, 1, max_norm=1)  
        self.linear_probe1  =   LinearWithConstraint(32, 56, max_norm=1)
    
        self.drop           = torch.nn.Dropout(p=0.50)
        
        self.loss_fn        = torch.nn.CrossEntropyLoss()
        self.is_sanity=True

    
    def _forward(self, x):
        # B, C, T = x.shape

        self.target_encoder.eval()
        self.conv.eval()

        x = x.unsqueeze(1)
        x = self.conv(x)
        x = x.squeeze(-2)
        # x = x * self.target_encoder.w

        x = self.chan_conv(x)
        z = self.target_encoder(x)
        
        B, T, C, D = z.shape
        h = z.view(B, T, C * D)          
    
        return x, h

    # 여기 고쳐야 함 batch_idx 필요없게
    # 정확히 batch_idx의 역할이 먼지 알아보자
    def forward(self, batch, input_masks_batch = None):
        # training_step defined the train loop.
        # It is independent of forward
        x, y = batch
        
        x, logit = self._forward(x)
        
        logit = self.linear_probe1(logit.permute(0, 2, 1)).permute(0, 2, 1) # B, S, 56
        
        out = self.pretrained_LM(inputs_embeds = logit, attention_mask = input_masks_batch,
                                        return_dict = True, labels = y)# (B, 56, vocab_size)
        return out, out.loss
    
    @torch.no_grad()
    def generate(
            self,
            input_embeddings_batch, input_masks_batch,dummy_decoder_input_ids,
            **kwargs,
    ):

        _, logit = self._forward(input_embeddings_batch)
        output=self.pretrained_LM.generate(
            inputs_embeds = logit,
            attention_mask = input_masks_batch[:,:logit.shape[1]],
            decoder_input_ids = dummy_decoder_input_ids,  # Exclude the last token for decoder input
            **kwargs,
            )
        

        return output

from utils import *

if __name__ =="__main__":
    x = torch.rand(32, 62, 2560)
    x = x.unsqueeze(1)
    conv = Convolution_Block(input_size=1, hidden_dim = 32, chan_size = 62, time_size = 512, pooling_size = 64)
    y = conv(x)
    print(y.shape)
    encoder = EEGTransformer(
        # img_size=[58, 256*4],
        img_size= [62, 2560],
        # patch_size=32*2,
        patch_size=32,
        mlp_ratio=4.0,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        init_std=0.02,
        qkv_bias=True, 
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        embed_dim= 128,
        embed_num= 4,
        depth= 8,
        num_heads= 4,)
    
    # 518 줄
    predictor = EEGTransformerPredictor(
        num_patches=encoder.num_patches,
        use_part_pred=True,################
        mlp_ratio=4.0,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        init_std=0.02,
        qkv_bias=True, 
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        embed_dim= 128,
        embed_num= 4,
        depth= 8,
        num_heads= 4,
        predictor_embed_dim = 128)
    def make_masks(num_patchs, mC_x=12, p_n_y=0.5, p_c_y=0.2):
        
        C, N = num_patchs
        
        while True:
            mask_x = []# mN, mC
            mask_y = []
            mask_y_bx = []
            for i in range(N):
                c_idx = torch.randperm(C) + i*C
                if random.random()>p_n_y:
                    mask_x.append(c_idx[:mC_x])
                    mask_y_bx.append(c_idx[mC_x:])
                else:
                    mask_y.append(c_idx)
            if len(mask_x)==0: continue
            if len(mask_y_bx)==0: continue
            mask_y_bx = torch.cat(mask_y_bx, dim=0)
            mask_y_bx = mask_y_bx[torch.rand(mask_y_bx.shape)<p_c_y]
            if len(mask_y_bx)==0: continue
            break

    mask_x, mask_y = make_masks(encoder.num_patches)
    z = encoder(y, mask_x=mask_x)
    print(z.shape)
