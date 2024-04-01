import argparse
import sys
from typing import List

from transformers import CLIPVisionModelWithProjection, AutoTokenizer, AutoModelForCausalLM
from diffusers import KandinskyV22Pipeline, KandinskyV22PriorPipeline
from datasets import load_dataset
import torch
from torch import nn

from torch.utils.data import Dataset
import random
from torch.utils.data import DataLoader
from itertools import chain

from tqdm import tqdm
import wandb
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.lr_scheduler import StepLR, MultiStepLR, CosineAnnealingLR
from warmup_scheduler import GradualWarmupScheduler


from dataset import CLIPTextEmbedsKandinskyV22
from tiny_gill import TinyGILL, prepare_lm_and_tokenizer
from train import train_epoch
from generate import create_generator
from param_count import save_param_count_txt


        
def parse_args(args):
    parser = argparse.ArgumentParser(description='GILL training')
    parser.add_argument('--batch-size', default=48, type=int, metavar='N')
    parser.add_argument('--projection-method', default='CLIP-like', 
                        choices=['sum', 'CLIP-like'], type=str)
    parser.add_argument('--grad_mask_coef', default=1., type=float, metavar='N')
    parser.add_argument('--lr-base', default=1e-3, type=float, metavar='N')
    parser.add_argument('--lr-min', default=1e-4, type=float, metavar='N')
    parser.add_argument('--lr-period', default=4200, type=int, metavar='N')
    parser.add_argument('--loss-variant', default='both',  
                        choices=['both', 'promt_emb_only', 'hidden_states_only'], type=str)
    parser.add_argument('--accum-steps', default=15, type=int, metavar='N')
    parser.add_argument('--num-img-tokens', default=4, type=int, metavar='N')
    return parser.parse_args(args)


def tinygill_run(args):
    args = parse_args(args)
    
    batch_size = args.batch_size
    projection_method = args.projection_method
    grad_mask_coef = args.grad_mask_coef
    lr_base = args.lr_base
    lr_min = args.lr_min
    lr_period = args.lr_period
    loss_variant = args.loss_variant
    accum_steps = args.accum_steps
    num_img_tokens = args.num_img_tokens
    
    DEVICE =  torch.device('cuda:0')

    ############################################################################
    # 1. Dataset
    ds_train = CLIPTextEmbedsKandinskyV22(device=DEVICE)
    train_dataloader = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
    print('ds_size', len(ds_train))
    print('num_of_batches', len(train_dataloader))
    
    ############################################################################
    # 2. Prepare model
    lm_extended, tokenizer_extended, img_tokens_vocab_idx = prepare_lm_and_tokenizer(device=DEVICE, num_img_tokens=num_img_tokens)
    
    ############################################################################
    # 3. Train
    model = TinyGILL(
        lm_extended,
        img_tokens_vocab_idx,
        in_dim=2048,
        out_dim=1280,
        projection_method=projection_method,
        pad_token_id=tokenizer_extended.pad_token_id,
        device=DEVICE,
    ).half().to(DEVICE)

    model.lm.eval()
    for param in model.lm.parameters():
        param.requires_grad = False
    model.lm.model.embed_tokens.weight.requires_grad = True 

    save_param_count_txt(model, log_dir = './')
    
    wandb.init(project="LLM2CLIP")
    
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr_base, eps=1e-3)
    scheduler = CosineAnnealingLR(optimizer, lr_period, eta_min=lr_min, last_epoch=-1)

    model.train()
    train_epoch(
        train_dataloader, 
        model, 
        tokenizer_extended,
        img_tokens_vocab_idx,
        loss_fn,
        scheduler,
        optimizer,
        accum_steps=accum_steps, 
        loss_variant=loss_variant, 
        num_img_tokens=num_img_tokens,
        log_to_wandb=True,
        )
    
    ############################################################################
    # 4. Generation
    model.eval()
    user_prompt = 'A bathroom with a small white toilet sitting next to a walk.'
    
    kandinsky = create_generator()
    img_kandinsky = kandinsky(user_prompt)
    wandb.log({"img_kandinsky": wandb.Image(img_kandinsky)})
    
    tinygill = create_generator(model, tokenizer_extended)
    img_tinygill = tinygill(user_prompt)
    wandb.log({"img_tinygill": wandb.Image(img_tinygill)})
   
    wandb.finish()
    ############################################################################
    

    
if __name__ == "__main__":
    tinygill_run(sys.argv[1:])
    