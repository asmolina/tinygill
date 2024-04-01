import torch
from tqdm.autonotebook import tqdm
import wandb
from utils import add_img_postfix, get_expanded_ids


def train_epoch(
    train_loader, 
    model, 
    tokenizer,
    img_tokens_vocab_idx,
    loss_fn,
    scheduler,
    optimizer,
    accum_steps=15, 
    loss_variant='promt_emb_only', 
    num_img_tokens=4,
    log_to_wandb=True,
):
    tqdm_params = dict(colour='CYAN', leave='False', total=len(train_loader))
    for i, batch_data in tqdm(enumerate(train_loader), **tqdm_params):
        prompt, clip_hidden_states, clip_emb = batch_data
        
        # Add [IMG1], ..., [IMGr] tokens to the prompt
        prompt_postfixed = add_img_postfix(prompt, num_img_tokens=num_img_tokens)
        input_ids = get_expanded_ids(prompt_postfixed, tokenizer)
        
        embeds, prompt_emb = model(input_ids)
        
        if loss_variant == 'both':
            loss_hidden_states = loss_fn(embeds, clip_hidden_states)
            loss_promt_emb = loss_fn(prompt_emb, clip_emb)
            loss = loss_hidden_states + loss_promt_emb
        elif loss_variant == 'promt_emb_only':
            loss_promt_emb = loss_fn(prompt_emb, clip_emb)
            loss = loss_promt_emb
        elif loss_variant == 'hidden_states_only':
            loss_hidden_states = loss_fn(embeds, clip_hidden_states)
            loss = loss_hidden_states
        else:
            NotImplementedError
                
        loss /= accum_steps
        loss.backward()
        
        if (i + 1) % accum_steps == 0:
            # Zero out gradients of the embedding matrix outside of [IMG] tokens
            for param in model.lm_input_embeddings.parameters():
                assert param.grad.shape[0] == len(tokenizer)
                mask = torch.zeros((param.grad.shape[0], 1)).to(param.grad)
                for gen_idx in img_tokens_vocab_idx:
                    mask[gen_idx] = 1
                param.grad *= mask
            optimizer.step()
            optimizer.zero_grad()
    
        scheduler.step()
        
        if log_to_wandb:
            if loss_variant == 'both':
                wandb.log({'loss_promt_emb': loss_promt_emb.item()})
                wandb.log({'loss_hidden_states': loss_hidden_states.item()})
                wandb.log({'loss': loss.item() * accum_steps})
            elif loss_variant == 'promt_emb_only':
                wandb.log({'loss_promt_emb': loss.item() * accum_steps})
            elif loss_variant == 'hidden_states_only':
                wandb.log({'loss_hidden_states': loss.item() * accum_steps})
            
            last_lr = scheduler.get_last_lr()[0]
            wandb.log({'lr': last_lr})
            print(f'\r Last loss: {loss.item() * accum_steps}, Last lr: {last_lr}', end=" ", flush=True)

            with torch.no_grad():
                cos = torch.nn.CosineSimilarity(dim=-1, eps=1e-08)
                l2 = torch.nn.MSELoss()
                wandb.log({'l2_hidden_states': l2(embeds[0], clip_hidden_states[0]).item()})
                wandb.log({'cos_hidden_states': cos(embeds[0], clip_hidden_states[0])})
                wandb.log({'l2_promt_emb': l2(prompt_emb[0], clip_emb[0]).item()})
                wandb.log({'cos_promt_emb': cos(prompt_emb[0], clip_emb[0])})