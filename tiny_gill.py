import torch
from torch import nn
from typing import List


class TinyGILL(nn.Module):
    def __init__(
        self,
        llm,
        img_tokens_idx: List[int],
        in_dim: int = 2048,
        out_dim: int = 1280,
        pad_token_id: int = 2, 
        projection_method: str = 'sum',
        device: str = 'cuda',
    ):
        super().__init__()
        self.device = device
        self.img_tokens_idx = img_tokens_idx
        self.pad_token_id = pad_token_id
        
        self.lm = llm
        self.lm_input_embeddings = self.lm.get_input_embeddings() # A torch.module mapping vocabulary ids to hidden states
        
        self.gill_mapper = GILLMapper(in_dim, out_dim)
        
        self.projection_method = projection_method
        if self.projection_method == 'CLIP-like':
            self.projection = nn.Linear(out_dim, out_dim, bias=False)
        

    def forward(self, ids: List[int]) -> torch.Tensor:
        # Boolean mask with True at (IMG1, ... IMGr) tokens positions [bs, padded_len]
        img_tokens_mask = sum(ids==i for i in self.img_tokens_idx).bool() 
        
        # Obtain raw_tokens_embs and lm_last_hidden_states for all tokens [bs, padded_len, hidden_size_lm]
        raw_embs = self.lm_input_embeddings(ids.to(self.device))
        last_hidden_states = self.lm(inputs_embeds=raw_embs, output_hidden_states=True).hidden_states[-1]
        
        # Obtain raw_tokens_embs and lm_last_hidden_states for IMG tokens only [bs, num_IMG_tokens, hidden_size_lm]
        img_tokens_raw_embs = torch.stack([
            batch_embs[batch_mask, :] for batch_embs, batch_mask in zip(raw_embs, img_tokens_mask)
        ]) 
        img_tokens_hid_states = torch.stack([
            batch_embs[batch_mask, :] for batch_embs, batch_mask in zip(last_hidden_states, img_tokens_mask)
        ]) 
        
        # Obtain features for image generation [bs, num_queries, 1280]
        embeds = self.gill_mapper(img_tokens_raw_embs + img_tokens_hid_states) 
        
        # Obtain prompt_embedding [bs, 1280]
        if self.projection_method == 'CLIP-like':
            pooled_out = embeds[
                torch.arange(embeds.shape[0], device=self.device),
                (ids.to(dtype=torch.int, device=embeds.device) == self.pad_token_id).int().argmax(dim=-1),
            ]
            prompt_embedding = self.projection(pooled_out)
        elif self.projection_method == 'sum':
            prompt_embedding = embeds.sum(axis=1)
        else:
            raise NotImplementedError
            
        return embeds, prompt_embedding # [bs, num_queries, 1280], [bs, 1280]


class GILLMapper(nn.Module):
    def __init__(self, 
                 in_dim: int, 
                 out_dim: int, 
                 num_output_tokens: int = 77
                ):
        super().__init__()

        self.num_output_tokens = num_output_tokens

        hidden_dim = 512
        self.fc = nn.Linear(in_dim, hidden_dim)
        self.tfm = nn.Transformer(
            batch_first=True, 
            norm_first=True,
            d_model=hidden_dim, 
            num_encoder_layers=4, 
            num_decoder_layers=4,
            dim_feedforward=hidden_dim * 4, 
            dropout=0.0, 
            nhead=4
        )
        self.model = nn.Linear(hidden_dim, out_dim)
        self.query_embs = nn.Parameter(torch.randn(1, num_output_tokens, hidden_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        queries = self.query_embs.repeat(x.shape[0], 1, 1)
        x = self.tfm(x, queries)
        outputs = self.model(x)

        return outputs # [bs, num_output_tokens, out_dim]
    