import torch
from torch.utils.data import Dataset, DataLoader
from itertools import chain
from datasets import load_dataset
from transformers import CLIPVisionModelWithProjection
from diffusers import KandinskyV22PriorPipeline


class CLIPTextEmbedsKandinskyV22(Dataset):
    def __init__(self, device='cuda'):
        self.device = device
        
        # Each COCO example has multiple captions. Stacking them all independently 
        dataset_dict = load_dataset('embedding-data/coco_captions')
        print('1/3 | COCO-captions dataset is loaded.')
        self.ds = list(chain(*dataset_dict['train']['set']))
        
        # Preparing Kandinsky 2.2 CLIP-text embedder
        image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            'kandinsky-community/kandinsky-2-2-prior',
            subfolder='image_encoder'
        ).half()
        print('2/3 | CLIPVisionModelWithProjection is loaded.')
        kandinsky_prior_pipe = KandinskyV22PriorPipeline.from_pretrained(
            'kandinsky-community/kandinsky-2-2-prior',
            image_encoder=image_encoder,
            torch_dtype=torch.float16
        ).to(self.device)
        print('3/3 | KandinskyV22PriorPipeline is loaded.')
        self.prior_pipe = kandinsky_prior_pipe

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        prompt = self.ds[idx]
    
        prompt_embeds, text_encoder_hidden_states, text_mask = self.prior_pipe._encode_prompt(
              prompt=prompt,
              device=self.device,
              num_images_per_prompt=1,
              do_classifier_free_guidance=True,
          )
        
        # Using conditional [1] embeddings only
        clip_hidden_states = text_encoder_hidden_states[1]
        clip_embed = prompt_embeds[1]

        return prompt, clip_hidden_states, clip_embed
