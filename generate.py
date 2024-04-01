import torch
from torch import nn
from transformers import CLIPVisionModelWithProjection
from diffusers import KandinskyV22Pipeline, KandinskyV22PriorPipeline
from diffusers.models import UNet2DConditionModel
from collections import namedtuple

from utils import add_img_postfix


class FakeTextEncoder(nn.Module):
    def __init__(self, model):
        super().__init__()

        self.model = model

        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, input_ids):
        Out = namedtuple('Output', ['last_hidden_state', 'text_embeds'])

        return Out(*self.model(input_ids))
    
    
def create_generator(model=None, tokenizer=None):
    def generator(prompt):
        DEVICE = torch.device('cuda:0')

        image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            'kandinsky-community/kandinsky-2-2-prior',
            subfolder='image_encoder'
        ).half().to(DEVICE)

        prior_pipe = KandinskyV22PriorPipeline.from_pretrained(
            'kandinsky-community/kandinsky-2-2-prior',
            image_encoder=image_encoder,
            torch_dtype=torch.float16
        ).to(DEVICE)

        unet = UNet2DConditionModel.from_pretrained(
            'kandinsky-community/kandinsky-2-2-decoder',
            subfolder='unet'
        ).half().to(DEVICE)
        
        decoder = KandinskyV22Pipeline.from_pretrained(
            'kandinsky-community/kandinsky-2-2-decoder',
            unet=unet,
            torch_dtype=torch.float16
        ).to(DEVICE)
         
        # The generator will remain as the Kandinsky 2.2 if tokenizer and model are not specified
        if tokenizer is not None:
            prior_pipe.tokenizer = tokenizer
            
        if model is not None:
            prior_pipe.text_encoder = FakeTextEncoder(model).to(DEVICE)
        
        prompt_postfixed = add_img_postfix(prompt)
        img_emb = prior_pipe(
            prompt=prompt_postfixed,
            num_inference_steps=25,
            num_images_per_prompt=1
        )

        neg_prompt = 'lowres, worst quality, low quality, jpeg artifacts, ugly, out of frame'
        neg_prompt_postfixed = add_img_postfix(neg_prompt)
        negative_emb = prior_pipe(
            prompt=neg_prompt_postfixed,
            num_inference_steps=25,
            num_images_per_prompt=1
        )

        images = decoder(
            image_embeds=img_emb.image_embeds,
            negative_image_embeds=negative_emb.image_embeds,
            num_inference_steps=75,
            height=512,
            width=512)
        
        return images.images[0]
    return generator