"""
    generate image according to textual embedding. 
"""
import os 
import torch 
from diffusers import StableDiffusionPipeline 
from transformers import CLIPTextModel, CLIPTokenizer
from tqdm import tqdm 
from utils import get_prompt_embedding 
from transformers import ChineseCLIPProcessor, ChineseCLIPModel 


device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
ckpt_path = 'ckpt'
cn_ckpt_path = './cn_clip' 


# Load tokenizer
tokenizer = CLIPTokenizer.from_pretrained(
    os.path.join(ckpt_path, "tokenizer")
)

# Load models
text_encoder = CLIPTextModel.from_pretrained(
    os.path.join(ckpt_path, "text_encoder")
)

pipe = StableDiffusionPipeline.from_pretrained(
    ckpt_path,
    text_encoder=text_encoder,
    tokenizer=tokenizer,
).to(device) 


cn_tokenizer = ChineseCLIPProcessor.from_pretrained(cn_ckpt_path).tokenizer
cn_text_encoder = ChineseCLIPModel.from_pretrained(cn_ckpt_path) 
cn_text_encoder.load_state_dict(torch.load('out.pt')) 
cn_text_encoder = cn_text_encoder.to(device) 



prompt = 'city under the sun' 
prompt_embeds = get_prompt_embedding(prompt, tokenizer, text_encoder, device)
images = pipe(prompt_embeds=prompt_embeds, num_inference_steps=50, guidance_scale=7.5).images

images[0].save('case.png')




