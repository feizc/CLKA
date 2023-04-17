import torch 
import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DiffusionPipeline,
    DPMSolverMultistepScheduler,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from transformers import ChineseCLIPProcessor, ChineseCLIPModel 
from transformers import CLIPTextModel, CLIPTokenizer 
from tqdm import tqdm 


def main():  
    # hyper-param
    ckpt_path = './ckpt' 
    cn_ckpt_path = './cn_clip' 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load model 
    cn_tokenizer = ChineseCLIPProcessor.from_pretrained(cn_ckpt_path).tokenizer
    cn_text_encoder = ChineseCLIPModel.from_pretrained(cn_ckpt_path) 

    




if __name__ == "__main__":
    main()






