import torch 
import torch.nn.functional as F
import diffusers 
from diffusers.optimization import get_scheduler
from transformers import ChineseCLIPProcessor, ChineseCLIPModel 
from transformers import CLIPTextModel, CLIPTokenizer 
from tqdm import tqdm 
import os
import argparse  
from utils import CrossLingualDataset



def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of stable diffusion.")
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-6,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.") 

    args = parser.parse_args()
    return args 



def main():  
    args = parse_args() 
    # hyper-param
    ckpt_path = './ckpt' 
    cn_ckpt_path = './cn_clip' 
    data_path = 'data.json' 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    train_batch_size = 1 
    gradient_accumulation_steps = 8

    # load model 
    cn_tokenizer = ChineseCLIPProcessor.from_pretrained(cn_ckpt_path).tokenizer
    cn_text_encoder = ChineseCLIPModel.from_pretrained(cn_ckpt_path) 
    cn_text_encoder = cn_text_encoder.to(device)

    en_tokenizer = CLIPTokenizer.from_pretrained(os.path.join(ckpt_path, 'tokenizer')) 
    en_text_encoder = CLIPTextModel.from_pretrained(os.path.join(ckpt_path, 'text_encoder'))
    en_text_encoder = en_text_encoder.to(device) 

    optimizer_class = torch.optim.AdamW 
    optimizer = optimizer_class(
        cn_text_encoder.parameters(), 
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    ) 

    # Dataset and DataLoaders creation
    train_dataset = CrossLingualDataset(data_path, cn_tokenizer, en_tokenizer) 
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=train_batch_size,
        shuffle=True,
    )


    for epoch in range(args.num_train_epochs): 
        cn_text_encoder.train() 
        loss_cum = 0
        iteration = 0
        with tqdm(enumerate(train_dataloader), total=len(train_dataloader)) as t:
            for step, batch in t: 
                iteration += 1 
                cn_ids, en_ids = batch 
                cn_ids, en_ids = cn_ids.to(device), en_ids.to(device) 

                with torch.no_grad(): 
                    en_embeds = en_text_encoder(en_ids,)
                cn_embeds = cn_text_encoder.get_text_features(cn_ids,)
                cn_embeds = cn_embeds / cn_embeds.norm(p=2, dim=-1, keepdim=True) 
                loss = F.mse_loss(cn_embeds.float(), en_embeds.float(), reduction="mean") 
                loss = loss / gradient_accumulation_steps 

                loss.backward() 
                if iteration % gradient_accumulation_steps == 0: 
                    torch.nn.utils.clip_grad_norm_(cn_text_encoder.parameters(), 1.0) 
                    optimizer.step()
                    optimizer.zero_grad() 
                loss_cum += loss.item() 
                t.set_description('Epoch %i' % epoch)
                t.set_postfix(loss=loss_cum / (step + 1))


if __name__ == "__main__":
    main()






