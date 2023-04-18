from torch.utils.data import Dataset 
import torch 
import json 



class CrossLingualDataset(Dataset): 
    def __init__(self, data_path, cn_tokenizer, en_tokenizer):
        self.cn_tokenizer = cn_tokenizer 
        self.en_tokenizer = en_tokenizer 
        with open(data_path, 'r') as f: 
            data_list = json.load(f) 
        self.data_list = data_list 
    
    def __len__(self): 
        return len(self.data_list) 
    
    def __getitem__(self, index):
        cn_sentence = self.data_list[index][0]
        en_sentence = self.data_list[index][1] 

        cn_ids = self.cn_tokenizer(
            cn_sentence,
            padding=True, 
            return_tensors="pt"
        ).input_ids
        en_ids = self.en_tokenizer(
            en_sentence,
            padding="max_length",
            max_length=self.en_tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids

        return cn_ids, en_ids 




def get_prompt_embedding(prompt, tokenizer, text_encoder, device):
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    untruncated_ids = tokenizer(prompt, padding="longest", return_tensors="pt").input_ids
    if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
        text_input_ids, untruncated_ids
    ):
        removed_text = tokenizer.batch_decode(
            untruncated_ids[:, tokenizer.model_max_length - 1 : -1]
        )
        print(removed_text) 

    if hasattr(text_encoder.config, "use_attention_mask") and text_encoder.config.use_attention_mask:
        attention_mask = text_inputs.attention_mask.to(device)
    else:
        attention_mask = None

    prompt_embeds = text_encoder(
        text_input_ids.to(device),
        attention_mask=attention_mask,
    )
    prompt_embeds = prompt_embeds[0]
    return prompt_embeds 

