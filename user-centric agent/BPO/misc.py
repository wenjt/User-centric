import torch
import random
import numpy as np
import os
from torch.utils.data import Dataset
from tqdm import tqdm
import json
import pandas as pd
from transformers import DefaultDataCollator,PreTrainedTokenizer

def load_csv_data(file_path):
    encodings = ['utf-8', 'latin1', 'ISO-8859-1']
    for enc in encodings:
        try:
            return pd.read_csv(file_path, encoding=enc, engine='python')
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Unable to read {file_path} with any of the encodings: {', '.join(encodings)}")

class InstrutionDataset(Dataset):
    def __init__(self, f: str):
        self.datalist = load_csv_data(f)
        self.data = dict()
        # 用于存储每个 Tweet 的单词数量
        self.preprocess_data()

    def preprocess_data(self):
        self.data["Tweet"] = []
        self.data["Topic"] = []
        self.data["Stance"] = []

        for i in self.datalist.index:
            row = self.datalist.iloc[i]
            tweet = row["Tweet"]
            self.data["Tweet"].append(tweet)
            self.data["Topic"].append(row["Target"])
            self.data["Stance"].append(row["Stance"])

    def __getitem__(self, index):
        item = dict()
        for k in self.data:
            item[k] = self.data[k][index]
        return item

    def __len__(self):
        return len(self.datalist)


class Instruction_formatter:
    def __init__(self, tokenizer, max_len, system_prompt,prompt_embedding, embeddings):
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.system_prompt = system_prompt
        self.prompt_embedding = prompt_embedding
        self.embeddings = embeddings
    
    def __call__(self, examples):
        prompt = f'Instrcution: Analyze the following Tweet and then provide user profile prompts.'
        #max_len = min(self.max_len, max([len(prompt)+len(f'### Tweet: ')+len(['Tweet']) for inp in examples]))
        input_embeds = None
        answers = []
        querys = []
        for example in examples:
            #prompt = example['prompt']
            query = example['Tweet']
            query = f'### [INPUT]:{query}'
            input_query = prompt + query
            input_text=f'{input_query}\n### Response:\n'
            input_text = self.system_prompt + '\n' + input_text
            # print(input_text)
            
            input_ids = self.tokenizer(input_text, return_tensors="pt")['input_ids']
            input_embed = self.embeddings[input_ids]
            # prompt_embedding = self.prompt_embedding.to(device=input_embed.device, dtype=input_embed.dtype)
            # input_embed = torch.cat((prompt_embedding, input_embed), 1)
            if input_embeds == None:
                input_embeds = input_embed
            else:
                input_embeds = torch.cat([input_embeds, input_embed], dim=0)
            answer = example['Stance']
            answers.append(answer)
            querys.append(query)
        return input_embeds, answers, querys

def set_all_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    return f"Set all the seeds to {seed} successfully!"

def cma_es_concat(starting_point_for_cma, EI, tkwargs):
        if starting_point_for_cma.type() == 'torch.cuda.DoubleTensor':
            starting_point_for_cma = starting_point_for_cma.detach().cpu().squeeze()
        es = cma.CMAEvolutionStrategy(x0=starting_point_for_cma, sigma0=0.8, inopts={'bounds': [-1, 1], "popsize": 50},)
        iter = 1
        while not es.stop():
            iter += 1
            xs = es.ask()
            X = torch.tensor(np.array(xs)).float().unsqueeze(1).to(**tkwargs)
            with torch.no_grad():
                Y = -1 * EI(X)
            es.tell(xs, Y.cpu().numpy())  # return the result to the optimizer
            print("current best")
            print(f"{es.best.f}")
            if (iter > 10):
                break

        return es.best.x, -1 * es.best.f