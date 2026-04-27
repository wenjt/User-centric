import sys
import random
import torch
import numpy as np
import copy
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
from args import parse_args
from torch.quasirandom import SobolEngine
from tqdm import tqdm
import time
from botorch.models import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch import fit_gpytorch_model
from botorch.acquisition.analytic import ExpectedImprovement

from instruction_coupled_kernel import cma_es_concat
import pickle as pkl
from misc import set_all_seed, InstrutionDataset, Instruction_formatter

import openai

SMOKE_TEST = os.environ.get("SMOKE_TEST")

os.environ["CUDA_VISIBLE_DEVICES"]="3"

os.environ["OPENAI_API_KEY"] = "sk-oKh7ezw04Ku9XjFf8f8a5d1b41Ea4296B54317569e784821"
openai.base_url = "https://api.vveai.com/v1/"

tkwargs = {
    "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    "dtype": torch.double,
}

N_INIT = 5
N_ITERATIONS = 2 if not SMOKE_TEST else 1
BATCH_SIZE = 20 if not SMOKE_TEST else 1

def load_object(path):
    with open(path, 'rb') as f:
        obj = pkl.load(f)
    return obj

def save_object(obj, path):
    with open(path,'wb') as f:
        pkl.dump(obj,f)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

class LMForwardAPI:
    # model_name: The model name of the open-source LLM.
    # init_prompt: f'### Instruction:\n[INPUT]\n\n### Response:\n'
    # random_proj: The initialization of the projection matrix A.
    # intrinsic_dim: The instrinsic dimension of the projection matrix
    # n_prompt_tokens: The number of prompt tokens.
    # large_model: large_model_str
    # small_model: The model name of the open-source LLM.

    def __init__(self, model_name=None, init_prompt=None, random_proj=None, intrinsic_dim=None, n_prompt_tokens=None,
                 large_model=None, small_model=None, args=None, train_data=None):
        
        p = torch.ones(10)
        kwargs={
            'torch_dtype': torch.float16,
            'use_cache': True
            }
        self.ops_model = model_name #优化模型
        self.train_data = train_data #
        self.batch = args.batch

        self.small_model = AutoModelForCausalLM.from_pretrained(
                                small_model, trust_remote_code=True, device_map="auto"
                            ).eval()

        self.small_tokenizer = AutoTokenizer.from_pretrained(
                                small_model,
                                trust_remote_code=True, device_map="auto"
                            )

        self.embedding = self.small_model.get_input_embeddings().weight.clone()  ## shape [32000,4096]
        input_ids = self.small_tokenizer(init_prompt, return_tensors="pt").input_ids.cuda()
        self.init_prompt = self.embedding[input_ids]  ### [1,17,4096]

        ################# setup n_prompts_token #################
        self.n_prompt_tokens = n_prompt_tokens ## 5
        self.hidden_size = self.init_prompt.shape[-1] ##维度
        print('Shape of initial prompt embedding: {}'.format(self.init_prompt.shape))  ###每个维度


        # Create the template for Vicuna and WizardLM
        self.count = 0
        self.linear = torch.nn.Linear(intrinsic_dim, self.n_prompt_tokens * self.hidden_size, bias=False) ###liner 层

        self.system_prompt="You are an expert in analyzing social media behavior and constructing user profiles."

        self.role = ["### Instruction:", "### Response:"]

        if random_proj == 'normal':
            # calculate std for normal distribution
            mu_hat = np.mean(self.embedding.reshape(-1).detach().cpu().numpy())  ###所有embedding的一个均值
            std_hat = np.std(self.embedding.reshape(-1).detach().cpu().numpy())  ##方差
            mu = 0.0
            std = args.alpha * std_hat / (np.sqrt(intrinsic_dim) * args.sigma)

            print('[Embedding] mu: {} | std: {} [RandProj]  mu: {} | std: {}'.format(mu_hat, std_hat, mu, std))
            for p in self.linear.parameters():   
                torch.nn.init.uniform_(p, -1, 1)   ###参数规范化

        elif random_proj == 'uniform':
            for p in self.linear.parameters():
                torch.nn.init.uniform_(p, -1, 1)   ##从uniform distributionU(a,b)U(a,b)U(a,b)中生成值，填充输入的张量或变量。

        self.best_train_perf = 0.0
        self.best_dev_perf = 0.0
        self.best_last_perf = 10
        self.best_prompt = None
        self.num_call = 0
        self.best_instruction = None
        self.prompts_set = dict()
        self.best_acc = 0.0

    def plain_chat(self, content):
        completion = openai.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": "You are a helpful assistant."},{"role": "user", "content": content}]
        )
        return completion.choices[0].message.content

    def eval(self, prompt_embedding=None, test_data=None):
        self.num_call += 1
        if prompt_embedding is None:
            prompt_embedding = self.best_prompt
        tmp_prompt = copy.deepcopy(prompt_embedding)  # list or numpy.ndarray
        if isinstance(prompt_embedding, list):  # multiple queries
            pe_list = []
            for pe in prompt_embedding:
                z = torch.tensor(pe).type(torch.float32)  # z
                z = self.linear(z)  # Az
            prompt_embedding = torch.cat(pe_list)  # num_workers*bsz x prompt_len x dim
    
        elif isinstance(prompt_embedding, np.ndarray):  # single query or None
            prompt_embedding = torch.tensor(prompt_embedding).type(torch.float32)  # z
            prompt_embedding = self.linear(prompt_embedding)  # Az
            # if self.init_prompt is not None:
            #     prompt_embedding = prompt_embedding + self.init_prompt  # Az + p_0
            prompt_embedding = prompt_embedding.reshape(1, self.n_prompt_tokens, -1)
        elif isinstance(prompt_embedding, torch.Tensor): 
            prompt_embedding = prompt_embedding.type(torch.float32)
            prompt_embedding = self.linear(prompt_embedding)  # Az
            prompt_embedding = prompt_embedding.reshape(1, self.n_prompt_tokens, -1) ### (1,5,1536)把低维向量映射到高维
        else:
            raise ValueError(
                f'[Prompt Embedding] Only support [list, numpy.ndarray], got `{type(prompt_embedding)}` instead.'
            )

        callator = Instruction_formatter(self.small_tokenizer,max_len=512, system_prompt=self.system_prompt, prompt_embedding = prompt_embedding, embeddings=self.embedding)
        dataloader = DataLoader(dataset=self.train_data,
                                batch_size=self.batch,
                                collate_fn=callator)
        cors = []
        generation_kwargs = {
                    "min_length": 0,
                    "max_new_tokens": 512,
                    'do_sample':True
                }
        logfile = open('log.txt', 'a', encoding='utf8')
        
        for step, data in tqdm(enumerate(dataloader)):
            input_embeds = data[0]
            answers = data[1]
            querys = data[2]
            with torch.no_grad():
                outputs = self.small_model.generate(inputs_embeds=input_embeds, **generation_kwargs)
            instructions = self.small_tokenizer.batch_decode(outputs, skip_special_tokens=True)
            print(instructions)


            for query, instruction, answer in tqdm(zip(querys, instructions, answers)):
                prompt = '''Based on the content of the Tweet and the user profile prompt, your task is to analyze the sentence from the perspective of the user in the profile and determine whether the sentiment expressed in the sentence is in support of, against, or has no relation to the target object. Your answer must be one of the following: "Against", "Favor", "None". Do not include any additional text, explanations, or clarifications.\n'''
                input_text = prompt+instruction+query
                pred = self.plain_chat(input_text)
                logfile.write(f'{query}'+ '\n' + 'pred:' + f'{pred}' + '\n' + 'answer:' + f'{answer}' + '\n' + '\n')
                cor = pred == answer
                cors.append(cor)

        acc = np.mean(cors)

        print("Acc:",acc)

        if acc >= self.best_acc:
            self.count += 1
            self.best_acc = acc

        if acc >= self.best_acc:
            self.best_prompt = copy.deepcopy(tmp_prompt)
            self.prompts_set[self.best_prompt] = acc
            self.best_soft_token = prompt_embedding
        return acc

    def return_best_prompt(self):
        return self.best_prompt

    def return_best_soft_token(self):
        return self.best_soft_token

    def return_prompts_set(self):
        return self.prompts_set

    
def run(args):
    large_model, small_model = args.large_model, args.small_model   # large model, small model's name
    random_proj, intrinsic_dim, n_prompt_tokens = args.random_proj, args.intrinsic_dim, args.n_prompt_tokens  # 投影矩阵A的初始化；投影矩阵的维度；prompt tokens数量

    ### process dataset
    data_path = args.data_path

    Input_data = InstrutionDataset(data_path)

    init_prompt = f'### Instruction:\n[INPUT]:\n### Response:\n'
    model_forward_api = LMForwardAPI(model_name=args.model_name, init_prompt=init_prompt,
                                    random_proj=random_proj, intrinsic_dim=intrinsic_dim, n_prompt_tokens=n_prompt_tokens, 
                                    large_model=large_model, small_model=small_model, args=args, train_data=Input_data)

    # start bayesian opt
    X = SobolEngine(dimension=intrinsic_dim, scramble=True, seed=0).draw(N_INIT) ###(20,10)的tensor元素

    X_return = [model_forward_api.eval(x) for x in X]
    Y = [X for X in X_return]

    X = X.to(**tkwargs)
    Y = torch.FloatTensor(Y).unsqueeze(-1).to(**tkwargs)  ### Dev_perf is the performance on the development set and the instruction score means the score for the specific instruction
    print(f"Best initial point: {Y.max().item():.3f}")

    # standardization Y (no standardization for X)
    X_train = X
    y_train = (Y - Y.mean(dim=-2))/(Y.std(dim=-2))

    gp_model = SingleTaskGP(X_train, y_train)
    gp_mll = ExactMarginalLogLikelihood(gp_model.likelihood, gp_model)
    
    for i in range(N_ITERATIONS):
        print(f"X_train shape {X_train.shape}")
        print(f"y_train shape {y_train.shape}")

        start_time = time.time()

        fit_gpytorch_model(gp_mll)#, options = {'maxiter':10})
        print(f"Fitting done in {time.time()-start_time}")
        start_time = time.time()
        EI = ExpectedImprovement(gp_model, best_f = y_train.max().item())
        
        starting_idxs = torch.argsort(-1*y_train)[:BATCH_SIZE]
        starting_points = X_train[starting_idxs]

        best_points = []
        best_vals = []
        for starting_point_for_cma in starting_points:
            if (torch.max(starting_point_for_cma) > 1 or torch.min(starting_point_for_cma) < -1):
                continue
            newp, newv = cma_es_concat(starting_point_for_cma, EI, tkwargs)
            best_points.append(newp)
            best_vals.append(newv)

        print(f"best point {best_points[np.argmax(best_vals)]} \n with EI value {np.max(best_vals)}")
        print(f"Time for CMA-ES {time.time() - start_time}")
        for idx in np.argsort(-1*np.array(best_vals)):
            X_next_point =  torch.from_numpy(best_points[idx]).float().unsqueeze(0)
            # Y_next_point = [model_forward_api.eval(X_next_point)]
            
            X_next_points_return = [model_forward_api.eval(X_next_point)]
            Y_next_point = [X for X in X_next_points_return]

            X_next_point = X_next_point.to(**tkwargs)
            Y_next_point = torch.FloatTensor(Y_next_point).unsqueeze(-1).to(**tkwargs)

            X = torch.cat([X, X_next_point])
            Y = torch.cat([Y, Y_next_point])

        # standardization Y
        X_train = X.clone()
        y_train = (Y - Y.mean(dim=-2))/(Y.std(dim=-2))

        gp_model = SingleTaskGP(X_train, y_train)
        gp_mll = ExactMarginalLogLikelihood(gp_model.likelihood, gp_model)

        print(f"Best value found till now: {torch.max(Y)}")
        # print("1")

    prompts = model_forward_api.return_best_prompt()
    print("Best instruction is:")
    print(prompts)

    print("The final instruction set is:")
    print(model_forward_api.return_prompts_set())

    best_soft_token = model_forward_api.return_best_soft_token()

    save_dir = f'{args.save_dir}/{args.model_name}'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    save_object(best_soft_token,f'{args.save_dir}/{args.model_name}/{args.model_name}_{n_prompt_tokens}_{intrinsic_dim}_2711.pkl')
    return test_score

if __name__ == '__main__':
    args = parse_args()

    print(f"Using a total of {N_INIT + BATCH_SIZE * N_ITERATIONS} function evaluations") #
    print(set_all_seed(args.seed))
    test_score = run(args=args)
    print("Finished!!!")
    print(f'Test score on large model: {test_score}')