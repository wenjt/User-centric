import copy
import csv
import openai
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
import random
import time
import logging
from zhipuai import ZhipuAI
from sklearn.metrics import f1_score
import os
from torch.utils.data import Dataset
from sklearn.metrics import f1_score, accuracy_score

import httpx
from httpx_socks import SyncProxyTransport

transport = SyncProxyTransport.from_url("socks5://127.0.0.1:7891")

http_client = httpx.Client(transport=transport)

openai.httpx_client = http_client

os.environ["http_proxy"] = "http://localhost:7890"
os.environ["https_proxy"] = "http://localhost:7890"

def load_csv_data(file_path):
    encodings = ['utf-8', 'latin1', 'ISO-8859-1']
    for enc in encodings:
        try:
            return pd.read_csv(file_path, encoding=enc, engine='python')
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Unable to read {file_path} with any of the encodings: {', '.join(encodings)}")

from torch.utils.data import Dataset
import pandas as pd

def load_csv_data(f):
    return pd.read_csv(f)

class StanceDataset(Dataset):
    def __init__(self, f: str):
        self.datalist = load_csv_data(f)
        self.data = dict()
        # 用于存储每个 Tweet 的单词数量
        self.word_counts = []
        self.preprocess_data()

    def preprocess_data(self):
        self.data["Tweet"] = []
        self.data["Topic"] = []
        self.data["Stance"] = []
        self.data["Word"] = []

        for i in self.datalist.index:
            row = self.datalist.iloc[i]
            tweet = row["Tweet"]
            self.data["Tweet"].append(tweet)
            self.data["Topic"].append(row["Target"])
            self.data["Stance"].append(row["Stance"])
            # 统计单词数量，简单以空格分割字符串
            word_count = len(tweet.split())
            self.data["Word"].append(word_count)
            self.word_counts.append(word_count)

    def __getitem__(self, index):
        item = dict()
        for k in self.data:
            item[k] = self.data[k][index]
        # 将单词数量添加到返回的样本中
        item["WordCount"] = self.word_counts[index]
        return item

    def __len__(self):
        return len(self.datalist)

class SubsetDataset(Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)

def get_completion_with_role(role, instruction, tweet, target):
    max_retries = 10
    for i in range(max_retries):
        #try:
        messages = [
                {"role": "system", "content": f"{role}"},
                {"role": "user", "content": f"{instruction}"}
        ]

        # response = client.chat.completions.create(
        #     model="glm-4-Air",
        #     messages=messages,
        #     temperature=0
        # )

        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=messages,
            temperature=0,
        )

        #print(openai.error.Timeout)
        return response.choices[0].message.content


def get_completion_with_role1(role, instruction, tweet, target, a):
    max_retries = 10
    for i in range(max_retries):
        #try:
        messages = [
                {"role": "system", "content": f"{role}"},
                {"role": "user", "content": f"{instruction}"}
        ]

        # response = client.chat.completions.create(
        #     model="glm-4-Air",
        #     messages=messages,
        #     temperature=0
        # )

        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=messages,
            temperature=0,
        )

        #print(openai.error.Timeout)
        return response.choices[0].message.content

def get_completion_with_topic(instruction, content):
    max_retries = 10
    for i in range(max_retries):
        # try:
        messages = [
            {"role": "system", "content": f"{instruction}"},
            {"role": "user", "content": f"{content}\n"}
        ]

        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=messages,
            temperature=0,
        )

        return response.choices[0].message.content

def get_completion(content):
    max_retries = 100000

    #client = OpenAI(api_key="<DeepSeek API Key>", base_url="https://api.deepseek.com")


    for i in range(max_retries):
        messages = [{"role": "user", "content": content}]

        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=messages,
            temperature=0
        )

        # response = openai.ChatCompletion.create(
        #         model="gpt-3.5-turbo",
        #         messages=messages,
        #         temperature=0
        # )```````````````````````````````````````````
        return response.choices[0].message.content

def profile_analysis(tweet, prompt):
    role = "You are a user profile expert."
    instruction = f"Based on the content of the Tweet: {tweet} and the user profile prompt: {prompt}, construct an English user profile. Do not include any analysis process. Simply output the constructed user profile. Refrain from any additional actions."
    return get_completion_with_role(role, instruction, tweet, prompt)

def profile_analysis1(tweet, prompt1, profile):
    role = "You are a user profile expert."
    instruction = f"Based on the content of the Tweet: {tweet}, the naive uesr profile: {prompt1} and the opinion-aware user discirbe: {profile}, construct an English user profile. Do not include any analysis process. Simply output the constructed user profile. Refrain from any additional actions."
    return get_completion_with_role1(role, instruction, tweet, prompt1, profile)

def user_analysis(tweet, target):
    role = "You are a social media analysis expert."
    instruction = f"Analyze the following Tweet {tweet} and then provide user profile prompts. Do not include any analysis process. The uesr profile prompts are designed to guide the creation of user profiles that prioritize capturing user opinions. Simply output the constructed user profile prompt. Refrain from any additional actions."
    return get_completion_with_role(role, instruction, tweet, target)

def role_playing_three(tweet, target, profile):


    content = (
        f"""### Task Overview
1. **User Personification**: 
- You are to completely step into the role of the user in the contextual scenario described in the following user profile. Absorb every aspect of the user, including their personality traits, world - view, value system, and any other relevant details. These aspects will be the foundation for all your subsequent responses.
- Try to think, feel, and respond as if you were this user in real - life situations.
2. **Stance Detection**: 
- I will provide you with a sentence and a target object. Your task is to analyze the sentence from the perspective of the user in the profile and determine whether the sentiment expressed in the sentence is in support of, against, or has no relation to the target object.
- When making a judgment, first look for direct statements in the user profile that align with or oppose the stance in the sentence. If there are no direct statements, consider the user's general values, beliefs, and attitudes in the profile to infer a stance.
- Even if the relation between the sentence and the target object seems weak, make an effort to discern a possible leaning based on the user's character.
### User Profile
{profile}
### Contextual Scenario 
{'''Scenario Reconstruction

Background: The setting is an online campaign celebrating the empowerment of young girls and women, spearheaded by actress Maisie Williams. Known for her role as Arya Stark on the popular TV series "Game of Thrones," Maisie uses her platform to advocate for gender equality and challenge stereotypes through the #LikeAGirl campaign.

Task Description: The task involves promoting and supporting the #LikeAGirl campaign, which aims to redefine what it means to do something "like a girl" by showcasing strength, resilience, and leadership. Participants are encouraged to share their personal stories, engage in discussions, and use social media to amplify the campaign's message.

Event Description: The campaign gains momentum on social media, where fans and activists alike rally behind Maisie Williams, expressing admiration for her leadership qualities and the way she embodies the strength of her on-screen character, Arya Stark.
The hashtag #GoT (Game of Thrones) is used alongside #SemST to connect fans of the show with the broader empowerment movement.

Event Process: Throughout the campaign, social media platforms buzz with activity as people post videos, photos, and stories illustrating moments of empowerment and breaking stereotypes.
Testimonials and messages of support flood in from the global community, highlighting instances where individuals have excelled in traditionally male-dominated fields or defied societal expectations.
Maisie Williams actively participates in the dialogue by sharing her own experiences and encouraging others to contribute.
Online discussions are held to brainstorm further actions and collaborations that could extend the impact of the campaign beyond social media.'''}
### Specific Query
- **Sentence**: {tweet}
- **Target Object**: {target}"""
    )
    return get_completion(content)

def add_user_profile(tweet, prompt):
    profile_response = profile_analysis(tweet, prompt)

    return profile_response

def add_user_prompt(tweet, target):
    prompt_response = user_analysis(tweet, target)

    return prompt_response

def final_user_profile(tweet, profile1, profile):
    profile_response = profile_analysis1(tweet, profile1, profile)

    return profile_response

def stance_detection(dataset):
    results = []  # To store the results
    profiles = []

    for data in tqdm(dataset, total=len(dataset), desc='Stance Detecting'):
        tweet = data['Tweet']
        target = data['Topic']
        #profile_prompt = ''
        profile_prompt1 = ''
        profile1 = add_user_profile(tweet, profile_prompt1)
        profile_prompt = add_user_prompt(tweet, target)
        #print("用户画像提示：", profile_prompt, "\n")
        profile = add_user_profile(tweet, profile_prompt)

        profile2 = final_user_profile(tweet, profile1, profile)


        profiles.append(profile2)

    with open('profile423.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        # 写入列名
        writer.writerow(['profile'])
        # 逐行写入数据（每个元素单独成行）
        writer.writerows([[item] for item in profiles])



def word(dataset):
    results = 0  # To store the results

    for data in tqdm(dataset, total=len(dataset), desc='Stance Detecting'):
        word = data['Word']
        results += word

    print(results)

def read_txt_to_list(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            # 读取所有行并保存到列表中
            lines = file.readlines()
            # 去除每行末尾的换行符（可选）
            lines = [line.strip() for line in lines]
        return lines
    except FileNotFoundError:
        print(f"文件 {file_path} 未找到。")
        return []
    except Exception as e:
        print(f"读取文件时发生错误: {e}")
        return []

def F1_three(file_path):
    content_list = read_txt_to_list(file_path)
    predictions = []
    labels = []
    for i in content_list:
        if i == 'AGAINST':
            i = 0
        elif i == 'NONE':
            i = 1
        elif i == 'FAVOR':
            i = 2
        predictions.append(i)

    for data in tqdm(subset_dataset, total=len(subset_dataset)):
        if data['Stance'] == 'AGAINST':
            i = 0
        elif data['Stance'] == 'NONE':
            i = 1
        elif data['Stance'] == 'FAVOR':
            i = 2
        labels.append(i)

    f1 = f1_score(labels, predictions, average='weighted')  # 对于二分类任务
    accuracy = accuracy_score(labels, predictions)
    print(f"准确率: {accuracy}")
    print(f"F1-score: {f1}")


if __name__ == "__main__":
    # API key
    # Load the data
    client =
    openai.api_key = ''
    dataset = StanceDataset("./data/Sem16/test.csv")

    indices = list(range(500, 501))

    subset_dataset = SubsetDataset(dataset, indices)

    stance_detection(subset_dataset)

