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
import pandas as pd
from sklearn.metrics import f1_score
import os
from torch.utils.data import Dataset
from sklearn.metrics import f1_score, accuracy_score


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

class HateDataset(Dataset):
    def __init__(self, f: str):

        with open(f, 'r', encoding='utf-8') as file:
            self.datalist = json.load(file)

        self.datalist = self.datalist['test']

        self.data = dict()
        self.preprocess_data()

    def preprocess_data(self):
        self.data["Tweet"] = []
        self.data["Label"] = []
        self.data["Target"] = []
        self.data["Word"] = []

        for entry in tqdm(self.datalist, total=len(self.datalist), desc='Processing tweets'):
            #print(entry['text'])
            # 遍历每个子列表
            sentence = ' '.join(entry['text'])
            # 将合并后的句子添加到sentences列表中
            self.data["Tweet"].append(sentence)
            self.data["Target"].append(entry["final_target_category"])
            self.data["Label"].append(entry["final_label"])
            self.data["Label"].append(entry["final_label"])
            word_count = len(sentence.split())
            # 将单词数量添加到 self.data["Word"] 列表中
            self.data["Word"].append(word_count)

    def __getitem__(self, index):
        item = dict()
        for k in self.data:
            item[k] = self.data[k][index]
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

def get_completion_with_role(instruction, tweet, target):
    max_retries = 10
    for i in range(max_retries):
        #try:
        messages = [
                {"role": "system", "content": f"{instruction}"},
                {"role": "user", "content": f"tweet:{tweet};target:{target}\n"}
        ]

        # response = client.chat.completions.create(
        #     model="glm-4-Air",
        #     messages=messages,
        #     temperature=0
        # )

        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=messages,
            temperature=0
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
            temperature=0
        )

        return response.choices[0].message.content

def get_completion(instruction, content):
    max_retries = 100000

    for i in range(max_retries):
        messages = [
            {"role": "system", "content": f"{instruction}"},
            {"role": "user", "content": f"{content}\n"}
        ]

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

def profile_analysis(tweet, target):
    instruction = "You are an expert in building user profiles; please construct an english user profile based on the content and target of the tweet. Do nothing else."
    return get_completion_with_role(instruction, tweet, target)

def user_analysis(tweet, target):
    role = "You are a social media analysis expert."
    instruction = f"Analyze the following Tweet: {tweet} and then provide user profile prompts for constructing a relevant user profile based on the analysis results."
    return get_completion_with_role(role, instruction, tweet, target)

def role_playing_two(tweet, target, profile):
    instruction = f"You are asked to play the user according to the following user profile\n<<<{profile}>>>"

    input = (f"Hello, please answer me a question, "
             f"you tweeted for {target}'s {tweet} would like to do an interview with you, "
             f"is there toxicity in what you are trying to convey? "
             f"Please just answer me toxic or non-toxic, no extra output is needed!"
             f"You must answer either toxic or non toxic!!!!"
             f"Here are a few examples for your reference"
             f"u really think i would not have been raped by feral hindu or muslim back in india or bangladesh and a neo nazi would rape me as well just to see me cry"
             f"answer is toxic"
             f"i live and work with many legal mexican immigrants who are great citizens and trump supporters they have no problem with deporting illegals maga"
             f"answer is non-toxic")

    return get_completion(instruction, input)

def role_playing_three(tweet, target, profile):
    # instruction = f"You are asked to play the user according to the following user profile\n<<<{profile}>>>"
    instruction = f"You are now a person who fits the following user profile: {profile}. Please answer the question according to the characteristics of this role."

    input = (f"Hello, please answer me a question, "
             f"you tweeted for {target}'s {tweet} would like to do an interview with you, "
             f"is there toxicity in what you are trying to convey?"
             f"Your answer must be one of the following options: offensive, hatespeech or normal. Absolutely no additional text, explanations, or clarifications are allowed."
             f"Ensure that your response strictly aligns with the user as outlined in the profile. Any deviation from the user's established position will render the answer inaccurate."
             )

    return get_completion(instruction, input)


def add_user_profile(tweet, target):
    profile_response = profile_analysis(tweet, target)

    return profile_response

def hate_speech_detection(dataset):
    results = []  # To store the results

    # for data in tqdm(dataset, total=len(dataset), desc='Hate Speech Detecting'):
    #     tweet = data['Tweet']
    #     target = data['Target']
    #     profile = add_user_profile(tweet, target)
    #     final_response = role_playing_three(tweet, target, profile).lower()
    #     results.append(final_response)
    results = 0
    for data in tqdm(dataset, total=len(dataset), desc='Hate Speech Detecting'):
        word = data['Word']
        results += word

    print(results)

    # with open("result_three_0_10.txt", "w") as file:
    #     # 遍历列表中的每个元素
    #     for number in results:
    #         # 将元素转换为字符串（尽管在这种情况下Python会自动转换）
    #         # 并将其写入文件，每个元素后添加一个换行符
    #         file.write(str(number) + "\n")

#
    #for idx, res in enumerate(temps):
        #for key, value in res.items():
            #new_data.at[idx, key] = value

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
        if i == 'offensive':
            i = 0
        elif i == 'hatespeech':
            i = 1
        elif i == 'normal':
            i = 2
        else:
            i = 2
        predictions.append(i)

    for data in tqdm(subset_dataset, total=len(subset_dataset)):
        if data['Label'] == 'offensive':
            i = 0
        elif data['Label'] == 'hatespeech':
            i = 1
        elif data['Label'] == 'normal':
            i = 2
        labels.append(i)

    # 计算 F1-score
    f1 = f1_score(labels, predictions, average='weighted')  # 对于二分类任务
    accuracy = accuracy_score(labels, predictions)
    print(f"F1-score: {f1} \n")
    print(f"Acc: {accuracy}")


def F1_two(file_path):
    content_list = read_txt_to_list(file_path)
    predictions = []
    labels = []
    for i in content_list:
        if i == 'toxic':
            i = 0
        elif i == 'non-toxic':
            i = 1
        else:
            i = 0
        predictions.append(i)

    for data in tqdm(subset_dataset, total=len(subset_dataset)):
        if data['Label'] == 'toxic':
            i = 0
        elif data['Label'] == 'non-toxic':
            i = 1
        else:
            i = 0
        labels.append(i)

    # 计算 F1-score

    accuracy = accuracy_score(labels, predictions)

    f1 = f1_score(labels, predictions, average='weighted')  # 对于二分类任务
    print(f"F1-score: {f1} \n")
    print(f"Acc: {accuracy}")


if __name__ == "__main__":
    # API key
    # Load the data
    client = 'openai'
    openai.api_key =
    dataset = HateDataset("./data/hateXplain/hatexplain_thr_div.json")

    hate_speech_detection(dataset)





