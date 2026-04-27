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

import json
from tqdm import tqdm
from torch.utils.data import Dataset


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

class SentimentDataset(Dataset):
    def __init__(self, f: str):
        self.datalist = []
        try:
            with open(f, 'r', encoding='utf-8') as file:
                for line in file:
                    try:
                        entry = json.loads(line)
                        self.datalist.append(entry)
                    except json.JSONDecodeError:
                        print(f"Error decoding line: {line}")
        except FileNotFoundError:
            print(f"File {f} not found.")

        self.data = dict()
        self.preprocess_data()

    def preprocess_data(self):
        self.data["Tweet"] = []
        self.data["Label"] = []
        self.data["Target"] = []
        self.data["Word"] = []

        for entry in tqdm(self.datalist, total=len(self.datalist), desc='Processing tweets'):
            self.data["Tweet"].append(entry["sentence"])
            self.data["Target"].append(entry["aspect"])
            self.data["Label"].append(entry["sentiment"])
            word_count = len(entry["sentence"].split())
            self.data["Word"].append(word_count)

    def __getitem__(self, index):
        item = {k: self.data[k][index] for k in self.data}
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

        return response.choices[0].message.content

def profile_analysis(tweet, prompt):
    role = "You are a user profile expert."
    instruction = f"Based on the content of the Laptop reviews: {tweet} and the user profile prompt: {prompt}, construct an English user profile. Do not include any analysis process. Simply output the constructed user profile. Refrain from any additional actions."
    return get_completion_with_role(role, instruction, tweet, prompt)

def user_analysis(tweet, target):
    role = "You are a social media analysis expert."
    instruction = f"Analyze the following Restaurant reviews: {tweet} and then provide user profile prompts for constructing a relevant user profile based on the analysis results. Do not include any analysis process. Simply output the constructed user profile prompt. Refrain from any additional actions."
    return get_completion_with_role(role, instruction, tweet, target)


def role_playing_three(tweet, target, profile):



    content = (
        f"""Task Overview
        In-depth Role-Taking:
        You are required to fully embody the user described in the following user profile. Thoroughly understand every aspect of this user, including personality traits, worldview, value system, and all other relevant details presented in the profile. These elements will serve as the core basis for all your subsequent responses.
        Simulate the user's thinking patterns, emotional experiences, and response styles in real - life scenarios as if you were that user.
        Aspect-based Sentiment Analysis:
        I will provide a sentence and a target aspect (an aspect refers to a specific object, theme, or concept within the sentence that may be subject to evaluation). Your task is to analyze the sentence from the perspective of the user in the profile and determine whether the sentiment expressed in the sentence towards the target aspect is positive, negative, or neutral.
        When making a judgment, first search the user profile for statements directly related to the sentiment towards the target aspect in the sentence. If there are no such direct statements, infer the sentiment based on the overall values, beliefs, and attitudes reflected in the profile.
        Even if the connection between the sentence and the target aspect seems weak, try your best to discern a possible sentiment tendency based on the user's character and the information in the profile.
        User Profile
        {profile}\n
        Specific Query
        Sentence: {tweet}
        Target Aspect: {target}
        Response Requirements
        Your response must be one of the following: "positive", "negative", or "neutral". Do not add any additional text, explanations, or clarifications.
        If the information in the user profile is insufficient to clearly determine the sentiment, provide the most likely sentiment judgment based on the overall tone and values presented in the profile."""
    )
    return get_completion(content)

def add_user_profile(tweet, prompt):
    profile_response = profile_analysis(tweet, prompt)

    return profile_response

def add_user_prompt(tweet, target):
    prompt_response = user_analysis(tweet, target)

    return prompt_response

def sentiment_detection(dataset):
    #results = []  # To store the results
    results = 0
    for data in tqdm(dataset, total=len(dataset), desc='Sentiment Detecting'):
        word = data['Word']
        results += word
    print(results)


    with open("sentiment_result_three_lap14.txt", "w") as file:
        for number in results:
            file.write(str(number) + "\n")

def read_txt_to_list(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
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
        if i == 'positive':
            i = 0
        elif i == 'negative':
            i = 1
        elif i == 'neutral':
            i = 2
        predictions.append(i)

    for data in tqdm(dataset, total=len(dataset)):
        if data['Label'] == 'positive':
            i = 0
        elif data['Label'] == 'negative':
            i = 1
        elif data['Label'] == 'neutral':
            i = 2
        labels.append(i)

    #print(predictions)

    #print(labels)

    f1 = f1_score(labels, predictions, average='macro')  # 对于二分类任务
    accuracy = accuracy_score(labels, predictions)
    print(f"准确率: {accuracy}")
    print(f"F1-score: {f1}")


def F1_two(file_path):
    content_list = read_txt_to_list(file_path)
    predictions = []
    labels = []
    for i in content_list:
        if i == 'positive':
            i = 0
        elif i == 'negative':
            i = 1
        else:
            i = 0
        predictions.append(i)

    for data in tqdm(dataset, total=len(dataset)):
        if data['Label'] == 'positive':
            i = 0
        elif data['Label'] == 'negative':
            i = 1
        else:
            i = 0
        labels.append(i)

    # 计算 F1-score
    f1 = f1_score(labels, predictions, average='weighted')

    accuracy = accuracy_score(labels, predictions)
    print(f"准确率: {accuracy}")

    print(f"F1-score: {f1}")

if __name__ == "__main__":
    # API key
    # Load the data

    # Replace with any model (the model after BPO).

    client =
    openai.api_key =

    dataset = SentimentDataset("./data/rest16/test.jsonl")

    sentiment_detection(dataset)

    F1_three('./sentiment_result_three_rest16.txt')




