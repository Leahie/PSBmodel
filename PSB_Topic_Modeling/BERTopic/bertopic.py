#basic imports
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
sns.set()
import os
import sys
import tomotopy as tp
import os
import re
import pyLDAvis
import seaborn as sns
import nltk
import pickle
import gzip
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer

def get_all_files_in_directory(directory):
    file_paths = []
    timestamps = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.txt'):
                file_paths.append(os.path.join(root, file))
                timestamp = os.path.basename(root)
                timestamps.append(timestamp)
    return file_paths, timestamps
    
def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = text.lower()
    stop_words = set(["a", "an", "the", "and", "or", "but", "if", "on", "in", "to", "is", "of", "for"])
    words = [word for word in re.split(r'(\s+)', text) if word.strip() and (word in {'\n', '<br>', '<p>'} or (len(word) > 2 and word not in stop_words))]
    processed_text = ' '.join(words)
    return processed_text


main_directory_path = './PSB_Papers/main_body'

file_paths, timestamps = get_all_files_in_directory(main_directory_path)

texts = []
titles = []
for file_path in file_paths:
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
        processed_text = preprocess_text(text)
        texts.append(processed_text)
        titles.append(file_path[23:])

timestamps = pd.to_datetime(timestamps, errors='coerce')

sentence_texts = []

for text in texts:
    sentence_texts.extend(nltk.sent_tokenize(text))

sentence_texts

topic_model = BERTopic(language="english", calculate_probabilities=True, verbose=True)
topic_model.save("my_BERTopic_model_2")
topics, probs = topic_model.fit_transform(sentence_texts)

fin_df = {'Topic':[], 'Words':[], 'Scores':[]}
for key in topic_model.topic_labels_:
    fin_df['Topic'].append(f"Topic {key}")
    values = topic_model.get_topic(key)
    words, scores = "", ""
    for i in range(5):
        val = values[i]
        words += (val[0] + " ")
        scores += (str(val[1])[:4] + " ")
    print(words, scores)
    fin_df['Words'].append(words)
    fin_df['Scores'].append(scores)
print(fin_df)

topics, probs = topic_model.transform(texts)
df = {"document":[], "topic":[]}
# 'topics' will contain the topic assignment for each document
for i, topic in enumerate(topics):
    df['document'].append(titles[i])
    df["topic"].append(topic)
    print(f"Document {titles[i]} is assigned to topic {topic}")
fin_df = pd.DataFrame(data=df)
fin_df.to_csv('BERTopic_topics_documents.csv') 

topic_model.visualize_topics()

df = pd.DataFrame({"Text": texts, "Timestamp": timestamps, "Topic": topics})

topic_over_time = df.groupby(['Timestamp', 'Topic']).size().unstack(fill_value=0)
topic_over_time = topic_over_time.div(topic_over_time.sum(axis=1), axis=0)

topic_over_time = topic_over_time.sort_index()

topic_over_time_smooth = topic_over_time.rolling(window=5, min_periods=1).mean()
plt.figure(figsize=(12, 8))
for topic in topic_over_time_smooth.columns:
    plt.plot(topic_over_time_smooth.index, topic_over_time_smooth[topic], label=f'Topic {topic}')
plt.xlabel('Time Period')
plt.ylabel('Average Topic Proportion')
plt.title('Topic Frequencies Over Time (Smoothed)')
plt.legend()
plt.show()