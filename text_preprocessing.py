import os
import pandas as pd
from sklearn.utils import shuffle
import numpy as np
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import Counter
from string import punctuation

import nltk
nltk.download('stopwords')
[nltk_data] Downloading package stopwords to
[nltk_data]     C:\Users\soura\AppData\Roaming\nltk_data...
[nltk_data]   Package stopwords is already up-to-date!
Out[1]:
True
Data Preprocessing
DataSets Considerd:
1. emoji
2. emotion
3. sentiment
Defining helper functions
In [6]:
def preprocess(text):
    new_text = []
    for t in text.split(" "):
        t = 'user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        t = "num" if t.isnumeric() else t
        new_text.append(t)
    return " ".join(new_text)
In [7]:
def clean_doc(doc):
    # split into tokens by white space
    rtr = []
    for tokens in doc:
        x= tokens
        tokens = preprocess(tokens)
        tokens = tokens.split()
        # remove punctuatio n from each token
        tokens = [word.lower() for word in tokens]
        table = str.maketrans('', '', punctuation)
        tokens = [w.translate(table) for w in tokens]
        # remove remaining tokens that are not alphabetic
        tokens = [word for word in tokens if word.isalpha()]
        # filter out stop words
        stop_words = set(stopwords.words('english'))
        tokens = [w for w in tokens if not w in stop_words]
        # filter out short tokens
        tokens = [word for word in tokens if len(word) > 1]
        if len(tokens) > 0:
            rtr.append(" ".join(tokens))
        else: rtr.append(x)
    return rtr
Processing files
In [8]:
# making directory to store processed data
os.mkdir("proccessed_data")
In [8]:
file_path = "datasets/"
dir_names = ["emoji","emotion","sentiment"]
text_files = ['test_text.txt','train_text.txt','val_text.txt']
label_files = ['test_labels.txt','train_labels.txt','val_labels.txt']
In [ ]:

for i in range(len(dir_names)): # iterating through choosen directories
# storing current working directory in dir_
dir_ = file_path + dir_names[i]
# Finding all the text files in cwd
file_list = os.listdir(dir_)
# Iterating through all present files in cwd
for file in file_list:
    idx=[]
    # If current file is data then we need to preprocess it
    if file in text_files:
        # storing cwd
        file_ = dir_+"/"+file
        # List to store raw data
        raw_text = []
        #opening text file 
        with open(file_,encoding="utf8") as fd:
            # iterating through text
            for line in fd:
                raw_text.append(line.strip())
        # Preproccessing the data
        processed_text = clean_doc(raw_text)
        #print(len(processed_text),len(raw_text))
        # path to store proccessed data
        path_ = "proccessed_data/"+dir_names[i]

        # path to stored proccessed data
        csv_file = path_+"/"+file[:-4]+".csv"
        # if proccessed_data directory is present
        if os.path.exists(path_):
            # saving file as csv file
            pd.DataFrame(processed_text,columns=["text"]).to_csv(csv_file,index=None)

        else:
            # if path is not present create a directory
            os.mkdir(path_)
            # saving csv file
            pd.DataFrame(processed_text,columns=["text"]).to_csv(csv_file,index=None)

    # if current file is labels we need not to proces it
    elif file in label_files:
        file_ = dir_+"/"+file
        labels = []
        with open(file_,encoding="utf8") as fd:
            for line in fd:
                labels.append(line.strip())
        path_ = "proccessed_data/"+dir_names[i]
        csv_file = path_+"/"+file[:-4]+".csv"
        if os.path.exists(path_):
            labels = pd.DataFrame(labels,columns=["text"])
            labels.to_csv(csv_file,index=None)

        else :
            os.mkdir(path_)
            labels = pd.DataFrame(labels,columns=["text"])
            labels.to_csv(csv_file,index=None)
