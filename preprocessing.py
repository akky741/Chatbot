from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pandas as pd
import regex as re
import nltk
import numpy as np
nltk.download("stopwords")
nltk.download("punkt")
nltk.download("wordnet")
data = pd.read_csv("data.csv",encoding="utf-8")
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_sentence(sentence):
    tokens = word_tokenize(sentence.lower())
    filtered_tokens = [lemmatizer.lemmatize(token) for token in tokens 
                       if token.isalpha() and token not in stop_words]    
    return filtered_tokens
                    
data['key_words'] = data['utterance'].apply(preprocess_sentence)

print(data.isnull().sum())

data['utterance'] = data['utterance'].astype(str)

data['key_words'] = data['key_words'].astype(str)

data['utterance'] = data['utterance'].apply(lambda x: re.sub(r'[^a-zA-Z0-9\s]', '', x))
data['key_words'] = data['key_words'].apply(lambda x: re.sub(r'[^a-zA-Z0-9\s]', '', x))

print(data.isnull().sum())
print(data['key_words'].unique())

data["key_words"].replace("",np.nan,inplace=True)

data['key_words'].fillna(data['utterance'],inplace=True)

data.to_csv("updated_data.csv",encoding="utf-8",index=False)