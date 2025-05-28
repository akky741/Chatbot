
import joblib
import gradio as gr
import warnings
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pandas as pd
import regex as re
import nltk
import spacy
import en_core_web_sm
# nlp = spacy.load("en_core_web_sm")

nlp = en_core_web_sm.load()

nltk.download("stopwords")
nltk.download("punkt")
nltk.download("wordnet")
data = pd.read_csv("data.csv",encoding="utf-8")
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

warnings.filterwarnings("ignore")

def clean_utterance(utterance):
    cleaned_utterance = re.sub(r'[^a-zA-Z0-9\s]', '', utterance)
    return cleaned_utterance

def preprocess_utterance(utterance):
    tokens = word_tokenize(utterance.lower())
    filtered_tokens = [lemmatizer.lemmatize(token) for token in tokens if token.isalpha() and token not in stop_words]
    return ' '.join(filtered_tokens)

def Predicttalkk(utterance):
    try:
        keywords = preprocess_utterance(utterance)
        cleaned_utterance = clean_utterance(keywords)
        print(cleaned_utterance)
        st_model = "SmallTalkk.pkl"
        st_vector = "SmallTalkkVector.pkl"
        loaded_model = joblib.load(st_model)
        small_talk_vectorizer = joblib.load(st_vector)
        new_utterances = [utterance]
        new_utterances_tfidf = small_talk_vectorizer.transform(new_utterances)
        prediction_probabilities = loaded_model.predict_proba(new_utterances_tfidf)
        predictions = loaded_model.predict(new_utterances_tfidf)
        max_confidence = -1  
        predicted_class = None 

        for i in range(len(predictions)):
            for j, class_prob in enumerate(prediction_probabilities[i]):
                class_name = loaded_model.classes_[j]
                confidence = class_prob * 100
                if confidence > max_confidence:
                    max_confidence = confidence
                    predicted_class = class_name

        return {"response":predicted_class, "confidence":max_confidence}
    except:
        print("Sorry our services are currently unavailable")
        return "Sorry our services are currently unavailable"

def answer(utterance):
    prediction = Predicttalkk(utterance)
    if prediction['response'] == "It's Narendra Modi" and prediction['confidence'] > 75:
        doc = nlp(utterance)
        country = [ent.text for ent in doc.ents if ent.label_ == "GPE"]
        try:
            if country[0]== "India":
                return prediction["response"]
            else:
                msg = "My knowledge is only limited to India"
                return msg
        except:
            msg = "My knowledge is only limited to India"
            return msg      
              
    elif prediction['confidence'] > 75:
        return prediction['response']
    else:
        msg = "Sorry I don't have answer to the question..!! :("
        return msg
  
# while True:
#     print(answer(input("user: ")))

iface = gr.Interface(fn=answer, inputs="text", outputs="text")
iface.launch()