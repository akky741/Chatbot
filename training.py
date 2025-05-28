import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
import joblib

df = pd.read_csv('updated_data.csv')

dataset = pd.concat([df,df],ignore_index=True)

X = dataset['key_words']
y = dataset['response']

vectorizer = TfidfVectorizer()
X_tfidf = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.20, random_state=42)

model = SGDClassifier(loss='log_loss')
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy:\n{accuracy*100}% is the accuracy of the chatbot")

joblib.dump(model, 'SmallTalkk.pkl')

joblib.dump(vectorizer, 'SmallTalkkVector.pkl')

loaded_model = joblib.load('SmallTalkk.pkl')