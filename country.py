import spacy

nlp = spacy.load("en_core_web_sm")

utterance = "hello visit to UK"

doc = nlp(utterance)

country = [ent.text for ent in doc.ents if ent.label_ == "GPE"]

print(country[0])