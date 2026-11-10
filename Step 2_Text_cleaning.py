import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download("stopwords")
nltk.download("wordnet")

df = pd.read_csv("resume_dataset.csv")

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)   # links kadha
    text = re.sub(r"[^a-z\s]", " ", text)         # numbers/symbols kadha
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words and len(w) > 2]
    return " ".join(words)

df["Cleaned"] = df["Resume"].apply(clean_text)

print("Before:", df["Resume"][0][:200])
print("After:", df["Cleaned"][0][:200])

df.to_csv("cleaned_resumes.csv", index=False)