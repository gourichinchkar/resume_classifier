import re
import joblib
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.metrics.pairwise import cosine_similarity

model = joblib.load("model.pkl")
tfidf = joblib.load("tfidf.pkl")

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words and len(w) > 2]
    return " ".join(words)

def analyze(resume_text, jd_text):
    r_clean = clean_text(resume_text)
    j_clean = clean_text(jd_text)

    r_vec = tfidf.transform([r_clean])
    j_vec = tfidf.transform([j_clean])

    category = model.predict(r_vec)[0]
    score = cosine_similarity(r_vec, j_vec)[0][0] * 100

    # Missing keywords: JD madhle important words je resume madhe nahit
    vocab = tfidf.get_feature_names_out()
    j_weights = j_vec.toarray()[0]

    # resume che single words + 2-word phrases
    tokens = r_clean.split()
    r_ngrams = set(tokens) | {" ".join(p) for p in zip(tokens, tokens[1:])}

    # generic shabd jyanna keyword manu naye
    generic = {"knowledge", "plus", "good", "strong", "basic", "understanding",
               "familiarity", "experience", "skill", "work", "role", "team","write",
               "train","join","text","build","create","looking"}

    top_idx = j_weights.argsort()[::-1][:30]
    missing = [
        vocab[i] for i in top_idx
        if j_weights[i] > 0
        and vocab[i] not in r_ngrams
        and not any(w in generic for w in vocab[i].split())
    ]

    return category, round(score, 1), missing[:8]

if __name__ == "__main__":
    resume = """Python developer with experience in machine learning, scikit-learn,
    pandas, numpy and NLP. Built text classification and data visualization projects."""

    jd = """We are hiring a Machine Learning Engineer with Python, scikit-learn,
    TensorFlow, SQL and deep learning experience. Knowledge of NLP and model deployment is a plus."""

    category, score, missing = analyze(resume, jd)
    print("Predicted category:", category)
    print("Match score:", score, "%")
    print("Missing keywords:", missing)